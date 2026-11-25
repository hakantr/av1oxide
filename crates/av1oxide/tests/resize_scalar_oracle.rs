//! S2c.3 superres büyütme çekirdeği oracle'ı (Rust tarafı).
//!
//! `scripts/oracle/run_resize_oracle.sh` dav1d C `resize_c` ile durum dosyası
//! ve çağrı başına SHA-256 listesi üretir (`resize-<küme>-cases.bin`,
//! `-expect.bin`). Bu test aynı durumlardan aynı girdiyi yeniden üretir
//! (`resize_oracle.c` `sm64`/`pix` ile birebir), [`av1oxide::dsp::resize::scalar_for`]
//! ile koşar ve `resize-<küme>-rust.bin` yazar; betik iki listeyi karşılaştırır.
//! Kaynak satırlarının `src_w`'den sonraki dolgusu C'dekinden **farklı** bir
//! değerle doldurulur ve son satır dilimi tam `src_w` pikselde biter: kenar
//! ötesi okuma SHA farkı ya da panik üretir.
//!
//! Differential betik kipindedir (`#[ignore]`; betik `--include-ignored` ve
//! `RESIZE_ORACLE_OUT` / `RESIZE_ORACLE_SET` ile koşturur). Tablo testi ortam
//! değişkeni yoksa varsayılan `.cache/resize-oracle` dökümünü okur; döküm yoksa
//! ön koşul eksiktir (`on_kosul::eksik`, P29) — sessizce geçmez.

mod common;

use av1oxide::dsp::resize::{ResizeBitDepth, scalar_for};
use av1oxide::dsp::resize_tables::RESIZE_FILTER;
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

/// AVX2 büyütme yuvaları: P33 8 bpc, P34 16 bpc (AVX2 yoksa `None`; skaler
/// karşılaştırma yine koşar).
fn avx2_yuvasi() -> Option<[av1oxide::dsp::resize::ResizeFn; 2]> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        return Some([
            av1oxide::dsp::x86::buyutme::resize_fn(t),
            av1oxide::dsp::x86::buyutme16::resize_fn(t),
        ]);
    }
    None
}

/// P35: SSSE3 büyütme yuvaları (8 bpc, 16 bpc; SSSE3 yoksa `None`).
fn ssse3_yuvasi() -> Option<[av1oxide::dsp::resize::ResizeFn; 2]> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).ssse3_token()
    {
        return Some([
            av1oxide::dsp::x86::buyutme_ssse3::resize_fn(t),
            av1oxide::dsp::x86::buyutme16_ssse3::resize_fn(t),
        ]);
    }
    None
}

/// P36: AVX512ICL büyütme yuvaları (8 bpc, 16 bpc; AVX512ICL yoksa `None`).
fn avx512_yuvasi() -> Option<[av1oxide::dsp::resize::ResizeFn; 2]> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx512icl_token()
    {
        return Some([
            av1oxide::dsp::x86::buyutme_avx512::resize_fn(t),
            av1oxide::dsp::x86::buyutme16_avx512::resize_fn(t),
        ]);
    }
    None
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
// Girdi üreteci — `scripts/oracle/resize_oracle.c` `sm64`/`pix` ile birebir.
// ---------------------------------------------------------------------------

fn sm64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn pix(pattern: u32, x: i32, y: i32, w: i32, r: u64, max: i32) -> i32 {
    match pattern {
        0 => 0,
        1 => max,
        2 => {
            if (x ^ y) & 1 != 0 {
                max
            } else {
                0
            }
        }
        3 => (r & max as u64) as i32,
        4 => (x * 37 + y * 11) % (max + 1),
        5 => {
            if r & 1 != 0 {
                max
            } else {
                0
            }
        }
        _ => {
            if w > 1 {
                x * max / (w - 1)
            } else {
                max
            }
        }
    }
}

const FIELDS: usize = 12;
const KINDS: [&str; 3] = ["luma", "kroma", "rastgele"];

/// Tablo dökümünün dizini: `RESIZE_ORACLE_OUT` ya da varsayılan `.cache/resize-oracle`.
fn tablo_dizini() -> PathBuf {
    std::env::var_os("RESIZE_ORACLE_OUT").map_or_else(
        || {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../.cache/resize-oracle"
            ))
        },
        PathBuf::from,
    )
}

fn oracle_dir() -> Option<(PathBuf, String)> {
    let dir = std::env::var_os("RESIZE_ORACLE_OUT")?;
    let set = std::env::var("RESIZE_ORACLE_SET").ok()?;
    Some((PathBuf::from(dir), set))
}

#[test]
fn sha256_sabit_vektor() {
    assert_eq!(
        hex(&sha(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

/// `RESIZE_FILTER` dav1d `dav1d_resize_filter` dökümüyle bayt bayt aynı; her
/// satırın toplamı −128.
#[test]
fn resize_tablosu_dav1d_ile_bayt_esit() {
    for row in RESIZE_FILTER {
        assert_eq!(row.iter().map(|&v| i32::from(v)).sum::<i32>(), -128);
    }
    let yol = tablo_dizini().join("resize-tables.bin");
    let Ok(c) = std::fs::read(&yol) else {
        on_kosul::eksik(
            "resize_scalar_oracle::resize_tablosu_dav1d_ile_bayt_esit",
            Neden::OracleVerisiYok,
            "dav1d tablo dökümü yok; üretmek için: bash scripts/oracle/run_resize_oracle.sh",
        );
        return;
    };
    let rust: Vec<u8> = RESIZE_FILTER.iter().flatten().map(|&v| v as u8).collect();
    assert_eq!(c, rust, "dav1d_resize_filter farklı");
    println!("[resize tablo] 64×8 = {} bayt dav1d = Rust", rust.len());
}

#[test]
#[ignore = "betik kipi: scripts/oracle/run_resize_oracle.sh koşturur (RESIZE_ORACLE_OUT/RESIZE_ORACLE_SET)"]
fn resize_oracle_differential() {
    let Some((dir, set)) = oracle_dir() else {
        on_kosul::eksik(
            "resize_scalar_oracle::resize_oracle_differential",
            Neden::OracleVerisiYok,
            "RESIZE_ORACLE_OUT/RESIZE_ORACLE_SET yok (betik kipi)",
        );
        return;
    };
    let cases = std::fs::read(dir.join(format!("resize-{set}-cases.bin"))).expect("cases");
    assert_eq!(cases.len() % (FIELDS * 4), 0);
    let mut out = Vec::with_capacity(cases.len() / (FIELDS * 4) * 32);
    let mut cell: Vec<Vec<Sha256>> = (0..3)
        .map(|_| (0..3).map(|_| Sha256::new()).collect())
        .collect();
    let mut cell_n = [[0u32; 3]; 3];
    let mut calls = 0usize;
    // P33/P34: AVX2, P35: SSSE3 ve P36: AVX512ICL yuvaları (8 bpc ve 16 bpc) her durumda skalerle
    // bayt bayt aynı olmalı (skaler çıktı betikte dav1d C ile karşılaştırılır).
    let isa = [
        ("AVX2", avx2_yuvasi()),
        ("SSSE3", ssse3_yuvasi()),
        ("AVX512ICL", avx512_yuvasi()),
    ];
    // [ISA][8 bpc, 16 bpc] çağrı sayısı
    let mut isa_cagri = [[0usize; 2]; 3];
    for rec in cases.chunks(FIELDS * 4) {
        let f: Vec<u32> = rec
            .chunks(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (bdv, src_w, dst_w, h) = (f[0], f[1] as usize, f[2] as usize, f[3] as usize);
        let (dx, mx0) = (f[4] as i32, f[5] as i32);
        let (pattern, ssp, dsp_px, kind) = (f[6], f[7] as usize, f[8] as usize, f[11] as usize);
        let seed = u64::from(f[9]) | (u64::from(f[10]) << 32);
        let bd = match bdv {
            8 => ResizeBitDepth::B8,
            10 => ResizeBitDepth::B10,
            _ => ResizeBitDepth::B12,
        };
        let bpp = bd.bytes();
        let max = bd.bitdepth_max();
        let (ss, ds) = (ssp * bpp, dsp_px * bpp);
        // C dolgusu 0xA5; Rust dolgusu 0x3C (kenar ötesi okuma SHA'yı bozar).
        let mut src = vec![0x3Cu8; (h - 1) * ss + src_w * bpp];
        let mut s = seed;
        for y in 0..h {
            for x in 0..src_w {
                let v = pix(pattern, x as i32, y as i32, src_w as i32, sm64(&mut s), max);
                let o = y * ss + x * bpp;
                if bpp == 1 {
                    src[o] = v as u8;
                } else {
                    src[o..o + 2].copy_from_slice(&(v as u16).to_le_bytes());
                }
            }
        }
        let mut dst = vec![0u8; (h - 1) * ds + dst_w * bpp];
        (scalar_for(bd))(&mut dst, ds, &src, ss, dst_w, h, src_w, dx, mx0, bd);
        for (i, (ad, yuva)) in isa.iter().enumerate() {
            if let Some(yuvalar) = yuva {
                let b16 = usize::from(bd != ResizeBitDepth::B8);
                let mut d2 = vec![0u8; dst.len()];
                (yuvalar[b16])(&mut d2, ds, &src, ss, dst_w, h, src_w, dx, mx0, bd);
                assert_eq!(
                    d2, dst,
                    "{ad} ≠ skaler: {bdv} bit src_w={src_w} dst_w={dst_w} h={h} dx={dx} mx0={mx0}"
                );
                isa_cagri[i][b16] += 1;
            }
        }
        let mut hsh = Sha256::new();
        for y in 0..h {
            hsh.update(&dst[y * ds..y * ds + dst_w * bpp]);
        }
        let d = hsh.finish();
        out.extend_from_slice(&d);
        let bi = match bdv {
            8 => 0,
            10 => 1,
            _ => 2,
        };
        cell[bi][kind].update(&d);
        cell_n[bi][kind] += 1;
        calls += 1;
    }
    std::fs::write(dir.join(format!("resize-{set}-rust.bin")), &out).expect("rust.bin");
    let mut txt = String::new();
    for (bi, bdv) in [8u32, 10, 12].iter().enumerate() {
        for (t, name) in KINDS.iter().enumerate() {
            let c = std::mem::replace(&mut cell[bi][t], Sha256::new());
            let _ = writeln!(txt, "{name} {bdv} {} {}", cell_n[bi][t], hex(&c.finish()));
        }
    }
    std::fs::write(dir.join(format!("resize-{set}-cells-rust.txt")), txt).expect("cells");
    println!(
        "[resize {set}] {calls} çağrı, çıktı SHA-256 {} · 8 bpc AVX2 {} / SSSE3 {} / AVX512ICL {} çağrı, 16 bpc AVX2 {} / SSSE3 {} / AVX512ICL {} çağrı skalerle aynı",
        hex(&sha(&out)),
        isa_cagri[0][0],
        isa_cagri[1][0],
        isa_cagri[2][0],
        isa_cagri[0][1],
        isa_cagri[1][1],
        isa_cagri[2][1]
    );
}
