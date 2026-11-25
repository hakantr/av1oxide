//! S2c.2b — loop restoration differential'ı: skaler Rust ↔ dav1d C (bit-exact).
//!
//! Oracle, dav1d `src/looprestoration_tmpl.c`'yi (BITDEPTH 8/16, **asm yok**,
//! kaynak değiştirilmeden) ayrı bir süreçte derler
//! (`scripts/oracle/lr_oracle.c`) ve çağrı başına çıktı SHA-256'sını yazar.
//! Bu test aynı kayıtlardan aynı tamponları üretir (`fill`, C `lr_fill` ile
//! birebir), üretim crate'inin skaler çekirdeklerini koşturur ve özetleri
//! karşılaştırır; üretim crate'i oracle'a **bağlanmaz** (SRC-11).
//!
//! | Küme | Çağrı | Rust profili | Karşılaştırma |
//! |---|---|---|---|
//! | `quick` (kapı) | 10 080 | test (taşma denetimi açık) | çağrı SHA-256 + hücre SHA-256 |
//! | `full` | 114 000 | `robust` (release + taşma + `debug_assert`) | çağrı SHA-256 + hücre SHA-256 |
//!
//! **Sözleşme sıkılığı:** Rust çağrısına verilen dilimler bellek sözleşmesinin
//! (`dsp::lr` başlığı) **tam** sınırındadır (`dst`, `left`, `lpf` uzunlukları
//! en küçük değer; `HAVE_LEFT` yoksa `left` boş, `HAVE_TOP/BOTTOM` yoksa
//! `lpf` boş). Sözleşmenin okunmasına izin vermediği her bayt (kullanılmayan
//! `lpf` satırları/sütunları, `left[r][0]`, `dst` satır sonları) C'dekinden
//! **farklı** rastgele "zehir" ile doldurulur: Rust o baytı okursa çıktı
//! C'den sapar; yazarsa zehir bozulur. İkisi de FAIL'dir. Böylece 114 000
//! çağrının her biri aynı zamanda bir taşma (over-read/over-write) testidir.
//!
//! Yeniden üretim (tek komut): `bash scripts/oracle/run_lr_oracle.sh [--quick]`

#![forbid(unsafe_code)]

use av1oxide::dsp::lr::{LrBitDepth, LrDsp, LrEdgeFlags, LrParams};
use av1oxide::dsp::lr_tables::{SGR_PARAMS, SGR_X_BY_X};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

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
// Tampon üreteci — `scripts/oracle/lr_oracle.h` `lr_sm64`/`lr_pix`/`lr_fill`.
// ---------------------------------------------------------------------------

fn sm64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn pix(pattern: u8, x: i32, y: i32, pl: u32, r: u64, max: i32) -> i32 {
    let u = (r >> 32) as u32;
    let m1 = max as u32 + 1;
    match pattern {
        0 => 0,
        1 => max,
        2 => {
            if (x + y) & 1 != 0 {
                max
            } else {
                0
            }
        }
        3 => (u % m1) as i32,
        4 => {
            if u & 1 != 0 {
                max
            } else {
                0
            }
        }
        5 => (max >> 1) + (u % 33) as i32 - 16,
        6 => {
            ((((x + 1024) as u32)
                .wrapping_mul(37)
                .wrapping_add(((y + 1024) as u32).wrapping_mul(101))
                .wrapping_add(pl.wrapping_mul(53)))
                % m1) as i32
        }
        _ => {
            if (((x + 1024) >> 2) ^ ((y + 1024) >> 1)) & 1 != 0 {
                max
            } else {
                (u % m1) as i32
            }
        }
    }
}

fn store(buf: &mut [u8], off: usize, bpp: usize, v: i32) {
    buf[off] = (v & 255) as u8;
    if bpp == 2 {
        buf[off + 1] = ((v >> 8) & 255) as u8;
    }
}

/// C yerleşimindeki üç tampon (`lr_fill`).
struct CBufs {
    dst: Vec<u8>,
    left: Vec<u8>,
    lpf: Vec<u8>,
}

fn fill(c: &Case) -> CBufs {
    let bpp = c.bpp();
    let max = (1i32 << c.bd) - 1;
    let (w, h, sp, lx) = (c.w, c.h, c.stride_px, c.lpf_x);
    let _ = w;
    let mut st = u64::from(c.seed);
    let mut dst = vec![0u8; h * sp * bpp];
    let mut left = vec![0u8; h * 4 * bpp];
    let mut lpf = vec![0u8; 8 * sp * bpp];
    for y in 0..h {
        for x in 0..sp {
            let r = sm64(&mut st);
            store(
                &mut dst,
                (y * sp + x) * bpp,
                bpp,
                pix(c.pattern, x as i32, y as i32, 0, r, max),
            );
        }
    }
    for y in 0..h {
        for k in 0..4 {
            let r = sm64(&mut st);
            store(
                &mut left,
                (y * 4 + k) * bpp,
                bpp,
                pix(c.pattern, k as i32 - 4, y as i32, 1, r, max),
            );
        }
    }
    for row in 0..8usize {
        let ly = if row < 2 {
            row as i32 - 2
        } else if row >= 6 {
            (h + row - 6) as i32
        } else {
            1000 + row as i32
        };
        for col in 0..sp {
            let r = sm64(&mut st);
            store(
                &mut lpf,
                (row * sp + col) * bpp,
                bpp,
                pix(c.pattern, col as i32 - lx as i32, ly, 2, r, max),
            );
        }
    }
    CBufs { dst, left, lpf }
}

// ---------------------------------------------------------------------------
// Kayıtlar — `lr_oracle.h` biçimi.
// ---------------------------------------------------------------------------

const FUNC_NAMES: [&str; 5] = ["wiener7", "wiener5", "sgr5x5", "sgr3x3", "sgrmix"];
const REC_BYTES: usize = 64;

#[derive(Clone, Debug)]
struct Case {
    func: usize,
    bd: u32,
    edges: u8,
    pattern: u8,
    w: usize,
    h: usize,
    stride_px: usize,
    lpf_x: usize,
    seed: u32,
    filter: [[i16; 8]; 2],
    s0: u32,
    s1: u32,
    w0: i32,
    w1: i32,
}

impl Case {
    fn parse(r: &[u8]) -> Self {
        let u16_at = |o: usize| usize::from(u16::from_le_bytes([r[o], r[o + 1]]));
        let u32_at = |o: usize| u32::from_le_bytes([r[o], r[o + 1], r[o + 2], r[o + 3]]);
        let mut filter = [[0i16; 8]; 2];
        for (i, row) in filter.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                let o = 16 + (i * 8 + j) * 2;
                *v = i16::from_le_bytes([r[o], r[o + 1]]);
            }
        }
        Self {
            func: usize::from(r[0]),
            bd: u32::from(r[1]),
            edges: r[2],
            pattern: r[3],
            w: u16_at(4),
            h: u16_at(6),
            stride_px: u16_at(8),
            lpf_x: u16_at(10),
            seed: u32_at(12),
            filter,
            s0: u32_at(48),
            s1: u32_at(52),
            w0: u32_at(56) as i32,
            w1: u32_at(60) as i32,
        }
    }

    fn bpp(&self) -> usize {
        if self.bd > 8 { 2 } else { 1 }
    }

    fn bit_depth(&self) -> LrBitDepth {
        match self.bd {
            8 => LrBitDepth::B8,
            10 => LrBitDepth::B10,
            _ => LrBitDepth::B12,
        }
    }

    fn params(&self) -> LrParams {
        if self.func < 2 {
            LrParams::Wiener {
                filter: self.filter,
            }
        } else {
            LrParams::Sgr {
                s0: self.s0,
                s1: self.s1,
                w0: self.w0,
                w1: self.w1,
            }
        }
    }

    /// Çekirdeğin kenar ötesi uzanımı (3×3 → 2, diğerleri → 3).
    fn reach(&self) -> usize {
        if self.func == 3 { 2 } else { 3 }
    }
}

// ---------------------------------------------------------------------------
// Sözleşme-sıkı Rust çağrısı.
// ---------------------------------------------------------------------------

/// Zehir üreteci (C tamponlarından bağımsız).
struct Poison(u64);

impl Poison {
    fn byte(&mut self) -> u8 {
        (sm64(&mut self.0) >> 56) as u8
    }
}

/// Rust çağrısını sözleşmenin tam sınırındaki dilimlerle yapar; C yerleşimindeki
/// çıktı `dst` tamponunu döndürür. Sözleşme dışı okuma/yazma `Err` üretir.
fn run_rust(c: &Case, cb: &CBufs) -> Result<Vec<u8>, String> {
    let bpp = c.bpp();
    let edges = LrEdgeFlags::from_bits(c.edges);
    let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
    let have_right = edges.contains(LrEdgeFlags::HAVE_RIGHT);
    let have_top = edges.contains(LrEdgeFlags::HAVE_TOP);
    let have_bottom = edges.contains(LrEdgeFlags::HAVE_BOTTOM);
    let reach = c.reach();
    let lreach = if have_left { reach } else { 0 };
    let rreach = if have_right { reach } else { 0 };
    let (w, h) = (c.w, c.h);
    let csp = c.stride_px * bpp;
    let mut poison = Poison(u64::from(c.seed) ^ 0xa5a5_5a5a_dead_beef);
    let mut vs = u64::from(c.seed).wrapping_mul(31);
    let var = sm64(&mut vs);

    // dst: Rust satır adımı farklı; okunabilir aralık [0, w + rreach) kopyalanır.
    let dpad = (var & 3) as usize;
    let ds = (w + rreach + dpad) * bpp;
    let dst_len = (h - 1) * ds + (w + rreach) * bpp;
    let mut dst = vec![0u8; h * ds];
    for b in dst.iter_mut() {
        *b = poison.byte();
    }
    for y in 0..h {
        let n = (w + rreach) * bpp;
        dst[y * ds..y * ds + n].copy_from_slice(&cb.dst[y * csp..y * csp + n]);
    }
    dst.truncate(dst_len);
    let dst_before = dst.clone();

    // left: yalnız HAVE_LEFT; okunmayan pikseller (k < 4 − reach) zehirli.
    let mut left = Vec::new();
    if have_left {
        left = cb.left.clone();
        for y in 0..h {
            for k in 0..4 - reach {
                for b in 0..bpp {
                    left[(y * 4 + k) * bpp + b] = poison.byte();
                }
            }
        }
    }

    // lpf: Rust satır adımı ve lpf_x farklı; yalnız bayraklı satırların
    // okunabilir sütunları [−lreach, w + rreach) kopyalanır, kalan her şey zehir.
    let lx = lreach + ((var >> 2) & 3) as usize;
    let lpad = ((var >> 4) & 7) as usize;
    let ls = (lx + w + rreach + lpad) * bpp;
    let last_row = if have_bottom {
        Some(7usize)
    } else if have_top {
        Some(1)
    } else {
        None
    };
    let mut lpf = Vec::new();
    if let Some(last) = last_row {
        lpf = vec![0u8; 8 * ls];
        for b in lpf.iter_mut() {
            *b = poison.byte();
        }
        let rows: &[usize] = match (have_top, have_bottom) {
            (true, true) => &[0, 1, 6, 7],
            (true, false) => &[0, 1],
            (false, true) => &[6, 7],
            (false, false) => &[],
        };
        for &r in rows {
            let n = (lreach + w + rreach) * bpp;
            let src = r * csp + (c.lpf_x - lreach) * bpp;
            let dsto = r * ls + (lx - lreach) * bpp;
            lpf[dsto..dsto + n].copy_from_slice(&cb.lpf[src..src + n]);
        }
        lpf.truncate(last * ls + (lx + w + rreach) * bpp);
    }

    let dsp = LrDsp::scalar_for(c.bit_depth());
    let kernel = match c.func {
        0 => dsp.wiener[0],
        1 => dsp.wiener[1],
        2 => dsp.sgr[0],
        3 => dsp.sgr[1],
        _ => dsp.sgr[2],
    };
    let params = c.params();
    kernel(
        &mut dst,
        ds,
        &left,
        &lpf,
        ls,
        lx,
        w,
        h,
        &params,
        edges,
        c.bit_depth(),
    );

    // Yazma denetimi: [0, w) dışı hiçbir bayt değişmemeli.
    for y in 0..h {
        let row_end = if y + 1 == h { dst_len } else { (y + 1) * ds };
        for o in y * ds + w * bpp..row_end {
            if dst[o] != dst_before[o] {
                return Err(format!(
                    "sözleşme dışı yazma: satır {y}, bayt {}",
                    o - y * ds
                ));
            }
        }
    }

    // C yerleşimine geri koy: C yalnız [0, w) yazar.
    let mut out = cb.dst.clone();
    for y in 0..h {
        let n = w * bpp;
        out[y * csp..y * csp + n].copy_from_slice(&dst[y * ds..y * ds + n]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Testler
// ---------------------------------------------------------------------------

fn oracle_dir() -> Option<PathBuf> {
    let dir = std::env::var("LR_ORACLE_OUT").ok()?;
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    Some(base.join(dir))
}

#[test]
fn sha256_sabit_vektor() {
    assert_eq!(
        hex(&sha(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let long = vec![0x61u8; 1000];
    let mut s = Sha256::new();
    s.update(&long[..3]);
    s.update(&long[3..200]);
    s.update(&long[200..]);
    assert_eq!(s.finish(), sha(&long));
}

#[test]
fn lr_tablolar_dav1d_ile_bayt_esit() {
    let Some(dir) = oracle_dir() else {
        return;
    };
    let raw = std::fs::read(dir.join("lr-tables.bin")).expect("lr-tables.bin (run_lr_oracle.sh)");
    assert_eq!(&raw[..8], b"AV1OLRT1");
    let mut mine = Vec::new();
    for p in SGR_PARAMS {
        for v in p {
            mine.extend_from_slice(&v.to_le_bytes());
        }
    }
    mine.extend_from_slice(&SGR_X_BY_X);
    assert_eq!(&raw[8..], &mine[..], "SGR tabloları dav1d ile aynı değil");
    println!("\n[lr tablolar] {} bayt dav1d = Rust", mine.len());
}

#[test]
fn lr_oracle_differential() {
    let Some(dir) = oracle_dir() else {
        return;
    };
    let set = std::env::var("LR_ORACLE_SET").unwrap_or_else(|_| "quick".to_owned());
    let cases = std::fs::read(dir.join(format!("lr-{set}-cases.bin"))).expect("cases");
    let expect = std::fs::read(dir.join(format!("lr-{set}-expect.bin"))).expect("expect");
    assert_eq!(&cases[..8], b"AV1OLRC1");
    assert_eq!(&expect[..8], b"AV1OLRE1");
    let n = u32::from_le_bytes([cases[12], cases[13], cases[14], cases[15]]) as usize;
    let ne = u32::from_le_bytes([expect[8], expect[9], expect[10], expect[11]]) as usize;
    assert_eq!(n, ne, "kayıt sayısı");
    assert_eq!(cases.len(), 16 + n * REC_BYTES);
    assert_eq!(expect.len(), 12 + n * 32);

    let mut out = Vec::with_capacity(12 + n * 32);
    out.extend_from_slice(b"AV1OLRE1");
    out.extend_from_slice(&(n as u32).to_le_bytes());
    let mut cell_in: Vec<Sha256> = (0..15).map(|_| Sha256::new()).collect();
    let mut cell_out: Vec<Sha256> = (0..15).map(|_| Sha256::new()).collect();
    let mut cell_calls = [0u32; 15];
    let mut farklar = 0usize;
    let mut ihlal = 0usize;
    for i in 0..n {
        let c = Case::parse(&cases[16 + i * REC_BYTES..16 + (i + 1) * REC_BYTES]);
        let slot = c.func * 3
            + match c.bd {
                8 => 0,
                10 => 1,
                _ => 2,
            };
        let cb = fill(&c);
        cell_in[slot].update(&cb.dst);
        cell_in[slot].update(&cb.left);
        cell_in[slot].update(&cb.lpf);
        let digest = match run_rust(&c, &cb) {
            Ok(buf) => sha(&buf),
            Err(e) => {
                ihlal += 1;
                if ihlal <= 10 {
                    println!("\n[lr FARK] #{i} {} {c:?}: {e}", FUNC_NAMES[c.func]);
                }
                [0u8; 32]
            }
        };
        let exp = &expect[12 + i * 32..12 + (i + 1) * 32];
        if digest != exp {
            farklar += 1;
            if farklar <= 10 {
                println!(
                    "\n[lr FARK] #{i} {} bd={} edges={:04b} w={} h={} pattern={} filter={:?} s=({},{}) w=({},{})",
                    FUNC_NAMES[c.func],
                    c.bd,
                    c.edges,
                    c.w,
                    c.h,
                    c.pattern,
                    c.filter,
                    c.s0,
                    c.s1,
                    c.w0,
                    c.w1
                );
            }
        }
        cell_out[slot].update(&digest);
        cell_calls[slot] += 1;
        out.extend_from_slice(&digest);
    }
    std::fs::write(dir.join(format!("lr-{set}-rust.bin")), &out).expect("rust.bin");
    let mut txt = String::new();
    for (slot, (ci, co)) in cell_in.into_iter().zip(cell_out).enumerate() {
        let bd = [8, 10, 12][slot % 3];
        let _ = writeln!(
            txt,
            "{} {bd} {} {} {}",
            FUNC_NAMES[slot / 3],
            cell_calls[slot],
            hex(&ci.finish()),
            hex(&co.finish())
        );
    }
    std::fs::write(dir.join(format!("lr-{set}-cells-rust.txt")), &txt).expect("cells");
    println!(
        "\n[lr {set}] {n} çağrı, {farklar} SHA farkı, {ihlal} sözleşme ihlali, {} hücre (5 işlev × 3 bd), çıktı SHA-256 {}",
        cell_calls.iter().filter(|&&k| k > 0).count(),
        hex(&sha(&out))
    );
    assert_eq!(ihlal, 0, "sözleşme dışı yazma");
    assert_eq!(farklar, 0, "dav1d C ile bit-exact değil");
}
