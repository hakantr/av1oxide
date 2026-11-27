//! P54 (`F5d` — `F5-LF` + `F5-LR`) `aarch64` NEON loop filter (`8 / 8` matris kaydı = `12`
//! yuva @ `8 / 10 / 12` bpc) ve loop restoration (`10 / 10` matris kaydı = `12` yuva:
//! `wiener[0..1]` @ `8 / 10 / 12` bpc + `sgr[0..2]` @ `8 / 10` bpc; `12`-bpc SGR skaler)
//! C oracle eşitliği, `SentinelBuffer`, `OverReadProbe` ve sözleşme dışı skaler devir testleri.

#![cfg_attr(
    not(target_arch = "aarch64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::lf::{LimLut, LoopFilterDsp, lf_skaler_devir_sayisi};
use av1oxide::dsp::lr::{
    LrBitDepth, LrDsp, LrEdgeFlags, LrFilterFn, LrParams, lr_skaler_devir_sayisi,
};
use av1oxide::dsp::lr_tables::SGR_PARAMS;
use av1oxide::dsp::{ARM_ISA_CANDIDATES, DspDispatch};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

const LF_FIELDS: [&str; 4] = [
    "loop_filter_sb[0][0]",
    "loop_filter_sb[0][1]",
    "loop_filter_sb[1][0]",
    "loop_filter_sb[1][1]",
];

const LR_FIELDS: [&str; 5] = ["wiener[0]", "wiener[1]", "sgr[0]", "sgr[1]", "sgr[2]"];

// ---------------------------------------------------------------------------
// SHA-256 (LR C oracle karşılaştırması için)
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

fn sha(data: &[u8]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(data);
    s.finish()
}

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

fn store_px(buf: &mut [u8], off: usize, bpp: usize, v: i32) {
    buf[off] = (v & 255) as u8;
    if bpp == 2 {
        buf[off + 1] = ((v >> 8) & 255) as u8;
    }
}

struct CBufs {
    dst: Vec<u8>,
    left: Vec<u8>,
    lpf: Vec<u8>,
}

const LR_REC_BYTES: usize = 64;

#[derive(Clone, Debug)]
struct LrCase {
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

impl LrCase {
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

    fn reach(&self) -> usize {
        if self.func == 3 { 2 } else { 3 }
    }
}

fn fill_lr_case(c: &LrCase) -> CBufs {
    let bpp = c.bpp();
    let max = (1i32 << c.bd) - 1;
    let (h, sp, lx) = (c.h, c.stride_px, c.lpf_x);
    let mut st = u64::from(c.seed);
    let mut dst = vec![0u8; h * sp * bpp];
    let mut left = vec![0u8; h * 4 * bpp];
    let mut lpf = vec![0u8; 8 * sp * bpp];
    for y in 0..h {
        for x in 0..sp {
            let r = sm64(&mut st);
            store_px(
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
            store_px(
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
            store_px(
                &mut lpf,
                (row * sp + col) * bpp,
                bpp,
                pix(c.pattern, col as i32 - lx as i32, ly, 2, r, max),
            );
        }
    }
    CBufs { dst, left, lpf }
}

fn run_lr_kernel_on_case(kernel: LrFilterFn, c: &LrCase, cb: &CBufs) -> Result<Vec<u8>, String> {
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
    let mut ps = u64::from(c.seed) ^ 0xa5a5_5a5a_dead_beef;
    let mut poison = || (sm64(&mut ps) >> 56) as u8;
    let mut vs = u64::from(c.seed).wrapping_mul(31);
    let var = sm64(&mut vs);

    let dpad = (var & 3) as usize;
    let ds = (w + rreach + dpad) * bpp;
    let dst_len = (h - 1) * ds + (w + rreach) * bpp;
    let mut dst = vec![0u8; h * ds];
    for b in &mut dst {
        *b = poison();
    }
    for y in 0..h {
        let n = (w + rreach) * bpp;
        dst[y * ds..y * ds + n].copy_from_slice(&cb.dst[y * csp..y * csp + n]);
    }
    dst.truncate(dst_len);
    let dst_before = dst.clone();

    let mut left = Vec::new();
    if have_left {
        left = cb.left.clone();
        for y in 0..h {
            for k in 0..4 - reach {
                for b in 0..bpp {
                    left[(y * 4 + k) * bpp + b] = poison();
                }
            }
        }
    }

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
        for b in &mut lpf {
            *b = poison();
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

    let mut sb = SentinelBuffer::with_defaults(dst_len);
    sb.data_mut().copy_from_slice(&dst);
    let params = c.params();
    kernel(
        sb.data_mut(),
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
    sb.verify()
        .map_err(|v| format!("SentinelBuffer ihlali: {v:?}"))?;
    let res_dst = sb.data();

    for y in 0..h {
        let row_end = if y + 1 == h { dst_len } else { (y + 1) * ds };
        for o in y * ds + w * bpp..row_end {
            if res_dst[o] != dst_before[o] {
                return Err(format!(
                    "sözleşme dışı yazma: satır {y}, bayt {}",
                    o - y * ds
                ));
            }
        }
    }

    let mut out = cb.dst.clone();
    for y in 0..h {
        let n = w * bpp;
        out[y * csp..y * csp + n].copy_from_slice(&res_dst[y * ds..y * ds + n]);
    }
    Ok(out)
}

fn oracle_lut() -> LimLut {
    let mut lut = LimLut::default();
    for level in 0..64i32 {
        let sharp = level & 7;
        let mut limit = level;
        if sharp > 0 {
            limit >>= (sharp + 3) >> 2;
            limit = limit.min(9 - sharp);
        }
        limit = limit.max(1);
        lut.i[level as usize] = limit;
        lut.e[level as usize] = 2 * (level + 2) + limit;
    }
    lut
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn f5d_lf_ve_lr_arm64_aday_ve_kurulum() {
    let lf_adaylar: Vec<_> = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "loopfilter")
        .collect();
    assert_eq!(lf_adaylar.len(), 12);
    for bd in [8u8, 10, 12] {
        for field in LF_FIELDS {
            assert!(
                lf_adaylar.iter().any(|c| c.bit_depth == bd
                    && c.field == field
                    && c.implementation == "neon"
                    && c.requires == CpuFlags::NEON
                    && c.default),
                "eksik ARM64 LF adayı: loopfilter/{field}@{bd}"
            );
        }
    }

    let lr_adaylar: Vec<_> = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "looprestoration")
        .collect();
    assert_eq!(lr_adaylar.len(), 15);

    let d_none = DspDispatch::select(CpuLevel::None.allowed());
    for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
        for field in LF_FIELDS {
            assert_eq!(
                d_none.slot_implementation(bd, "loopfilter", field),
                "skaler"
            );
        }
        for field in LR_FIELDS {
            assert_eq!(
                d_none.slot_implementation(bd, "looprestoration", field),
                "skaler"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let d_neon = DspDispatch::select(CpuFlags::detect());
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            for field in LF_FIELDS {
                assert_eq!(d_neon.slot_implementation(bd, "loopfilter", field), "neon");
            }
            for field in LR_FIELDS {
                assert_eq!(
                    d_neon.slot_implementation(bd, "looprestoration", field),
                    "neon",
                    "bd={bd:?} {field}"
                );
            }
        }
    }
}

fn lf_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("loopfilter")
}

fn lr_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("lr")
}

fn find_lf_oracle_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("LOOPFILTER_ORACLE_OUT") {
        let p = PathBuf::from(d);
        if p.join("loopfilter-input.bin").is_file() && p.join("loopfilter-expect.bin").is_file() {
            return Some(p);
        }
    }
    let local = repo_root().join(".cache/loopfilter-oracle");
    if local.join("loopfilter-input.bin").is_file() && local.join("loopfilter-expect.bin").is_file()
    {
        return Some(local);
    }
    None
}

fn find_lr_oracle_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("LR_ORACLE_OUT") {
        let p = PathBuf::from(d);
        if p.join("lr-quick-cases.bin").is_file() && p.join("lr-quick-expect.bin").is_file() {
            return Some(p);
        }
    }
    let local = repo_root().join(".cache/lr-oracle");
    if local.join("lr-quick-cases.bin").is_file() && local.join("lr-quick-expect.bin").is_file() {
        return Some(local);
    }
    None
}

#[cfg(target_arch = "aarch64")]
fn verify_lf_oracle_files(
    lf_dsps: &[LoopFilterDsp; 3],
    input_path: &Path,
    expect_path: &Path,
    min_per_bd: usize,
) {
    let input = std::fs::read(input_path)
        .unwrap_or_else(|e| panic!("{} okunamadı: {e}", input_path.display()));
    let expect = std::fs::read(expect_path)
        .unwrap_or_else(|e| panic!("{} okunamadı: {e}", expect_path.display()));
    assert_eq!(input.len(), expect.len());
    let lim = oracle_lut();
    let mut p = 0usize;
    let mut verified = [0usize; 3];

    while p + 12 <= input.len() {
        let h0u = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
        let h1u = u32::from_le_bytes([input[p + 4], input[p + 5], input[p + 6], input[p + 7]]);
        let stride_bytes = usize::from(u16::from_le_bytes([input[p + 8], input[p + 9]]));
        let offp = usize::from(u16::from_le_bytes([input[p + 10], input[p + 11]]));
        p += 12;

        let bd = (h0u & 0xff) as usize;
        let func = ((h0u >> 8) & 0xff) as usize;
        let n = (h1u >> 16) as usize;

        let mut vmask = [0u32; 4];
        for v in &mut vmask {
            *v = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
            p += 4;
        }
        let lsize = usize::from(u16::from_le_bytes([input[p], input[p + 1]]));
        p += 2;
        let lb = &input[p..p + lsize];
        p += lsize;

        let bpp = if bd > 8 { 2 } else { 1 };
        let plane_len = 64 * 64 * bpp;
        let in_plane = &input[p..p + plane_len];
        let exp_plane = &expect[p..p + plane_len];
        p += plane_len;

        let (dsp, bi): (&LoopFilterDsp, usize) = match bd {
            8 => (&lf_dsps[0], 0),
            10 => (&lf_dsps[1], 1),
            12 => (&lf_dsps[2], 2),
            _ => continue,
        };

        let plane_idx = usize::from(func >= 2);
        let dir = func & 1;

        let mut l_raw = [[0u8; 4]; 544];
        for (i, e) in l_raw.iter_mut().enumerate() {
            e.copy_from_slice(&lb[i * 4..i * 4 + 4]);
        }

        let mut mask = vec![0u8; 32 * 32];
        let mut level = vec![[0u8; 4]; 32 * 32];
        let mut mask_hi = vec![0u8; 32 * 32];
        let mut level_hi = vec![[0u8; 4]; 32 * 32];

        let (w4, y0, count, has_second_half) = if dir == 0 {
            assert_eq!(offp, 8 * bpp);
            for gy in 0..n {
                let bit = 1u32 << gy;
                let v = if vmask[2] & bit != 0 {
                    3
                } else if vmask[1] & bit != 0 {
                    2
                } else {
                    u8::from(vmask[0] & bit != 0)
                };
                mask[gy * 32 + 2] = v;
                level[gy * 32 + 2] = l_raw[32 + gy * 32];
                level[gy * 32 + 1] = l_raw[32 + gy * 32 - 1];
            }
            (3usize, 0usize, n, false)
        } else if bd == 8 {
            assert_eq!(offp, 8 * stride_bytes);
            for gx in 0..n {
                let bit = 1u32 << gx;
                let v = if vmask[2] & bit != 0 {
                    3
                } else if vmask[1] & bit != 0 {
                    2
                } else {
                    u8::from(vmask[0] & bit != 0)
                };
                mask[2 * 32 + gx] = v;
                level[2 * 32 + gx] = l_raw[32 + gx];
                level[32 + gx] = l_raw[gx];
            }
            (n, 2usize, 1usize, false)
        } else {
            let gy0 = offp / (4 * stride_bytes);
            assert_eq!(offp, gy0 * 4 * stride_bytes);
            let w_first = n.min(8);
            for gx in 0..w_first {
                let bit = 1u32 << gx;
                let v = if vmask[2] & bit != 0 {
                    3
                } else if vmask[1] & bit != 0 {
                    2
                } else {
                    u8::from(vmask[0] & bit != 0)
                };
                mask[gy0 * 32 + gx] = v;
                level[gy0 * 32 + gx] = l_raw[32 + gx];
                level[(gy0 - 1) * 32 + gx] = l_raw[gx];
            }
            if n > 8 {
                for gx in 8..n {
                    let bit = 1u32 << gx;
                    let v = if vmask[2] & bit != 0 {
                        3
                    } else if vmask[1] & bit != 0 {
                        2
                    } else {
                        u8::from(vmask[0] & bit != 0)
                    };
                    mask_hi[gy0 * 32 + (gx - 8)] = v;
                    level_hi[gy0 * 32 + (gx - 8)] = l_raw[32 + gx];
                    level_hi[(gy0 - 1) * 32 + (gx - 8)] = l_raw[gx];
                }
            }
            (w_first, gy0, 1usize, n > 8)
        };

        let mut sb = SentinelBuffer::with_defaults(plane_len);
        sb.data_mut().copy_from_slice(in_plane);
        dsp.loop_filter_sb[plane_idx][dir](
            sb.data_mut(),
            stride_bytes,
            &mask,
            32,
            &level,
            32,
            0,
            &lim,
            w4,
            y0,
            count,
        );
        if has_second_half {
            let data = sb.data_mut();
            dsp.loop_filter_sb[plane_idx][dir](
                &mut data[stride_bytes..],
                stride_bytes,
                &mask_hi,
                32,
                &level_hi,
                32,
                0,
                &lim,
                n - 8,
                y0,
                count,
            );
        }
        sb.verify().expect("LF SentinelBuffer ihlali");
        assert_eq!(
            sb.data(),
            exp_plane,
            "LF ARM64 NEON ≠ dav1d C oracle: bd={bd} func={func} rec={}",
            verified[bi]
        );
        verified[bi] += 1;
    }

    assert!(
        verified[0] >= min_per_bd && verified[1] >= min_per_bd && verified[2] >= min_per_bd,
        "LF C oracle 8/10/12 bpc kayıtları tam doğrulanmalı (min={min_per_bd}): {verified:?}"
    );
}

#[cfg(target_arch = "aarch64")]
fn verify_lr_oracle_files(
    lr_dsps: &[LrDsp; 3],
    cases_path: &Path,
    expect_path: &Path,
    min_per_bd: usize,
) {
    let cases = std::fs::read(cases_path)
        .unwrap_or_else(|e| panic!("{} okunamadı: {e}", cases_path.display()));
    let expect = std::fs::read(expect_path)
        .unwrap_or_else(|e| panic!("{} okunamadı: {e}", expect_path.display()));
    assert_eq!(&cases[..8], b"AV1OLRC1");
    assert_eq!(&expect[..8], b"AV1OLRE1");
    let n_cases = u32::from_le_bytes([cases[12], cases[13], cases[14], cases[15]]) as usize;
    assert_eq!(cases.len(), 16 + n_cases * LR_REC_BYTES);
    assert_eq!(expect.len(), 12 + n_cases * 32);

    let mut verified = [0usize; 3];
    for ci in 0..n_cases {
        let rec = &cases[16 + ci * LR_REC_BYTES..16 + (ci + 1) * LR_REC_BYTES];
        let c = LrCase::parse(rec);
        let bi = match c.bd {
            8 => 0,
            10 => 1,
            12 => 2,
            _ => continue,
        };
        let dsp = &lr_dsps[bi];
        let kernel: LrFilterFn = match c.func {
            0 => dsp.wiener[0],
            1 => dsp.wiener[1],
            2 => dsp.sgr[0],
            3 => dsp.sgr[1],
            _ => dsp.sgr[2],
        };
        let cb = fill_lr_case(&c);
        let out = run_lr_kernel_on_case(kernel, &c, &cb)
            .unwrap_or_else(|e| panic!("LR ci={ci} ({c:?}): {e}"));
        let got = sha(&out);
        let want = &expect[12 + ci * 32..12 + (ci + 1) * 32];
        assert_eq!(
            &got[..],
            want,
            "LR ARM64 NEON ≠ dav1d C oracle: ci={ci} ({c:?})"
        );
        verified[bi] += 1;
    }

    assert!(
        verified[0] >= min_per_bd && verified[1] >= min_per_bd && verified[2] >= min_per_bd,
        "LR C oracle 8/10/12 bpc kayıtları tam doğrulanmalı (min={min_per_bd}): {verified:?}"
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5d_lf_arm64_small_c_oracle_sentinel_overread_ve_skaler_devir() {
    #[cfg(target_arch = "aarch64")]
    {
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let devir_once = lf_skaler_devir_sayisi();

        let d_neon = DspDispatch::select(CpuFlags::detect());
        let lf_dsps = [
            *d_neon.for_bit_depth(BitDepth::Eight).lf(),
            *d_neon.for_bit_depth(BitDepth::Ten).lf(),
            *d_neon.for_bit_depth(BitDepth::Twelve).lf(),
        ];
        let fdir = lf_fixture_dir();
        verify_lf_oracle_files(
            &lf_dsps,
            &fdir.join("loopfilter-small-input.bin"),
            &fdir.join("loopfilter-small-expect.bin"),
            8,
        );
        assert_eq!(lf_skaler_devir_sayisi(), devir_once);

        // Dar düzlem sınırında (`gx == 1` iken `wd = 4` -> dar 8 baytlık `dst`) OverReadProbe:
        let lim = oracle_lut();
        let probe = OverReadProbe::new(4 * 8, 64, 4 * 8);
        let mut out_a = [0u8; 32];
        let mut out_b = [0u8; 32];
        let mut mask_small = [0u8; 8];
        mask_small[1] = 1; // wd = 4
        let mut lvl_small = [[0u8; 4]; 8];
        lvl_small[1] = [20, 20, 20, 20];
        probe
            .run_pair(&mut out_a, &mut out_b, |inp, out| {
                out.copy_from_slice(&inp[probe.data_offset()..probe.data_offset() + 32]);
                lf_dsps[0].loop_filter_sb[0][0](
                    out,
                    8,
                    &mask_small,
                    2,
                    &lvl_small,
                    2,
                    0,
                    &lim,
                    2,
                    0,
                    1,
                );
            })
            .expect("LF dar düzlem OverReadProbe taşması");

        // Sözleşme dışı kısa `dst` güvenli skaler devir:
        let mut kisa = [0u8; 2];
        lf_dsps[0].loop_filter_sb[0][0](
            &mut kisa,
            8,
            &mask_small,
            2,
            &lvl_small,
            2,
            0,
            &lim,
            2,
            0,
            1,
        );
        assert!(lf_skaler_devir_sayisi() > devir_once);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5d_lf_arm64_full_c_oracle() {
    #[cfg(target_arch = "aarch64")]
    {
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let devir_once = lf_skaler_devir_sayisi();

        let Some(lf_dir) = find_lf_oracle_dir() else {
            on_kosul::eksik(
                "lf_lr_arm64::f5d_lf_arm64_full_c_oracle",
                Neden::OracleVerisiYok,
                "loopfilter-input.bin yok (.cache/loopfilter-oracle)",
            );
            return;
        };

        let d_neon = DspDispatch::select(CpuFlags::detect());
        let lf_dsps = [
            *d_neon.for_bit_depth(BitDepth::Eight).lf(),
            *d_neon.for_bit_depth(BitDepth::Ten).lf(),
            *d_neon.for_bit_depth(BitDepth::Twelve).lf(),
        ];
        verify_lf_oracle_files(
            &lf_dsps,
            &lf_dir.join("loopfilter-input.bin"),
            &lf_dir.join("loopfilter-expect.bin"),
            500,
        );
        assert_eq!(lf_skaler_devir_sayisi(), devir_once);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5d_lr_arm64_small_c_oracle_sentinel_overread_ve_skaler_devir() {
    #[cfg(target_arch = "aarch64")]
    {
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let devir_once = lr_skaler_devir_sayisi();

        let d_neon = DspDispatch::select(CpuFlags::detect());
        let lr_dsps: [LrDsp; 3] = [
            *d_neon.for_bit_depth(BitDepth::Eight).lr(),
            *d_neon.for_bit_depth(BitDepth::Ten).lr(),
            *d_neon.for_bit_depth(BitDepth::Twelve).lr(),
        ];

        let fdir = lr_fixture_dir();
        verify_lr_oracle_files(
            &lr_dsps,
            &fdir.join("lr-small-cases.bin"),
            &fdir.join("lr-small-expect.bin"),
            80,
        );

        // OverReadProbe: Wiener ve SGR dar genişliklerde (`w ∈ {1, 3, 7, 15, 33}`) `lpf` ve `left` sınırı
        let sp = SGR_PARAMS[0];
        let sgr_params = LrParams::Sgr {
            s0: u32::from(sp[0]),
            s1: u32::from(sp[1]),
            w0: 48,
            w1: 32,
        };
        let wiener_params = LrParams::Wiener {
            filter: [
                [3, -7, 15, 106, 15, -7, 3, 0],
                [0, -5, 18, 102, 18, -5, 0, 0],
            ],
        };
        for w in [1usize, 3, 7, 15, 33] {
            let h = 4usize;
            let stride = w + 3;
            let dst_len = (h - 1) * stride + (w + 3);
            let left_len = h * 4;
            let probe = OverReadProbe::new(left_len, 64, dst_len);
            let mut out_a = vec![0u8; dst_len];
            let mut out_b = vec![0u8; dst_len];
            probe
                .run_pair(&mut out_a, &mut out_b, |inp, out| {
                    out.fill(0x80);
                    let left = &inp[probe.data_offset()..probe.data_offset() + left_len];
                    (lr_dsps[0].wiener[0])(
                        out,
                        stride,
                        left,
                        &[],
                        0,
                        0,
                        w,
                        h,
                        &wiener_params,
                        LrEdgeFlags::from_bits(
                            LrEdgeFlags::HAVE_LEFT.bits() | LrEdgeFlags::HAVE_RIGHT.bits(),
                        ),
                        LrBitDepth::B8,
                    );
                    (lr_dsps[0].sgr[2])(
                        out,
                        stride,
                        left,
                        &[],
                        0,
                        0,
                        w,
                        h,
                        &sgr_params,
                        LrEdgeFlags::from_bits(
                            LrEdgeFlags::HAVE_LEFT.bits() | LrEdgeFlags::HAVE_RIGHT.bits(),
                        ),
                        LrBitDepth::B8,
                    );
                })
                .expect("LR OverReadProbe taşması");
        }

        assert_eq!(lr_skaler_devir_sayisi(), devir_once);

        // Sözleşme dışı (`w = 0` ve yanlış `LrBitDepth::B10`) güvenli skaler devir:
        let mut dummy = [0u8; 32];
        (lr_dsps[0].wiener[0])(
            &mut dummy,
            8,
            &[],
            &[],
            0,
            0,
            0,
            4,
            &wiener_params,
            LrEdgeFlags::NONE,
            LrBitDepth::B10,
        );
        assert!(lr_skaler_devir_sayisi() > devir_once);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5d_lr_arm64_full_c_oracle() {
    #[cfg(target_arch = "aarch64")]
    {
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let devir_once = lr_skaler_devir_sayisi();

        let Some(lr_dir) = find_lr_oracle_dir() else {
            on_kosul::eksik(
                "lf_lr_arm64::f5d_lr_arm64_full_c_oracle",
                Neden::OracleVerisiYok,
                "lr-quick-cases.bin yok (.cache/lr-oracle)",
            );
            return;
        };

        let d_neon = DspDispatch::select(CpuFlags::detect());
        let lr_dsps: [LrDsp; 3] = [
            *d_neon.for_bit_depth(BitDepth::Eight).lr(),
            *d_neon.for_bit_depth(BitDepth::Ten).lr(),
            *d_neon.for_bit_depth(BitDepth::Twelve).lr(),
        ];
        verify_lr_oracle_files(
            &lr_dsps,
            &lr_dir.join("lr-quick-cases.bin"),
            &lr_dir.join("lr-quick-expect.bin"),
            2000,
        );
        assert_eq!(lr_skaler_devir_sayisi(), devir_once);
    }
}
