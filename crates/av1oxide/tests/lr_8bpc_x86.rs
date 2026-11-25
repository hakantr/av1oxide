//! P46 (F4-LR, Adım 3 / Paket A) — 8 bpc loop restoration (`wiener[0..2]`,
//! `sgr[0..3]`; 16 kayıt = 2 SSE2 + 5 SSSE3 + 5 AVX2 + 4 AVX512ICL + `wiener[1]@8`
//! AVX512ICL takma adı = 17 aday: `D02749`, `D02755`, `D02750`, `D02756`,
//! `D02731`, `D02737`, `D02743`, `D02745`, `D02752`, `D02727`, `D02733`,
//! `D02739`, `D02747`, `D02729`, `D02735`, `D02741`) çekirdek testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **`test_lr_8bpc_yuva_kurulumu`:** `CpuLevel::None` (`5/5` skaler),
//!    `CpuLevel::Sse2` (`wiener[0..2]` `"sse2"`, `sgr[0..3]` `"skaler"`),
//!    `CpuLevel::Ssse3` / `CpuLevel::Sse41` (`5/5` `"ssse3"`),
//!    `CpuLevel::Avx2` (`5/5` `"avx2"`), `CpuLevel::Avx512Icl` (`5/5` `"avx512icl"`,
//!    `SLOW_GATHER` açık ve kapalıyken aynı — `SLOW_GATHER` hiçbir LR AVX-512
//!    yuvasını engellemez).
//! 2. **`test_lr_8bpc_skaler_ve_oracle_esitligi`:**
//!    - `.cache/lr-oracle/lr-quick-cases.bin` varsa tüm 8-bpc kayıtlarında
//!      `sse2`, `ssse3`, `avx2`, `avx512icl` gövdeleri sözleşme-sıkı zehirli
//!      dilimlerle dav1d C oracle SHA-256 özetlerine (`lr-quick-expect.bin`)
//!      %100 bit-birebir uyar ve `[0, w)` dışına tek bayt yazmaz.
//!    - Bütün `lr-bench` hücreleri, 16 `LrEdgeFlags` (`0..16`), tekil/küçük
//!      genişlikler (`w ∈ {1, 2, 3, 5, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256, 384}`),
//!      küçük yükseklikler (`h ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 64}`),
//!      `SentinelBuffer` (`0xA5` / `0x5A`) ve `OverReadProbe` denetimleri.
//! 3. **`test_lr_8bpc_sozlesme_disi_skalere_devreder`:** Sözleşme dışı çağrılar
//!    (`w = 0`, `w = 385`, `h = 0`, yanlış `LrParams`, yanlış `LrBitDepth::B10`,
//!    kısa `dst`/`left`/`lpf`, `lpf_x < reach`) paniklemez, [`lr_skaler_devir_sayisi`]
//!    sayacını artırır ve `dst`'yi değiştirmeden bırakır.
//! 4. **`test_lr_8bpc_isa_sinir_denetimi`:** `objdump -d` ile:
//!    - `lr_sse2` sembollerinde `0` SSE3+/SSSE3/SSE4.1/VEX/EVEX;
//!    - `lr_ssse3` sembollerinde `0` SSE4.1+/VEX/EVEX;
//!    - `x86::lr` (AVX2) sembollerinde `0` EVEX (`%zmm` / `{%k`);
//!    - `lr_avx512` sembollerinde `0` `vpgather*` / `vpscatter*`.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::lr::{
    LrBitDepth, LrDsp, LrEdgeFlags, LrFilterFn, LrParams, lr_skaler_devir_sayisi,
};
use av1oxide::dsp::lr_tables::SGR_PARAMS;
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

const LR_FIELDS: [&str; 5] = ["wiener[0]", "wiener[1]", "sgr[0]", "sgr[1]", "sgr[2]"];

struct Lr8Tablolar {
    skaler: LrDsp,
    sse2: LrDsp,
    ssse3: LrDsp,
    avx2: LrDsp,
    avx512: LrDsp,
}

fn lr8_tablolari(test: &str) -> Option<Lr8Tablolar> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect().difference(CpuFlags::SLOW_GATHER);
        let d_sse2 = DspDispatch::select(detected.intersection(CpuLevel::Sse2.allowed()));
        let d_ssse3 = DspDispatch::select(detected.intersection(CpuLevel::Ssse3.allowed()));
        let d_avx2 = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()));
        let d_avx512 = DspDispatch::select(detected);
        if d_sse2.sse2_token().is_some()
            && d_ssse3.ssse3_token().is_some()
            && d_avx2.avx2_token().is_some()
            && d_avx512.avx512icl_token().is_some()
        {
            return Some(Lr8Tablolar {
                skaler: LrDsp::scalar::<8>(),
                sse2: *d_sse2.for_bit_depth(BitDepth::Eight).lr(),
                ssse3: *d_ssse3.for_bit_depth(BitDepth::Eight).lr(),
                avx2: *d_avx2.for_bit_depth(BitDepth::Eight).lr(),
                avx512: *d_avx512.for_bit_depth(BitDepth::Eight).lr(),
            });
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P46] 8 bpc LR SSE2/SSSE3/AVX2/AVX512ICL gövdeleri x86_64 AVX512ICL ister",
    );
    None
}

fn find_oracle_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("LR_ORACLE_OUT") {
        let p = PathBuf::from(d);
        if p.join("lr-quick-cases.bin").is_file() && p.join("lr-quick-expect.bin").is_file() {
            return Some(p);
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let local = root.join(".cache/lr-oracle");
    if local.join("lr-quick-cases.bin").is_file() && local.join("lr-quick-expect.bin").is_file() {
        return Some(local);
    }
    let shared = PathBuf::from("/home/hakanbiris/github/av1oxide/.cache/lr-oracle");
    if shared.join("lr-quick-cases.bin").is_file() && shared.join("lr-quick-expect.bin").is_file() {
        return Some(shared);
    }
    None
}

// ---------------------------------------------------------------------------
// SHA-256 & Oracle yardımcı yapıları (`tests/lr_scalar_oracle.rs` ile birebir)
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

struct CBufs {
    dst: Vec<u8>,
    left: Vec<u8>,
    lpf: Vec<u8>,
}

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

fn fill_8bpc(c: &Case) -> CBufs {
    let max = 255i32;
    let (h, sp, lx) = (c.h, c.stride_px, c.lpf_x);
    let mut st = u64::from(c.seed);
    let mut dst = vec![0u8; h * sp];
    let mut left = vec![0u8; h * 4];
    let mut lpf = vec![0u8; 8 * sp];
    for y in 0..h {
        for x in 0..sp {
            let r = sm64(&mut st);
            dst[y * sp + x] = pix(c.pattern, x as i32, y as i32, 0, r, max) as u8;
        }
    }
    for y in 0..h {
        for k in 0..4 {
            let r = sm64(&mut st);
            left[y * 4 + k] = pix(c.pattern, k as i32 - 4, y as i32, 1, r, max) as u8;
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
            lpf[row * sp + col] = pix(c.pattern, col as i32 - lx as i32, ly, 2, r, max) as u8;
        }
    }
    CBufs { dst, left, lpf }
}

fn run_tight_8bpc(c: &Case, cb: &CBufs, kernel: LrFilterFn) -> Result<Vec<u8>, String> {
    let edges = LrEdgeFlags::from_bits(c.edges);
    let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
    let have_right = edges.contains(LrEdgeFlags::HAVE_RIGHT);
    let have_top = edges.contains(LrEdgeFlags::HAVE_TOP);
    let have_bottom = edges.contains(LrEdgeFlags::HAVE_BOTTOM);
    let reach = c.reach();
    let lreach = if have_left { reach } else { 0 };
    let rreach = if have_right { reach } else { 0 };
    let (w, h, csp) = (c.w, c.h, c.stride_px);
    let mut poison_st = u64::from(c.seed) ^ 0xa5a5_5a5a_dead_beef;
    let mut next_poison = || (sm64(&mut poison_st) >> 56) as u8;
    let mut vs = u64::from(c.seed).wrapping_mul(31);
    let var = sm64(&mut vs);

    let dpad = (var & 3) as usize;
    let ds = w + rreach + dpad;
    let dst_len = (h - 1) * ds + (w + rreach);
    let mut dst = vec![0u8; h * ds];
    for b in dst.iter_mut() {
        *b = next_poison();
    }
    for y in 0..h {
        let n = w + rreach;
        dst[y * ds..y * ds + n].copy_from_slice(&cb.dst[y * csp..y * csp + n]);
    }
    dst.truncate(dst_len);
    let dst_before = dst.clone();

    let mut left = Vec::new();
    if have_left {
        left = cb.left.clone();
        for y in 0..h {
            for k in 0..4 - reach {
                left[y * 4 + k] = next_poison();
            }
        }
    }

    let lx = lreach + ((var >> 2) & 3) as usize;
    let lpad = ((var >> 4) & 7) as usize;
    let ls = lx + w + rreach + lpad;
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
            *b = next_poison();
        }
        let rows: &[usize] = match (have_top, have_bottom) {
            (true, true) => &[0, 1, 6, 7],
            (true, false) => &[0, 1],
            (false, true) => &[6, 7],
            (false, false) => &[],
        };
        for &r in rows {
            let n = lreach + w + rreach;
            let src = r * csp + (c.lpf_x - lreach);
            let dsto = r * ls + (lx - lreach);
            lpf[dsto..dsto + n].copy_from_slice(&cb.lpf[src..src + n]);
        }
        lpf.truncate(last * ls + (lx + w + rreach));
    }

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
        LrBitDepth::B8,
    );

    for y in 0..h {
        let row_end = if y + 1 == h { dst_len } else { (y + 1) * ds };
        for o in y * ds + w..row_end {
            if dst[o] != dst_before[o] {
                return Err(format!(
                    "sözleşme dışı yazma: satır {y}, bayt {}",
                    o - y * ds
                ));
            }
        }
    }

    let mut out = cb.dst.clone();
    for y in 0..h {
        out[y * csp..y * csp + w].copy_from_slice(&dst[y * ds..y * ds + w]);
    }
    Ok(out)
}

struct Rng(u64);

impl Rng {
    fn u64(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn aralik(&mut self, lo: i32, hi: i32) -> i32 {
        lo + ((self.u64() >> 33) % (hi - lo + 1) as u64) as i32
    }
}

fn ornek_params(rng: &mut Rng, func: usize) -> LrParams {
    if func < 2 {
        let five = func == 1;
        let fh0 = if five { 0 } else { rng.aralik(-5, 10) as i16 };
        let fh1 = rng.aralik(-23, 8) as i16;
        let fh2 = rng.aralik(-17, 46) as i16;
        let fv0 = if five { 0 } else { rng.aralik(-5, 10) as i16 };
        let fv1 = rng.aralik(-23, 8) as i16;
        let fv2 = rng.aralik(-17, 46) as i16;
        let mut filter = [[0i16; 8]; 2];
        filter[0][0] = fh0;
        filter[0][6] = fh0;
        filter[0][1] = fh1;
        filter[0][5] = fh1;
        filter[0][2] = fh2;
        filter[0][4] = fh2;
        filter[0][3] = -(fh0 + fh1 + fh2) * 2;
        filter[1][0] = fv0;
        filter[1][6] = fv0;
        filter[1][1] = fv1;
        filter[1][5] = fv1;
        filter[1][2] = fv2;
        filter[1][4] = fv2;
        filter[1][3] = 128 - (fv0 + fv1 + fv2) * 2;
        LrParams::Wiener { filter }
    } else {
        let idx = match func {
            2 => [14usize, 15][(rng.u64() & 1) as usize],
            3 => [10usize, 11, 12, 13][(rng.u64() & 3) as usize],
            _ => (rng.u64() as usize) % 10,
        };
        let s0 = u32::from(SGR_PARAMS[idx][0]);
        let s1 = u32::from(SGR_PARAMS[idx][1]);
        let w0 = if s0 != 0 { rng.aralik(-96, 31) } else { 0 };
        let w1r = if s1 != 0 { rng.aralik(-32, 95) } else { 95 };
        let w1 = 128 - (w0 + w1r);
        LrParams::Sgr { s0, s1, w0, w1 }
    }
}

fn kernel_of(dsp: &LrDsp, func: usize) -> LrFilterFn {
    match func {
        0 => dsp.wiener[0],
        1 => dsp.wiener[1],
        2 => dsp.sgr[0],
        3 => dsp.sgr[1],
        _ => dsp.sgr[2],
    }
}

#[test]
fn test_lr_8bpc_yuva_kurulumu() {
    let Some(_t) = lr8_tablolari("lr_8bpc_x86::test_lr_8bpc_yuva_kurulumu") else {
        return;
    };
    for (bits, beklenen) in [
        (0x00u32, ["skaler", "skaler", "skaler", "skaler", "skaler"]),
        (0x01u32, ["sse2", "sse2", "skaler", "skaler", "skaler"]),
        (0x03u32, ["ssse3", "ssse3", "ssse3", "ssse3", "ssse3"]),
        (0x07u32, ["ssse3", "ssse3", "ssse3", "ssse3", "ssse3"]),
        (0x0fu32, ["avx2", "avx2", "avx2", "avx2", "avx2"]),
        (0x2fu32, ["avx2", "avx2", "avx2", "avx2", "avx2"]),
        (
            0x3fu32,
            [
                "avx512icl",
                "avx512icl",
                "avx512icl",
                "avx512icl",
                "avx512icl",
            ],
        ),
        (
            0x1fu32,
            [
                "avx512icl",
                "avx512icl",
                "avx512icl",
                "avx512icl",
                "avx512icl",
            ],
        ),
    ] {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        let d = DspDispatch::select(flags);
        for (field, exp) in LR_FIELDS.iter().zip(beklenen.iter()) {
            let st = planned
                .iter()
                .find(|s| {
                    s.key.family == "looprestoration"
                        && s.key.field == *field
                        && s.key.bit_depth == Some(8)
                })
                .expect("8 bpc looprestoration yuvası");
            assert_eq!(
                st.implementation, *exp,
                "planlanan bits={bits:#x} looprestoration/{field}@8"
            );
            assert_eq!(
                d.slot_implementation(BitDepth::Eight, "looprestoration", field),
                *exp,
                "canlı bits={bits:#x} looprestoration/{field}@8"
            );
        }
    }
}

#[test]
fn test_lr_8bpc_skaler_ve_oracle_esitligi() {
    let Some(t) = lr8_tablolari("lr_8bpc_x86::test_lr_8bpc_skaler_ve_oracle_esitligi") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir_once = lr_skaler_devir_sayisi();

    // 1) `.cache/lr-oracle/lr-quick-cases.bin` varsa tüm 8-bpc kayıtlarında C oracle SHA-256 eşitliği.
    if let Some(dir) = find_oracle_dir() {
        let cases = std::fs::read(dir.join("lr-quick-cases.bin")).expect("lr-quick-cases.bin");
        let expect = std::fs::read(dir.join("lr-quick-expect.bin")).expect("lr-quick-expect.bin");
        assert_eq!(&cases[..8], b"AV1OLRC1");
        assert_eq!(&expect[..8], b"AV1OLRE1");
        let n = u32::from_le_bytes([cases[12], cases[13], cases[14], cases[15]]) as usize;
        let mut tested_8bpc = 0usize;
        for i in 0..n {
            let c = Case::parse(&cases[16 + i * REC_BYTES..16 + (i + 1) * REC_BYTES]);
            if c.bd != 8 {
                continue;
            }
            let cb = fill_8bpc(&c);
            let exp = &expect[12 + i * 32..12 + (i + 1) * 32];
            let mut ref_out: Option<Vec<u8>> = None;
            for (isa_ad, dsp) in [
                ("sse2", &t.sse2),
                ("ssse3", &t.ssse3),
                ("avx2", &t.avx2),
                ("avx512icl", &t.avx512),
            ] {
                if isa_ad == "sse2" && c.func >= 2 {
                    continue;
                }
                let f = kernel_of(dsp, c.func);
                let out = run_tight_8bpc(&c, &cb, f).unwrap_or_else(|e| {
                    panic!("lr-quick-cases #{i} ({isa_ad} func={}): {e}", c.func)
                });
                if let Some(first) = &ref_out {
                    assert_eq!(
                        &out, first,
                        "lr-quick-cases #{i} çıktı farkı ({isa_ad} func={} w={} h={} edges={:04b})",
                        c.func, c.w, c.h, c.edges
                    );
                } else {
                    let digest = sha(&out);
                    assert_eq!(
                        &digest[..],
                        exp,
                        "lr-quick-cases #{i} SHA-256 farkı ({isa_ad} func={} w={} h={} edges={:04b})",
                        c.func,
                        c.w,
                        c.h,
                        c.edges
                    );
                    ref_out = Some(out);
                }
            }
            tested_8bpc += 1;
        }
        assert!(
            tested_8bpc >= 1000,
            "lr-quick-cases 8bpc kayıt sayısı beklenenden az: {tested_8bpc}"
        );
    }

    // 2) `lr-bench` 19/30 hücre geometrileri (`w ∈ {64, 128, 256, 384}`, `h ∈ {32, 64}`, `ALL`/`NONE`)
    //    ve tüm 16 `LrEdgeFlags`, tekil genişlikler ve küçük yükseklikler.
    let mut rng = Rng(0x2026_1119_0046_0803);
    let widths = [
        1usize, 2, 3, 5, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256, 384,
    ];
    let heights = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 32, 64];

    for func in 0..5 {
        let reach = if func == 3 { 2 } else { 3 };
        let f_ref = kernel_of(&t.skaler, func);

        // 2a) Tüm 16 `LrEdgeFlags` × seçili tekil/büyük boyutlar + `SentinelBuffer`
        for e_bits in 0..16u8 {
            let edges = LrEdgeFlags::from_bits(e_bits);
            let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
            let have_right = edges.contains(LrEdgeFlags::HAVE_RIGHT);
            let have_top = edges.contains(LrEdgeFlags::HAVE_TOP);
            let have_bottom = edges.contains(LrEdgeFlags::HAVE_BOTTOM);
            let rreach = if have_right { reach } else { 0 };
            let lreach = if have_left { reach } else { 0 };

            for &(w, h) in &[
                (1usize, 1usize),
                (2, 2),
                (3, 3),
                (5, 4),
                (7, 5),
                (15, 6),
                (16, 7),
                (17, 8),
                (31, 9),
                (33, 4),
                (63, 5),
                (65, 3),
                (127, 2),
                (384, 4),
            ] {
                let stride = w + rreach + 3;
                let dst_len = (h - 1) * stride + w + rreach;
                let mut orig = vec![0u8; dst_len];
                for px in &mut orig {
                    *px = rng.aralik(0, 255) as u8;
                }
                let left_len = if have_left { h * 4 } else { 0 };
                let mut left = vec![0u8; left_len];
                for px in &mut left {
                    *px = rng.aralik(0, 255) as u8;
                }
                let lpf_x = lreach;
                let lpf_stride = lpf_x + w + rreach + 2;
                let lpf_len = if have_bottom {
                    7 * lpf_stride + lpf_x + w + rreach
                } else if have_top {
                    lpf_stride + lpf_x + w + rreach
                } else {
                    0
                };
                let mut lpf = vec![0u8; lpf_len];
                for px in &mut lpf {
                    *px = rng.aralik(0, 255) as u8;
                }
                let params = ornek_params(&mut rng, func);

                let mut ref_dst = orig.clone();
                f_ref(
                    &mut ref_dst,
                    stride,
                    &left,
                    &lpf,
                    lpf_stride,
                    lpf_x,
                    w,
                    h,
                    &params,
                    edges,
                    LrBitDepth::B8,
                );

                for (isa_ad, dsp) in [
                    ("sse2", &t.sse2),
                    ("ssse3", &t.ssse3),
                    ("avx2", &t.avx2),
                    ("avx512icl", &t.avx512),
                ] {
                    if isa_ad == "sse2" && func >= 2 {
                        continue;
                    }
                    let f = kernel_of(dsp, func);
                    let mut sb = SentinelBuffer::with_defaults(dst_len);
                    assert!(sb.write_data(&orig));
                    f(
                        sb.data_mut(),
                        stride,
                        &left,
                        &lpf,
                        lpf_stride,
                        lpf_x,
                        w,
                        h,
                        &params,
                        edges,
                        LrBitDepth::B8,
                    );
                    assert_eq!(
                        sb.verify(),
                        Ok(()),
                        "SentinelBuffer taşması ({isa_ad} func={func} edges={e_bits:04b} {w}x{h})"
                    );
                    assert_eq!(
                        sb.data(),
                        ref_dst.as_slice(),
                        "Skaler farkı ({isa_ad} func={func} edges={e_bits:04b} {w}x{h})"
                    );
                }
            }
        }

        // 2b) Genişlik ve yükseklik taraması + `OverReadProbe`
        for &w in &widths {
            let h = heights[w % heights.len()].min(if w >= 128 { 16 } else { 64 });
            let edges = LrEdgeFlags::ALL;
            let stride = w + reach;
            let dst_len = (h - 1) * stride + w + reach;
            let left_len = h * 4;
            let lpf_x = reach;
            let lpf_stride = lpf_x + w + reach;
            let lpf_len = 7 * lpf_stride + lpf_x + w + reach;

            let mut left = vec![0u8; left_len];
            for px in &mut left {
                *px = rng.aralik(0, 255) as u8;
            }
            let mut lpf = vec![0u8; lpf_len];
            for px in &mut lpf {
                *px = rng.aralik(0, 255) as u8;
            }
            let params = ornek_params(&mut rng, func);

            let probe = OverReadProbe::new(dst_len, 64, dst_len);
            let mut ref_out = probe.payload();
            f_ref(
                &mut ref_out,
                stride,
                &left,
                &lpf,
                lpf_stride,
                lpf_x,
                w,
                h,
                &params,
                edges,
                LrBitDepth::B8,
            );

            for (isa_ad, dsp) in [
                ("sse2", &t.sse2),
                ("ssse3", &t.ssse3),
                ("avx2", &t.avx2),
                ("avx512icl", &t.avx512),
            ] {
                if isa_ad == "sse2" && func >= 2 {
                    continue;
                }
                let f = kernel_of(dsp, func);
                let out = probe
                    .run(|input, out| {
                        let start = probe.data_offset();
                        out.copy_from_slice(&input[start..start + dst_len]);
                        let mut sb = SentinelBuffer::with_defaults(dst_len);
                        assert!(sb.write_data(out));
                        f(
                            sb.data_mut(),
                            stride,
                            &left,
                            &lpf,
                            lpf_stride,
                            lpf_x,
                            w,
                            h,
                            &params,
                            edges,
                            LrBitDepth::B8,
                        );
                        assert_eq!(sb.verify(), Ok(()));
                        out.copy_from_slice(sb.data());
                    })
                    .unwrap_or_else(|e| {
                        panic!("OverReadProbe ihlali ({isa_ad} func={func} w={w} h={h}): {e:?}")
                    });
                assert_eq!(
                    out, ref_out,
                    "OverReadProbe çıktı farkı ({isa_ad} func={func} w={w} h={h})"
                );
            }
        }
    }

    assert_eq!(
        lr_skaler_devir_sayisi(),
        devir_once,
        "sözleşme içi çağrılarda skaler devir olmamalı"
    );
}

#[test]
fn test_lr_8bpc_sozlesme_disi_skalere_devreder() {
    let Some(t) = lr8_tablolari("lr_8bpc_x86::test_lr_8bpc_sozlesme_disi_skalere_devreder") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let wp = LrParams::Wiener {
        filter: [
            [3, -7, 15, -22, 15, -7, 3, 0],
            [0, -7, 15, 112, 15, -7, 0, 0],
        ],
    };
    let sp = LrParams::Sgr {
        s0: 140,
        s1: 3236,
        w0: -32,
        w1: 129,
    };
    let all = LrEdgeFlags::ALL;
    let mut dst = vec![0x42u8; 64 * 400];
    let before = dst.clone();
    let left = vec![0u8; 64 * 4];
    let lpf = vec![0u8; 8 * 400];

    for (isa_ad, dsp) in [
        ("sse2", &t.sse2),
        ("ssse3", &t.ssse3),
        ("avx2", &t.avx2),
        ("avx512icl", &t.avx512),
    ] {
        for func in 0..5 {
            if isa_ad == "sse2" && func >= 2 {
                continue;
            }
            let f = kernel_of(dsp, func);
            let good_p = if func < 2 { &wp } else { &sp };
            let bad_p = if func < 2 { &sp } else { &wp };
            let reach = if func == 3 { 2 } else { 3 };
            let d0 = lr_skaler_devir_sayisi();

            // 1) w = 0
            f(
                &mut dst,
                400,
                &left,
                &lpf,
                400,
                3,
                0,
                4,
                good_p,
                all,
                LrBitDepth::B8,
            );
            // 2) w = 385 (> LR_MAX_W)
            f(
                &mut dst,
                400,
                &left,
                &lpf,
                400,
                3,
                385,
                4,
                good_p,
                all,
                LrBitDepth::B8,
            );
            // 3) h = 0
            f(
                &mut dst,
                400,
                &left,
                &lpf,
                400,
                3,
                16,
                0,
                good_p,
                all,
                LrBitDepth::B8,
            );
            // 4) Yanlış LrParams varyantı
            f(
                &mut dst,
                400,
                &left,
                &lpf,
                400,
                3,
                16,
                4,
                bad_p,
                all,
                LrBitDepth::B8,
            );
            // 5) Yanlış LrBitDepth::B10
            f(
                &mut dst,
                400,
                &left,
                &lpf,
                400,
                3,
                16,
                4,
                good_p,
                all,
                LrBitDepth::B10,
            );
            // 6) Kısa dst
            f(
                &mut dst[..100],
                400,
                &left,
                &lpf,
                400,
                3,
                16,
                4,
                good_p,
                all,
                LrBitDepth::B8,
            );
            // 7) Kısa left
            f(
                &mut dst,
                400,
                &left[..8],
                &lpf,
                400,
                3,
                16,
                4,
                good_p,
                all,
                LrBitDepth::B8,
            );
            // 8) Kısa lpf
            f(
                &mut dst,
                400,
                &left,
                &lpf[..100],
                400,
                3,
                16,
                4,
                good_p,
                all,
                LrBitDepth::B8,
            );
            // 9) lpf_x < reach
            f(
                &mut dst,
                400,
                &left,
                &lpf,
                400,
                reach - 1,
                16,
                4,
                good_p,
                all,
                LrBitDepth::B8,
            );

            assert_eq!(
                lr_skaler_devir_sayisi(),
                d0 + 9,
                "sözleşme dışı çağrılarda skaler devir sayacı 9 artmalı ({isa_ad} func={func})"
            );
            assert_eq!(
                dst, before,
                "sözleşme dışı çağrılarda dst değişmemeli ({isa_ad} func={func})"
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_lr_8bpc_isa_sinir_denetimi() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let yasak_sse2 = [
        "pshufb",
        "phaddw",
        "phaddd",
        "phaddsw",
        "pmaddubsw",
        "phsubw",
        "phsubd",
        "phsubsw",
        "psignb",
        "psignw",
        "psignd",
        "pmulhrsw",
        "pabsb",
        "pabsw",
        "pabsd",
        "palignr",
        "lddqu",
        "haddps",
        "hsubps",
        "addsubps",
        "movshdup",
        "movsldup",
        "movddup",
    ];
    let yasak_ssse3 = [
        "pblendvb", "pblendw", "pinsrb", "pinsrd", "pinsrq", "pextrb", "pextrd", "pextrq",
        "pmovzxbw", "pmovzxbd", "pmovzxbq", "pmovzxwd", "pmovzxwq", "pmovzxdq", "pmovsxbw",
        "pmovsxbd", "pmovsxbq", "pmovsxwd", "pmovsxwq", "pmovsxdq", "pmuldq", "pmulld", "pminsb",
        "pmaxsb", "pminuw", "pmaxuw", "pminud", "pmaxud", "pminsd", "pmaxsd", "packusdw",
    ];

    let mut sse2_bulundu = 0usize;
    let mut ssse3_bulundu = 0usize;
    let mut avx2_bulundu = 0usize;
    let mut avx512_bulundu = 0usize;
    let mut aktif_kip: Option<&str> = None;
    let mut aktif_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") {
            aktif_sembol = satir.to_string();
            if satir.contains("lr_sse2") {
                aktif_kip = Some("sse2");
                sse2_bulundu += 1;
            } else if satir.contains("lr_ssse3") {
                aktif_kip = Some("ssse3");
                ssse3_bulundu += 1;
            } else if satir.contains("lr_avx512") {
                aktif_kip = Some("avx512");
                avx512_bulundu += 1;
            } else if satir.contains("dsp3x862lr")
                || satir.contains("wiener_filter7_8bpc_avx2")
                || satir.contains("wiener_filter5_8bpc_avx2")
                || satir.contains("sgr_filter_5x5_8bpc_avx2")
                || satir.contains("sgr_filter_3x3_8bpc_avx2")
                || satir.contains("sgr_filter_mix_8bpc_avx2")
                || satir.contains("wiener_avx2_giris")
                || satir.contains("sgr_5x5_avx2_giris")
                || satir.contains("sgr_3x3_avx2_giris")
                || satir.contains("sgr_mix_avx2_giris")
            {
                aktif_kip = Some("avx2");
                avx2_bulundu += 1;
            } else {
                aktif_kip = None;
            }
            continue;
        }
        if satir.is_empty() {
            aktif_kip = None;
            continue;
        }
        let Some(kip) = aktif_kip else {
            continue;
        };
        let Some((_, sag)) = satir.split_once('\t') else {
            continue;
        };
        let parcalar: Vec<&str> = sag.split_whitespace().collect();
        let Some(&op) = parcalar.first() else {
            continue;
        };
        match kip {
            "sse2" => {
                assert!(
                    !yasak_sse2.contains(&op) && !yasak_ssse3.contains(&op),
                    "SSE2 LR sembolünde SSE3+ yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSE2 LR sembolünde VEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm"),
                    "SSE2 LR sembolünde YMM/ZMM yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "ssse3" => {
                assert!(
                    !yasak_ssse3.contains(&op),
                    "SSSE3 LR sembolünde SSE4.1 yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSSE3 LR sembolünde VEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm"),
                    "SSSE3 LR sembolünde YMM/ZMM yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "avx2" => {
                assert!(
                    !sag.contains("%zmm") && !sag.contains("{%k"),
                    "AVX2 LR sembolünde EVEX yönergesi: {aktif_sembol} -> {satir}"
                );
            }
            "avx512" => {
                assert!(
                    !op.contains("gather") && !op.contains("scatter"),
                    "AVX512ICL LR sembolünde gather/scatter yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
            }
            _ => {}
        }
    }

    assert!(sse2_bulundu >= 1, "lr_sse2 sembolü bulunmalı");
    assert!(ssse3_bulundu >= 1, "lr_ssse3 sembolü bulunmalı");
    assert!(avx2_bulundu >= 1, "x86::lr (AVX2) sembolü bulunmalı");
    assert!(avx512_bulundu >= 1, "lr_avx512 sembolü bulunmalı");
}
