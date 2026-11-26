//! P46 Adım 3 (Paket B) — 16-bpc (`10/12` bit) Loop Restoration (Wiener 7/5 dokunuş ve
//! 10-bpc SGRPROJ 5×5 / 3×3 / karma) SSSE3 (`D02751`, `D02757`, `D02732`, `D02738`, `D02744`),
//! AVX2 (`D02746`, `D02753`, `D02728`, `D02734`, `D02740`) ve AVX512ICL (`D02748`, `D02754`,
//! `D02730`, `D02736`, `D02742`) çekirdek testleri (`15` matris kaydı / `21` aday:
//! `12` Wiener `@10+@12` + `9` SGR `@10`).
//!
//! Doğrulanan sözleşme:
//! 1. [`test_lr_16bpc_yuva_kurulumu`]: `CpuLevel::None`/`Sse2` vs `Ssse3` vs `Avx2` vs `Avx512Icl`
//!    (`SLOW_GATHER` açık ve kapalı):
//!    - `10-bpc` (`BitDepth::Ten`): `5/5` yuva (`wiener[0..2]` + `sgr[0..3]`) SSSE3, AVX2 ve
//!      AVX512ICL düzeylerinde SIMD'dir.
//!    - `12-bpc` (`BitDepth::Twelve`): `2/5` yuva (`wiener[0..2]`) SSSE3, AVX2 ve AVX512ICL
//!      düzeylerinde SIMD'dir; `3/5` yuva (`sgr[0..3]`) dav1d `src/x86/looprestoration.h:62,73,88`
//!      paritesiyle skalerde kalır.
//! 2. [`test_lr_16bpc_skaler_ve_oracle_esitligi`]:
//!    - Varsa `.cache/lr-oracle/lr-quick-cases.bin` ve `lr-quick-expect.bin` içindeki tüm
//!      10-bpc ve 12-bpc C oracle kayıtlarında sözleşme-sıkı zehirli tamponlarla SHA-256
//!      bit-birebir eşitlik ve `0` sınır dışı yazma doğrulanır.
//!    - Ayrıca `lr16-bench` içindeki 38 16-bpc hücresi, 16 `LrEdgeFlags` (`0..16`), tekil
//!      genişlikler (`w = 1, 2, 3, 5, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256, 384`),
//!      küçük yükseklikler (`h = 1..=9, 64`), `SentinelBuffer` (`0xA5` / `0x5A`) ve
//!      `OverReadProbe` ile SSSE3, AVX2 ve AVX512ICL yolları sınanır.
//! 3. [`test_lr_16bpc_sozlesme_disi_skalere_devreder`]: sözleşme dışı çağrılar (`w = 0`,
//!    `w = 385`, `h = 0`, yanlış `LrParams` türü, `LrBitDepth::B8`, 10-bpc SGR SIMD işlevine
//!    doğrudan `LrBitDepth::B12` geçilmesi, kısa `dst`/`left`/`lpf`, `lpf_x < reach`, tek bayt
//!    `stride`/`lpf_stride`) [`lr_skaler_devir_sayisi`]'nı artırır, paniklemez, geçersiz
//!    girdide `dst`'yi değiştirmez ve geçerli `B12` SGR girdisinde tam 12-bpc skaler çıktı üretir.
//! 4. [`test_lr_16bpc_isa_sinir_denetimi`]: test ikilisini `objdump -d` ile sökerek
//!    `lr16_ssse3` sembollerinde `0` SSE4.1+/VEX/EVEX, `lr16::` (AVX2) sembollerinde `0` EVEX
//!    ve `lr16_avx512` sembollerinde `0` `vpgather*`/`vpscatter*` yönergesi bulunduğunu denetler.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::BitDepth;
use av1oxide::cpu::CpuFlags;
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::lr::{
    LrBitDepth, LrDsp, LrEdgeFlags, LrFilterFn, LrParams, lr_skaler_devir_sayisi,
};
use av1oxide::dsp::lr_tables::SGR_PARAMS;
#[cfg(target_arch = "x86_64")]
use av1oxide::dsp::x86::{lr16, lr16_avx512, lr16_ssse3};
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

const LR_FIELDS: [&str; 5] = ["wiener[0]", "wiener[1]", "sgr[0]", "sgr[1]", "sgr[2]"];

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct IsaKernels {
    ad: &'static str,
    wiener: [LrFilterFn; 2],
    sgr10: [LrFilterFn; 3],
}

#[cfg(target_arch = "x86_64")]
fn isa_kernels(test: &str) -> Option<[IsaKernels; 3]> {
    let d = DspDispatch::select(CpuFlags::detect());
    if d.avx512icl_token().is_none() {
        on_kosul::eksik(
            test,
            Neden::isa(),
            "[P46] 16-bpc looprestoration SSSE3/AVX2/AVX512ICL testleri AVX512ICL ister",
        );
        return None;
    }
    Some([
        IsaKernels {
            ad: "ssse3",
            wiener: [
                lr16_ssse3::wiener_filter7_16bpc_ssse3,
                lr16_ssse3::wiener_filter5_16bpc_ssse3,
            ],
            sgr10: [
                lr16_ssse3::sgr_filter_5x5_16bpc_ssse3,
                lr16_ssse3::sgr_filter_3x3_16bpc_ssse3,
                lr16_ssse3::sgr_filter_mix_16bpc_ssse3,
            ],
        },
        IsaKernels {
            ad: "avx2",
            wiener: [
                lr16::wiener_filter7_16bpc_avx2,
                lr16::wiener_filter5_16bpc_avx2,
            ],
            sgr10: [
                lr16::sgr_filter_5x5_16bpc_avx2,
                lr16::sgr_filter_3x3_16bpc_avx2,
                lr16::sgr_filter_mix_16bpc_avx2,
            ],
        },
        IsaKernels {
            ad: "avx512icl",
            wiener: [
                lr16_avx512::wiener_filter7_16bpc_avx512icl,
                lr16_avx512::wiener_filter5_16bpc_avx512icl,
            ],
            sgr10: [
                lr16_avx512::sgr_filter_5x5_16bpc_avx512icl,
                lr16_avx512::sgr_filter_3x3_16bpc_avx512icl,
                lr16_avx512::sgr_filter_mix_16bpc_avx512icl,
            ],
        },
    ])
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4) — C oracle karşılaştırması için
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

    fn reach(&self) -> usize {
        if self.func == 3 { 2 } else { 3 }
    }
}

fn fill_case(c: &Case) -> CBufs {
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

struct Poison(u64);

impl Poison {
    fn byte(&mut self) -> u8 {
        (sm64(&mut self.0) >> 56) as u8
    }
}

fn run_kernel_on_case(kernel: LrFilterFn, c: &Case, cb: &CBufs) -> Result<Vec<u8>, String> {
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

    let dpad = (var & 3) as usize;
    let ds = (w + rreach + dpad) * bpp;
    let dst_len = (h - 1) * ds + (w + rreach) * bpp;
    let mut dst = vec![0u8; h * ds];
    for b in &mut dst {
        *b = poison.byte();
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
                    left[(y * 4 + k) * bpp + b] = poison.byte();
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

fn find_lr_oracle_dir() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = root.join("../..").join(".cache/lr-oracle");
    if local.join("lr-quick-cases.bin").exists() && local.join("lr-quick-expect.bin").exists() {
        return Some(local);
    }
    if let Ok(out) = std::process::Command::new("git")
        .args(["rev-parse", "--git-common-dir"])
        .current_dir(root)
        .output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout);
        let common = Path::new(s.trim());
        if let Some(repo_root) = common.parent() {
            let shared = repo_root.join(".cache/lr-oracle");
            if shared.join("lr-quick-cases.bin").exists()
                && shared.join("lr-quick-expect.bin").exists()
            {
                return Some(shared);
            }
        }
    }
    None
}

#[test]
fn test_lr_16bpc_yuva_kurulumu() {
    #[cfg(target_arch = "x86_64")]
    {
        let Some(_isas) = isa_kernels("lr_16bpc_x86::test_lr_16bpc_yuva_kurulumu") else {
            return;
        };

        for (bits, beklenen_w10, beklenen_s10, beklenen_w12, beklenen_s12) in [
            (0x00u32, "skaler", "skaler", "skaler", "skaler"),
            (0x01u32, "skaler", "skaler", "skaler", "skaler"),
            (0x03u32, "ssse3", "ssse3", "ssse3", "skaler"),
            (0x07u32, "ssse3", "ssse3", "ssse3", "skaler"),
            (0x0fu32, "avx2", "avx2", "avx2", "avx2"),
            (0x2fu32, "avx2", "avx2", "avx2", "avx2"),
            (0x1fu32, "avx512icl", "avx512icl", "avx512icl", "avx2"),
            (0x3fu32, "avx512icl", "avx512icl", "avx512icl", "avx2"),
        ] {
            let flags = CpuFlags::from_dav1d_x86_bits(bits);
            let planned = DspDispatch::planned_slot_states(flags);
            let d = DspDispatch::select(flags);

            for (idx, field) in LR_FIELDS.iter().enumerate() {
                let b10 = if idx < 2 { beklenen_w10 } else { beklenen_s10 };
                let b12 = if idx < 2 { beklenen_w12 } else { beklenen_s12 };

                let st10 = planned
                    .iter()
                    .find(|s| {
                        s.key.bit_depth == Some(10)
                            && s.key.family == "looprestoration"
                            && s.key.field == *field
                    })
                    .expect("10-bpc looprestoration yuvası");
                assert_eq!(
                    st10.implementation, b10,
                    "planned @10 bits={bits:#x} {field}"
                );
                assert_eq!(
                    d.slot_implementation(BitDepth::Ten, "looprestoration", field),
                    b10,
                    "live @10 bits={bits:#x} {field}"
                );

                let st12 = planned
                    .iter()
                    .find(|s| {
                        s.key.bit_depth == Some(12)
                            && s.key.family == "looprestoration"
                            && s.key.field == *field
                    })
                    .expect("12-bpc looprestoration yuvası");
                assert_eq!(
                    st12.implementation, b12,
                    "planned @12 bits={bits:#x} {field}"
                );
                assert_eq!(
                    d.slot_implementation(BitDepth::Twelve, "looprestoration", field),
                    b12,
                    "live @12 bits={bits:#x} {field}"
                );
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        on_kosul::eksik(
            "lr_16bpc_x86::test_lr_16bpc_yuva_kurulumu",
            Neden::isa(),
            "[P46] x86_64 gerektirir",
        );
    }
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

#[test]
fn test_lr_16bpc_skaler_ve_oracle_esitligi() {
    #[cfg(target_arch = "x86_64")]
    {
        let Some(isas) = isa_kernels("lr_16bpc_x86::test_lr_16bpc_skaler_ve_oracle_esitligi")
        else {
            return;
        };
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let devir_once = lr_skaler_devir_sayisi();

        // 1) `.cache/lr-oracle/lr-quick-cases.bin` + `lr-quick-expect.bin` C oracle doğrulaması
        if let Some(dir) = find_lr_oracle_dir() {
            let cases = std::fs::read(dir.join("lr-quick-cases.bin")).expect("lr-quick-cases.bin");
            let expect =
                std::fs::read(dir.join("lr-quick-expect.bin")).expect("lr-quick-expect.bin");
            assert_eq!(&cases[..8], b"AV1OLRC1");
            assert_eq!(&expect[..8], b"AV1OLRE1");
            let n = u32::from_le_bytes([cases[12], cases[13], cases[14], cases[15]]) as usize;
            let mut dogrulanan_16bpc = 0usize;

            for i in 0..n {
                let c = Case::parse(&cases[16 + i * REC_BYTES..16 + (i + 1) * REC_BYTES]);
                if c.bd == 8 {
                    continue;
                }
                let cb = fill_case(&c);
                let exp = &expect[12 + i * 32..12 + (i + 1) * 32];
                let mut ref_out: Option<Vec<u8>> = None;

                for isa in &isas {
                    if c.bd == 12 && c.func >= 2 && isa.ad != "avx2" {
                        continue;
                    }
                    let kernel = if c.func < 2 {
                        isa.wiener[c.func]
                    } else {
                        isa.sgr10[c.func - 2]
                    };
                    let out = run_kernel_on_case(kernel, &c, &cb).unwrap_or_else(|e| {
                        panic!("oracle #{i} ({}) sözleşme hatası: {e}", isa.ad)
                    });
                    if let Some(first) = &ref_out {
                        assert_eq!(
                            &out, first,
                            "C oracle çıktı farkı: #{i} ({}) func={} bd={} edges={:04b} {}x{}",
                            isa.ad, c.func, c.bd, c.edges, c.w, c.h
                        );
                    } else {
                        let digest = sha(&out);
                        assert_eq!(
                            &digest[..],
                            exp,
                            "C oracle SHA-256 farkı: #{i} ({}) func={} bd={} edges={:04b} {}x{} filter={:?} s=({},{}) w=({},{})",
                            isa.ad,
                            c.func,
                            c.bd,
                            c.edges,
                            c.w,
                            c.h,
                            c.filter,
                            c.s0,
                            c.s1,
                            c.w0,
                            c.w1
                        );
                        ref_out = Some(out);
                    }
                }
                dogrulanan_16bpc += 1;
            }
            assert!(
                dogrulanan_16bpc >= 4000,
                "en az 4000 adet 16-bpc C oracle kaydı doğrulanmalı: {dogrulanan_16bpc}"
            );
        }

        // 2) 38 adet `lr16-bench` hücresi (10-bpc 30 hücre Wiener+SGR + 12-bpc 12 hücre Wiener = 42/38 hücre)
        let scalar10 = LrDsp::scalar::<10>();
        let scalar12 = LrDsp::scalar::<12>();
        const BOY_ALL: [(usize, usize); 4] = [(64, 64), (128, 64), (256, 64), (384, 64)];
        const BOY_NONE: [(usize, usize); 2] = [(64, 32), (256, 64)];
        const SGR_5X5_IDX: [usize; 4] = [10, 11, 12, 13];
        const SGR_3X3_IDX: [usize; 4] = [14, 15, 14, 15];
        const SGR_MIX_IDX: [usize; 4] = [0, 3, 6, 9];

        let mut hucre_idx = 0usize;
        for bpc in [10u8, 12] {
            let bd = if bpc == 10 {
                LrBitDepth::B10
            } else {
                LrBitDepth::B12
            };
            let bd_max = (1i32 << bpc) - 1;
            for func in 0..5usize {
                for (edges, boylar) in [
                    (LrEdgeFlags::ALL, &BOY_ALL[..]),
                    (LrEdgeFlags::NONE, &BOY_NONE[..]),
                ] {
                    for &(w, h) in boylar {
                        let mut rng =
                            Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(hucre_idx as u64 + 1));
                        hucre_idx += 1;
                        let stride_bytes = 448 * 2;
                        let lpf_x = 32usize;
                        let mut orig = vec![0u8; stride_bytes * 64];
                        let mut left = vec![0u8; 64 * 4 * 2];
                        let mut lpf = vec![0u8; 8 * stride_bytes];
                        for px in orig.as_chunks_mut::<2>().0 {
                            *px = (rng.aralik(0, bd_max) as u16).to_le_bytes();
                        }
                        for px in left.as_chunks_mut::<2>().0 {
                            *px = (rng.aralik(0, bd_max) as u16).to_le_bytes();
                        }
                        for px in lpf.as_chunks_mut::<2>().0 {
                            *px = (rng.aralik(0, bd_max) as u16).to_le_bytes();
                        }
                        let params = if func < 2 {
                            let five = func == 1;
                            let fh0 = if five { 0 } else { rng.aralik(-5, 10) as i16 };
                            let fh1 = rng.aralik(-23, 8) as i16;
                            let fh2 = rng.aralik(-17, 46) as i16;
                            let fv0 = if five { 0 } else { rng.aralik(-5, 10) as i16 };
                            let fv1 = rng.aralik(-23, 8) as i16;
                            let fv2 = rng.aralik(-17, 46) as i16;
                            let mut filter = [[0i16; 8]; 2];
                            filter[0] =
                                [fh0, fh1, fh2, 128 - (fh0 + fh1 + fh2) * 2, fh2, fh1, fh0, 0];
                            filter[1] =
                                [fv0, fv1, fv2, 128 - (fv0 + fv1 + fv2) * 2, fv2, fv1, fv0, 0];
                            LrParams::Wiener { filter }
                        } else {
                            let idx = match func {
                                2 => SGR_5X5_IDX[0],
                                3 => SGR_3X3_IDX[0],
                                _ => SGR_MIX_IDX[0],
                            };
                            let s0 = u32::from(SGR_PARAMS[idx][0]);
                            let s1 = u32::from(SGR_PARAMS[idx][1]);
                            let w0 = if s0 != 0 { rng.aralik(-96, 31) } else { 0 };
                            let w1r = if s1 != 0 { rng.aralik(-32, 95) } else { 95 };
                            let w1 = 128 - (w0 + w1r);
                            LrParams::Sgr { s0, s1, w0, w1 }
                        };

                        let skaler_fn = match (bpc, func) {
                            (10, 0) => scalar10.wiener[0],
                            (10, 1) => scalar10.wiener[1],
                            (10, 2) => scalar10.sgr[0],
                            (10, 3) => scalar10.sgr[1],
                            (10, _) => scalar10.sgr[2],
                            (12, 0) => scalar12.wiener[0],
                            (12, 1) => scalar12.wiener[1],
                            (12, 2) => scalar12.sgr[0],
                            (12, 3) => scalar12.sgr[1],
                            (_, _) => scalar12.sgr[2],
                        };
                        let mut ref_dst = orig.clone();
                        skaler_fn(
                            &mut ref_dst,
                            stride_bytes,
                            &left,
                            &lpf,
                            stride_bytes,
                            lpf_x,
                            w,
                            h,
                            &params,
                            edges,
                            bd,
                        );

                        for isa in &isas {
                            if bpc == 12 && func >= 2 && isa.ad != "avx2" {
                                continue;
                            }
                            let simd_fn = if func < 2 {
                                isa.wiener[func]
                            } else {
                                isa.sgr10[func - 2]
                            };
                            let mut sb = SentinelBuffer::new(orig.len(), 64, 0x5A);
                            sb.data_mut().copy_from_slice(&orig);
                            simd_fn(
                                sb.data_mut(),
                                stride_bytes,
                                &left,
                                &lpf,
                                stride_bytes,
                                lpf_x,
                                w,
                                h,
                                &params,
                                edges,
                                bd,
                            );
                            sb.verify().expect("SentinelBuffer 0x5A koruyucu bozuldu");
                            assert_eq!(
                                sb.data(),
                                ref_dst.as_slice(),
                                "lr16-bench farkı: ({}) bpc={bpc} func={func} {w}x{h}",
                                isa.ad
                            );
                        }
                    }
                }
            }
        }

        // 3) Tekil genişlikler, küçük yükseklikler, 16 LrEdgeFlags, SentinelBuffer ve OverReadProbe
        let widths = [
            1usize, 2, 3, 5, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256, 384,
        ];
        let heights = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12];
        let mut rng = Rng(0x2026_1119_0046_1699);

        for (step_idx, &w) in widths.iter().enumerate() {
            let h = heights[step_idx % heights.len()];
            for e_off in 0..4u8 {
                let e_bits = ((step_idx as u8).wrapping_mul(4).wrapping_add(e_off)) & 15;
                for (func, bpc) in [
                    (0usize, 10u8),
                    (1, 10),
                    (2, 10),
                    (3, 10),
                    (4, 10),
                    (0, 12),
                    (1, 12),
                    (2, 12),
                    (3, 12),
                    (4, 12),
                ] {
                    let c = Case {
                        func,
                        bd: u32::from(bpc),
                        edges: e_bits,
                        pattern: (step_idx as u8) & 7,
                        w,
                        h,
                        stride_px: w + 16,
                        lpf_x: 4,
                        seed: rng.u64() as u32,
                        filter: if func == 0 {
                            [
                                [3, -7, 15, 106 + 128, 15, -7, 3, 0],
                                [4, -9, 18, 102, 18, -9, 4, 0],
                            ]
                        } else {
                            [
                                [0, -7, 15, 112 + 128, 15, -7, 0, 0],
                                [0, -9, 18, 110, 18, -9, 0, 0],
                            ]
                        },
                        s0: 140,
                        s1: 3236,
                        w0: -32,
                        w1: 129,
                    };
                    let cb = fill_case(&c);
                    let skaler_dsp = LrDsp::scalar_for(c.bit_depth());
                    let skaler_fn = if func < 2 {
                        skaler_dsp.wiener[func]
                    } else {
                        skaler_dsp.sgr[func - 2]
                    };
                    let expected =
                        run_kernel_on_case(skaler_fn, &c, &cb).expect("skaler sözleşme içi");
                    for isa in &isas {
                        if bpc == 12 && func >= 2 && isa.ad != "avx2" {
                            continue;
                        }
                        let simd_fn = if func < 2 {
                            isa.wiener[func]
                        } else {
                            isa.sgr10[func - 2]
                        };
                        let got = run_kernel_on_case(simd_fn, &c, &cb).unwrap_or_else(|err| {
                            panic!(
                                "sözleşme hatası ({}) bpc={bpc} func={func} edges={e_bits:04b} {w}x{h}: {err}",
                                isa.ad
                            )
                        });
                        assert_eq!(
                            got, expected,
                            "fark ({}) bpc={bpc} func={func} edges={e_bits:04b} {w}x{h}",
                            isa.ad
                        );
                    }
                }
            }
        }

        // 4) OverReadProbe ile doğrudan padding okuma denetimi
        for isa in &isas {
            for (func, reach) in [(0usize, 3usize), (1, 3), (2, 3), (3, 2), (4, 3)] {
                let simd_fn = if func < 2 {
                    isa.wiener[func]
                } else {
                    isa.sgr10[func - 2]
                };
                let (w, h) = (17usize, 5usize);
                let rreach = reach;
                let stride = (w + rreach) * 2;
                let dst_len = (h - 1) * stride + (w + rreach) * 2;
                let params = if func < 2 {
                    LrParams::Wiener {
                        filter: [
                            [0, -7, 15, 112 + 128, 15, -7, 0, 0],
                            [0, -7, 15, 112, 15, -7, 0, 0],
                        ],
                    }
                } else {
                    LrParams::Sgr {
                        s0: 140,
                        s1: 3236,
                        w0: -32,
                        w1: 129,
                    }
                };
                let probe = OverReadProbe::new(dst_len, 64, dst_len);
                probe
                    .run(|padded_in, out_buf| {
                        let off = probe.data_offset();
                        let mut raw_dst = Vec::with_capacity(padded_in.len());
                        for px in padded_in.as_chunks::<2>().0 {
                            let v = u16::from_le_bytes(*px) & 1023;
                            raw_dst.extend_from_slice(&v.to_le_bytes());
                        }
                        simd_fn(
                            &mut raw_dst[off..off + dst_len],
                            stride,
                            &[],
                            &[],
                            0,
                            0,
                            w,
                            h,
                            &params,
                            LrEdgeFlags::HAVE_RIGHT,
                            LrBitDepth::B10,
                        );
                        out_buf.copy_from_slice(&raw_dst[off..off + dst_len]);
                    })
                    .unwrap_or_else(|e| {
                        panic!("OverReadProbe ihlali ({} func={func}): {e:?}", isa.ad)
                    });
            }
        }

        assert_eq!(
            lr_skaler_devir_sayisi(),
            devir_once,
            "sözleşme içi çağrılarda skaler devir olmamalı"
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        on_kosul::eksik(
            "lr_16bpc_x86::test_lr_16bpc_skaler_ve_oracle_esitligi",
            Neden::isa(),
            "[P46] x86_64 gerektirir",
        );
    }
}

#[test]
fn test_lr_16bpc_sozlesme_disi_skalere_devreder() {
    #[cfg(target_arch = "x86_64")]
    {
        let Some(isas) = isa_kernels("lr_16bpc_x86::test_lr_16bpc_sozlesme_disi_skalere_devreder")
        else {
            return;
        };
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let wp = LrParams::Wiener {
            filter: [
                [0, -7, 15, 112 + 128, 15, -7, 0, 0],
                [0, -7, 15, 112, 15, -7, 0, 0],
            ],
        };
        let sp = LrParams::Sgr {
            s0: 140,
            s1: 3236,
            w0: -32,
            w1: 129,
        };
        let mut dst = vec![0x33u8; 64 * 800];
        let before = dst.clone();
        let left = vec![0x11u8; 64 * 8];
        let lpf = vec![0x22u8; 8 * 800];

        for isa in &isas {
            for (func, f) in [
                (0usize, isa.wiener[0]),
                (1, isa.wiener[1]),
                (2, isa.sgr10[0]),
                (3, isa.sgr10[1]),
                (4, isa.sgr10[2]),
            ] {
                let good_params = if func < 2 { &wp } else { &sp };
                let bad_params = if func < 2 { &sp } else { &wp };

                // 1) w = 0
                let d0 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    800,
                    &left,
                    &lpf,
                    800,
                    4,
                    0,
                    4,
                    good_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d0 + 1);
                assert_eq!(dst, before);

                // 2) w = 385 (> 384)
                let d1 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    800,
                    &left,
                    &lpf,
                    800,
                    4,
                    385,
                    4,
                    good_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d1 + 1);
                assert_eq!(dst, before);

                // 3) h = 0
                let d2 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    800,
                    &left,
                    &lpf,
                    800,
                    4,
                    16,
                    0,
                    good_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d2 + 1);
                assert_eq!(dst, before);

                // 4) Yanlış LrParams varyantı
                let d3 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    800,
                    &left,
                    &lpf,
                    800,
                    4,
                    16,
                    4,
                    bad_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d3 + 1);
                assert_eq!(dst, before);

                // 5) LrBitDepth::B8 16-bpc işlevine verildiğinde
                let d4 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    800,
                    &left,
                    &lpf,
                    800,
                    4,
                    16,
                    4,
                    good_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B8,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d4 + 1);
                assert_eq!(dst, before);

                // 6) Kısa dst / left / lpf ve lpf_x < reach
                let d5 = lr_skaler_devir_sayisi();
                f(
                    &mut dst[..32],
                    800,
                    &left,
                    &lpf,
                    800,
                    4,
                    16,
                    4,
                    good_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d5 + 1);
                assert_eq!(dst, before);

                let d6 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    800,
                    &left[..8],
                    &lpf,
                    800,
                    4,
                    16,
                    4,
                    good_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d6 + 1);
                assert_eq!(dst, before);

                let d7 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    800,
                    &left,
                    &lpf[..64],
                    800,
                    4,
                    16,
                    4,
                    good_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d7 + 1);
                assert_eq!(dst, before);

                let d8 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    800,
                    &left,
                    &lpf,
                    800,
                    1,
                    16,
                    4,
                    good_params,
                    LrEdgeFlags::ALL,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d8 + 1);
                assert_eq!(dst, before);

                // 7) Tek bayt stride (stride % 2 != 0)
                let d9 = lr_skaler_devir_sayisi();
                f(
                    &mut dst,
                    801,
                    &left,
                    &lpf,
                    800,
                    4,
                    16,
                    4,
                    good_params,
                    LrEdgeFlags::NONE,
                    LrBitDepth::B10,
                );
                assert_eq!(lr_skaler_devir_sayisi(), d9 + 1);
                assert_eq!(dst, before);
            }

            // 8) SGR SIMD işlevine geçerli LrBitDepth::B12 verildiğinde:
            // AVX2 doğrudan 12-bpc SIMD yolunu çalıştırır (skaler devir 0),
            // SSSE3 ve AVX512ICL 10-bpc çekirdekleri ise 12-bpc skaler devre düşer;
            // tümü 12-bpc skaler SGR ile birebir aynı çıktıyı üretir.
            let scalar12 = LrDsp::scalar::<12>();
            for sgr_idx in 0..3usize {
                let (w, h) = (16usize, 4usize);
                let stride = w * 2;
                let mut simd_dst = vec![0u8; stride * h];
                for (i, px) in simd_dst.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let v = ((i as u32).wrapping_mul(2_654_435_761) >> 20) as u16 & 4095;
                    *px = v.to_le_bytes();
                }
                let mut ref_dst = simd_dst.clone();
                scalar12.sgr[sgr_idx](
                    &mut ref_dst,
                    stride,
                    &[],
                    &[],
                    0,
                    0,
                    w,
                    h,
                    &sp,
                    LrEdgeFlags::NONE,
                    LrBitDepth::B12,
                );
                let d_before = lr_skaler_devir_sayisi();
                isa.sgr10[sgr_idx](
                    &mut simd_dst,
                    stride,
                    &[],
                    &[],
                    0,
                    0,
                    w,
                    h,
                    &sp,
                    LrEdgeFlags::NONE,
                    LrBitDepth::B12,
                );
                let beklenen_devir = if isa.ad == "avx2" {
                    d_before
                } else {
                    d_before + 1
                };
                assert_eq!(lr_skaler_devir_sayisi(), beklenen_devir);
                assert_eq!(
                    simd_dst, ref_dst,
                    "SGR ({}) B12 çağrısında 12-bpc skaler çıktıyla eşleşmeli",
                    isa.ad
                );
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        on_kosul::eksik(
            "lr_16bpc_x86::test_lr_16bpc_sozlesme_disi_skalere_devreder",
            Neden::isa(),
            "[P46] x86_64 gerektirir",
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_lr_16bpc_isa_sinir_denetimi() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "-C", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let yasak_ssse3 = [
        "pblendvb",
        "pblendw",
        "pinsrb",
        "pinsrd",
        "pinsrq",
        "pextrb",
        "pextrd",
        "pextrq",
        "pmovzxbw",
        "pmovzxbd",
        "pmovzxbq",
        "pmovzxwd",
        "pmovzxwq",
        "pmovzxdq",
        "pmovsxbw",
        "pmovsxbd",
        "pmovsxbq",
        "pmovsxwd",
        "pmovsxwq",
        "pmovsxdq",
        "pmuldq",
        "pmulld",
        "pminsb",
        "pmaxsb",
        "pminuw",
        "pmaxuw",
        "pminud",
        "pmaxud",
        "pminsd",
        "pmaxsd",
        "packusdw",
        "ptest",
        "pcmpeqq",
        "pcmpgtq",
        "mpsadbw",
        "phminposuw",
    ];

    let mut ssse3_bulundu = 0usize;
    let mut avx2_bulundu = 0usize;
    let mut avx512_bulundu = 0usize;
    let mut aktif_kip: Option<&str> = None;
    let mut aktif_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") {
            aktif_sembol = satir.to_string();
            if satir.contains("dsp::x86::lr16_ssse3::") {
                aktif_kip = Some("ssse3");
                ssse3_bulundu += 1;
            } else if satir.contains("dsp::x86::lr16::") {
                aktif_kip = Some("avx2");
                avx2_bulundu += 1;
            } else if satir.contains("dsp::x86::lr16_avx512::") {
                aktif_kip = Some("avx512");
                avx512_bulundu += 1;
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
            "ssse3" => {
                assert!(
                    !yasak_ssse3.contains(&op),
                    "SSSE3 lr16_ssse3 sembolünde SSE4.1 yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSSE3 lr16_ssse3 sembolünde VEX/EVEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm") && !sag.contains("{%k"),
                    "SSSE3 lr16_ssse3 sembolünde YMM/ZMM/k yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "avx2" => {
                assert!(
                    !sag.contains("%zmm") && !sag.contains("{%k"),
                    "AVX2 lr16 sembolünde EVEX yönergesi: {aktif_sembol} -> {satir}"
                );
            }
            "avx512" => {
                assert!(
                    !op.starts_with("vpgather") && !op.starts_with("vpscatter"),
                    "AVX512ICL lr16_avx512 sembolünde gather/scatter yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
            }
            _ => unreachable!(),
        }
    }

    assert!(
        ssse3_bulundu >= 5 && avx2_bulundu >= 5 && avx512_bulundu >= 5,
        "semboller bulunamadı: ssse3={ssse3_bulundu}, avx2={avx2_bulundu}, avx512={avx512_bulundu}"
    );
}
