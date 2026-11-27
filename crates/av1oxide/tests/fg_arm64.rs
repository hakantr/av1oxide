//! P54 (`F5d` — `F5-FG`) `aarch64` NEON film grain (`16 / 16` matris kaydı = `24` yuva @
//! `8 / 10 / 12` bpc: `generate_grain_y`, `generate_grain_uv[0..2]`, `fgy_32x32xn`,
//! `fguv_32x32xn[0..2]`) C oracle SHA-256 eşitliği, `SentinelBuffer`, `OverReadProbe` ve
//! sözleşme dışı güvenli skaler devir testleri.

#![cfg_attr(
    not(target_arch = "aarch64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::bitstream::frame_header::FilmGrainData;
use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::fg::{
    EMPTY_LUT, FgBitDepth, FgDsp, GRAIN_HEIGHT, GRAIN_WIDTH, GrainLut, SUB_GRAIN_HEIGHT,
    SUB_GRAIN_WIDTH, fg_skaler_devir_sayisi,
};
use av1oxide::dsp::{ARM_ISA_CANDIDATES, DspDispatch};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

const FG_FIELDS: [&str; 8] = [
    "generate_grain_y",
    "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420-1]",
    "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422-1]",
    "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444-1]",
    "fgy_32x32xn",
    "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
    "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
    "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
];

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

fn sm64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

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

fn fg_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("fg")
}

fn find_fg_oracle_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("FG_ORACLE_OUT") {
        let p = PathBuf::from(d);
        if p.join("fg-quick-cases.bin").is_file() && p.join("fg-quick-expect.bin").is_file() {
            return Some(p);
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let local = root.join(".cache/fg-oracle");
    if local.join("fg-quick-cases.bin").is_file() && local.join("fg-quick-expect.bin").is_file() {
        return Some(local);
    }
    None
}

#[cfg(target_arch = "aarch64")]
fn verify_fg_oracle_files(
    neon_dsps: &[FgDsp; 3],
    cases_path: &Path,
    expect_path: &Path,
    min_per_bd: usize,
) {
    let cases = std::fs::read(cases_path)
        .unwrap_or_else(|e| panic!("{} okunamadı: {e}", cases_path.display()));
    let expect = std::fs::read(expect_path)
        .unwrap_or_else(|e| panic!("{} okunamadı: {e}", expect_path.display()));
    assert_eq!(cases.len() % REC_SIZE, 0);
    assert_eq!(expect.len(), (cases.len() / REC_SIZE) * 32);

    let mut lut: Box<GrainLut> = Box::new(EMPTY_LUT);
    let mut lut2: Box<GrainLut> = Box::new(EMPTY_LUT);
    let mut verified = [0usize; 3];

    for (ci, rec) in cases.chunks(REC_SIZE).enumerate() {
        let e = parse_case(rec);
        if e.kind == 4 {
            continue;
        }
        let (bd, bi) = match e.bd {
            8 => (FgBitDepth::B8, 0),
            10 => (FgBitDepth::B10, 1),
            _ => (FgBitDepth::B12, 2),
        };
        let bpp = bd.bytes();
        let max = bd.bitdepth_max();
        let gctr = 128i32 << (e.bd - 8);
        let dsp = &neon_dsps[bi];
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
                let src_len = (e.bh - 1) * stride + e.pw * bpp;
                let dst_len = stride * e.bh;
                let luma_len = (luma_rows - 1) * lstride + luma_w * bpp;

                let mut src = vec![0x3Cu8; src_len];
                let mut dst_sb = SentinelBuffer::with_defaults(dst_len);
                dst_sb.data_mut().fill(0x5A);
                let mut luma = vec![0x3Cu8; luma_len];
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
                        dst_sb.data_mut(),
                        &src,
                        stride,
                        d,
                        e.pw,
                        &scaling,
                        &lut,
                        e.bh,
                        e.row_num,
                        bd,
                    );
                } else {
                    (dsp.fguv_32x32xn[e.layout])(
                        dst_sb.data_mut(),
                        &src,
                        stride,
                        d,
                        e.pw,
                        &scaling,
                        &lut,
                        e.bh,
                        e.row_num,
                        &luma,
                        lstride,
                        e.uv,
                        e.is_id,
                        bd,
                    );
                }
                dst_sb
                    .verify()
                    .unwrap_or_else(|err| panic!("FG SentinelBuffer ihlali ci={ci}: {err:?}"));
                let dst = dst_sb.data();
                for (i, &b) in dst.iter().enumerate() {
                    let inside = i / stride < e.bh && i % stride < e.pw * bpp;
                    assert!(
                        inside || b == 0x5A,
                        "FG bölge dışı yazdı ci={ci} kind={} bd={}",
                        e.kind,
                        e.bd
                    );
                }
                for y in 0..e.bh {
                    h.update(&dst[y * stride..y * stride + e.pw * bpp]);
                }
            }
        }

        let got = h.finish();
        let want = &expect[ci * 32..(ci + 1) * 32];
        assert_eq!(
            &got[..],
            want,
            "FG ARM64 NEON ≠ dav1d C oracle: ci={ci} kind={} bd={} layout={} pw={} bh={}",
            e.kind,
            e.bd,
            e.layout,
            e.pw,
            e.bh
        );
        verified[bi] += 1;
    }

    assert!(
        verified[0] >= min_per_bd && verified[1] >= min_per_bd && verified[2] >= min_per_bd,
        "tüm bpc'lerde yeterli C oracle çağrısı doğrulanmalı (min={min_per_bd}): {verified:?}"
    );
}

#[test]
fn f5d_fg_arm64_aday_ve_kurulum() {
    let fg_adaylar: Vec<_> = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "filmgrain")
        .collect();
    assert_eq!(fg_adaylar.len(), 24);
    for bd in [8u8, 10, 12] {
        for field in FG_FIELDS {
            assert!(
                fg_adaylar.iter().any(|c| c.bit_depth == bd
                    && c.field == field
                    && c.implementation == "neon"
                    && c.requires == CpuFlags::NEON
                    && c.default),
                "eksik ARM64 FG adayı: filmgrain/{field}@{bd}"
            );
        }
    }

    let d_none = DspDispatch::select(CpuLevel::None.allowed());
    for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
        for field in FG_FIELDS {
            assert_eq!(d_none.slot_implementation(bd, "filmgrain", field), "skaler");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let d_neon = DspDispatch::select(CpuFlags::detect());
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            for field in FG_FIELDS {
                assert_eq!(d_neon.slot_implementation(bd, "filmgrain", field), "neon");
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5d_fg_arm64_small_c_oracle_sentinel_overread_ve_skaler_devir() {
    #[cfg(target_arch = "aarch64")]
    {
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let devir_once = fg_skaler_devir_sayisi();

        let d_neon = DspDispatch::select(CpuFlags::detect());
        let neon_dsps: [FgDsp; 3] = [
            *d_neon.for_bit_depth(BitDepth::Eight).fg(),
            *d_neon.for_bit_depth(BitDepth::Ten).fg(),
            *d_neon.for_bit_depth(BitDepth::Twelve).fg(),
        ];

        // 1) Depo içi küçük C Oracle (`tests/fixtures/fg/fg-small-{cases,expect}.bin`, 24/24 hücre)
        let fdir = fg_fixture_dir();
        verify_fg_oracle_files(
            &neon_dsps,
            &fdir.join("fg-small-cases.bin"),
            &fdir.join("fg-small-expect.bin"),
            96,
        );

        // 2) OverReadProbe (fgy ve fguv, 8/10/12 bpc, tekil genişlikler pw ∈ {1, 3, 7, 15, 33})
        let mut lut: Box<GrainLut> = Box::new(EMPTY_LUT);
        let mut fg_data = FilmGrainData {
            seed: 0xBEEF,
            scaling_shift: 8,
            ar_coeff_lag: 2,
            ar_coeff_shift: 8,
            grain_scale_shift: 0,
            overlap_flag: true,
            clip_to_restricted_range: false,
            chroma_scaling_from_luma: false,
            uv_mult: [32, -24],
            uv_luma_mult: [16, 20],
            uv_offset: [8, -8],
            ..FilmGrainData::default()
        };
        fg_data.num_y_points = 2;
        fg_data.num_uv_points = [2, 2];
        for (bi, bd) in [FgBitDepth::B8, FgBitDepth::B10, FgBitDepth::B12]
            .into_iter()
            .enumerate()
        {
            let dsp = &neon_dsps[bi];
            let bpp = bd.bytes();
            let max_u = bd.bitdepth_max() as usize;
            let scaling = vec![128u8; max_u + 1];
            (dsp.generate_grain_y)(&mut lut, &fg_data, bd);
            for pw in [1usize, 3, 7, 15, 33] {
                let bh = 5usize;
                let stride = (pw + 3) * bpp;
                let src_len = (bh - 1) * stride + pw * bpp;
                let dst_len = (bh - 1) * stride + pw * bpp;
                let probe = OverReadProbe::new(src_len, 64, dst_len);
                let mut out_a = vec![0u8; dst_len];
                let mut out_b = vec![0u8; dst_len];
                probe
                    .run_pair(&mut out_a, &mut out_b, |inp, out| {
                        let mut valid_src =
                            inp[probe.data_offset()..probe.data_offset() + src_len].to_vec();
                        if bpp == 2 {
                            for ch in valid_src.as_chunks_mut::<2>().0 {
                                let v = u16::from_le_bytes(*ch) & (max_u as u16);
                                *ch = v.to_le_bytes();
                            }
                        }
                        let s_slice = if bpp == 1 {
                            &inp[probe.data_offset()..probe.data_offset() + src_len]
                        } else {
                            &valid_src[..]
                        };
                        (dsp.fgy_32x32xn)(
                            out, s_slice, stride, &fg_data, pw, &scaling, &lut, bh, 1, bd,
                        );
                    })
                    .expect("fgy OverReadProbe taşması");
            }
        }

        // Normal çalışmada skaler devir 0 olmalı:
        assert_eq!(fg_skaler_devir_sayisi(), devir_once);

        // 3) Sözleşme dışı girdilerde paniklemeden güvenli skaler devir:
        let dsp8 = &neon_dsps[0];
        let mut kisa_dst = [0u8; 4];
        let kisa_src = [0u8; 4];
        let kisa_scal = [0u8; 16];
        (dsp8.fgy_32x32xn)(
            &mut kisa_dst,
            &kisa_src,
            8,
            &fg_data,
            8,
            &kisa_scal,
            &lut,
            4,
            0,
            FgBitDepth::B10, // yanlış bit derinliği -> skaler devir
        );
        assert!(fg_skaler_devir_sayisi() > devir_once);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5d_fg_arm64_full_c_oracle() {
    #[cfg(target_arch = "aarch64")]
    {
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let devir_once = fg_skaler_devir_sayisi();

        let Some(dir) = find_fg_oracle_dir() else {
            on_kosul::eksik(
                "fg_arm64::f5d_fg_arm64_full_c_oracle",
                Neden::OracleVerisiYok,
                "fg-quick-cases.bin yok (.cache/fg-oracle)",
            );
            return;
        };

        let d_neon = DspDispatch::select(CpuFlags::detect());
        let neon_dsps: [FgDsp; 3] = [
            *d_neon.for_bit_depth(BitDepth::Eight).fg(),
            *d_neon.for_bit_depth(BitDepth::Ten).fg(),
            *d_neon.for_bit_depth(BitDepth::Twelve).fg(),
        ];
        verify_fg_oracle_files(
            &neon_dsps,
            &dir.join("fg-quick-cases.bin"),
            &dir.join("fg-quick-expect.bin"),
            1000,
        );
        assert_eq!(fg_skaler_devir_sayisi(), devir_once);
    }
}
