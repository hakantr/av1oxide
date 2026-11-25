//! S2b.4 — MC aile differential'ı: skaler Rust ↔ dav1d C (bit-exact).
//!
//! Oracle, dav1d `src/mc_tmpl.c`'yi (BITDEPTH 8/16, **asm yok**) ayrı bir
//! süreçte derler (`scripts/oracle/mc_family_oracle.c`) ve beklenen çıktıyı
//! üretir. Bu test üretim crate'inin skaler çekirdeklerini aynı girdiyle
//! koşturup karşılaştırır; üretim crate'i oracle'a **bağlanmaz** (SRC-11).
//!
//! | Küme | Konum | Karşılaştırma | Git |
//! |---|---|---|---|
//! | Küçük sınır kümesi | `tests/fixtures/mc-family/` | **tam bayt** (kayıt kayıt) | ✅ |
//! | Büyük küme (hücre başına 1 000 çağrı) | `.cache/mc-family/` | **hücre SHA-256** | ⬜ |
//!
//! Büyük kümede tam bayt dökümü ~5 GB tuttuğu için hücre başına SHA-256
//! özeti kullanılır (bkz. `scripts/oracle/mc_family_oracle.h`); iki taraf da
//! **aynı özeti** üretir. Küçük kümede karşılaştırma bayt düzeyindedir.
//!
//! Yeniden üretim (tek komut):
//! `bash scripts/oracle/run_mc_oracle.sh`

#![forbid(unsafe_code)]
#![allow(clippy::chunks_exact_to_as_chunks, clippy::while_let_loop)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::{Filter2d, MC_KUYRUK, McDsp, McPrepFn, McPutFn, N_W_MASK};
use av1oxide::dsp::scalar::mc as scalar;
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Aile kimlikleri — `scripts/oracle/mc_family_oracle.h` ile birebir.
// ---------------------------------------------------------------------------

const FAM_PUT: u32 = 0;
const FAM_PUT_SCALED: u32 = 1;
const FAM_PREP: u32 = 2;
const FAM_PREP_SCALED: u32 = 3;
const FAM_AVG: u32 = 4;
const FAM_W_AVG: u32 = 5;
const FAM_MASK: u32 = 6;
const FAM_W_MASK: u32 = 7;
const FAM_BLEND: u32 = 8;
const FAM_BLEND_V: u32 = 9;
const FAM_BLEND_H: u32 = 10;
const FAM_WARP: u32 = 11;
const FAM_WARP_T: u32 = 12;
const FAM_EMU_EDGE: u32 = 13;
const FAM_COUNT: u32 = 14;

const FAM_NAMES: [&str; 14] = [
    "put",
    "put_scaled",
    "prep",
    "prep_scaled",
    "avg",
    "w_avg",
    "mask",
    "w_mask",
    "blend",
    "blend_v",
    "blend_h",
    "warp8x8",
    "warp8x8t",
    "emu_edge",
];

const REC_HEADER_BYTES: usize = 80;
const CALL_PARAMS: usize = 8;
const CALL_BYTES: usize = CALL_PARAMS * 4;

/// Bütün blok genişlikleri (dav1d `src/levels.h` blok boyutu tablosu).
const WIDTHS: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4) — test yardımcısı; harici crate kullanılmaz.
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
        for i in 0..16 {
            w[i] = u32::from_be_bytes([p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]]);
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
        while !data.is_empty() {
            let take = (64 - self.fill).min(data.len());
            self.buf[self.fill..self.fill + take].copy_from_slice(&data[..take]);
            self.fill += take;
            data = &data[take..];
            if self.fill == 64 {
                let block = self.buf;
                self.block(&block);
                self.fill = 0;
            }
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.fill != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        let mut out = [0u8; 32];
        for (i, v) in self.state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }
}

/// SHA-256'nın doğru çalıştığını kanıtlayan sabit vektör (FIPS 180-4).
#[test]
fn sha256_sabit_vektor() {
    let mut h = Sha256::new();
    h.update(b"abc");
    assert_eq!(
        h.finish(),
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ]
    );
    let mut h = Sha256::new();
    h.update(b"");
    assert_eq!(
        h.finish(),
        [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ]
    );
}

// ---------------------------------------------------------------------------
// Fixture okuyucu
// ---------------------------------------------------------------------------

struct Fixture {
    r: BufReader<File>,
    bpc: u8,
}

impl Fixture {
    fn open(path: &Path, magic: &[u8; 8]) -> Self {
        let f = File::open(path)
            .unwrap_or_else(|e| panic!("fixture açılamadı {}: {e}", path.display()));
        let mut r = BufReader::with_capacity(1 << 20, f);
        let mut head = [0u8; 16];
        r.read_exact(&mut head).expect("başlık");
        assert_eq!(&head[..8], magic, "fixture büyüsü: {}", path.display());
        let version = u32::from_le_bytes([head[8], head[9], head[10], head[11]]);
        assert_eq!(version, 1, "fixture sürümü");
        let bpc = u32::from_le_bytes([head[12], head[13], head[14], head[15]]);
        assert!(bpc == 8 || bpc == 10 || bpc == 12, "bpc={bpc}");
        Self { r, bpc: bpc as u8 }
    }

    fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b).expect("u32");
        u32::from_le_bytes(b)
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        self.r.read_exact(&mut v).expect("bayt");
        v
    }

    /// Bir kaydı okur; dosya bittiyse `None`.
    fn next_record(&mut self) -> Option<Record> {
        let mut head = [0u8; REC_HEADER_BYTES];
        match self.r.read_exact(&mut head) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => panic!("kayıt başlığı: {e}"),
        }
        let g = |o: usize| u32::from_le_bytes([head[o], head[o + 1], head[o + 2], head[o + 3]]);
        let gi = |o: usize| i32::from_le_bytes([head[o], head[o + 1], head[o + 2], head[o + 3]]);
        let family = g(0);
        let subtype = g(4);
        let w = g(8) as usize;
        let h = g(12) as usize;
        let n_calls = g(16) as usize;
        let strides = [g(20) as usize, g(24) as usize, g(28) as usize];
        let lens = [g(32) as usize, g(36) as usize, g(40) as usize];
        let out_len = g(44) as usize;
        let aux = [gi(48), gi(52), gi(56), gi(60)];
        let cell_w = g(64) as usize;
        assert!(family < FAM_COUNT, "bilinmeyen aile {family}");
        assert!(n_calls >= 1, "n_calls");
        let mut params = Vec::with_capacity(n_calls);
        let mut raw = vec![0u8; n_calls * CALL_BYTES];
        self.r.read_exact(&mut raw).expect("çağrı parametreleri");
        for c in 0..n_calls {
            let mut p = [0i32; CALL_PARAMS];
            for (k, slot) in p.iter_mut().enumerate() {
                let o = c * CALL_BYTES + k * 4;
                *slot = i32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
            }
            params.push(p);
        }
        let planes = [
            self.bytes(lens[0]),
            self.bytes(lens[1]),
            self.bytes(lens[2]),
        ];
        Some(Record {
            family,
            subtype,
            w,
            h,
            n_calls,
            strides,
            out_len,
            aux,
            cell_w: if cell_w == 0 { w } else { cell_w },
            params,
            planes,
        })
    }
}

struct Record {
    family: u32,
    subtype: u32,
    w: usize,
    h: usize,
    n_calls: usize,
    strides: [usize; 3],
    out_len: usize,
    aux: [i32; 4],
    cell_w: usize,
    params: Vec<[i32; CALL_PARAMS]>,
    planes: [Vec<u8>; 3],
}

/// 32-bit `i16` yükünü çözer (küçük sonlu).
fn tmp_plane(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Satır satır `dst`ten `out`a kopyalar (C oracle `copy_rows_out` gibi).
fn copy_rows(dst: &[u8], dst_stride: usize, out: &mut [u8], w: usize, h: usize, bpp: usize) {
    for y in 0..h {
        let d = y * dst_stride;
        out[y * w * bpp..(y + 1) * w * bpp].copy_from_slice(&dst[d..d + w * bpp]);
    }
}

fn bit_depth(bpc: u8) -> BitDepth {
    BitDepth::from_bits(u32::from(bpc)).expect("8/10/12")
}

/// Büyük kümede AVX2 yuvalarıyla karşılaştırılan aileler (P32 `put`/`prep`,
/// P33 bileşik, karışım, warp, `emu_edge` ve ölçekli).
#[cfg(target_arch = "x86_64")]
const AVX2_AILELER: &[u32] = &[
    FAM_PUT,
    FAM_PREP,
    FAM_AVG,
    FAM_W_AVG,
    FAM_MASK,
    FAM_W_MASK,
    FAM_BLEND,
    FAM_BLEND_V,
    FAM_BLEND_H,
    FAM_WARP,
    FAM_WARP_T,
    FAM_EMU_EDGE,
    FAM_PUT_SCALED,
    FAM_PREP_SCALED,
];

/// ISA yuvaları (P32–P34 AVX2, P35 SSSE3): gövdesi olan yuvalar `Some`, öbürleri
/// skaler tabloda kalır; `aileler8`/`aileler16` büyük kümede karşılaştırılan aileler.
#[derive(Clone, Copy)]
struct SimdYuvalari {
    ad: &'static str,
    aileler8: &'static [u32],
    aileler16: &'static [u32],
    put: Option<[McPutFn; 10]>,
    prep: Option<[McPrepFn; 10]>,
    avg: Option<av1oxide::dsp::mc::AvgFn>,
    w_avg: Option<av1oxide::dsp::mc::WAvgFn>,
    mask: Option<av1oxide::dsp::mc::MaskFn>,
    w_mask: Option<[av1oxide::dsp::mc::WMaskFn; 3]>,
    blend: Option<av1oxide::dsp::mc::BlendFn>,
    blend_v: Option<av1oxide::dsp::mc::BlendDirFn>,
    blend_h: Option<av1oxide::dsp::mc::BlendDirFn>,
    warp8x8: Option<av1oxide::dsp::mc::Warp8x8Fn>,
    warp8x8t: Option<av1oxide::dsp::mc::Warp8x8tFn>,
    emu_edge: Option<av1oxide::dsp::mc::EmuEdgeFn>,
    mc_scaled: Option<[av1oxide::dsp::mc::McPutScaledFn; 10]>,
    mct_scaled: Option<[av1oxide::dsp::mc::McPrepScaledFn; 10]>,
    put16: Option<[McPutFn; 10]>,
    prep16: Option<[McPrepFn; 10]>,
    avg16: Option<av1oxide::dsp::mc::AvgFn>,
    w_avg16: Option<av1oxide::dsp::mc::WAvgFn>,
    mask16: Option<av1oxide::dsp::mc::MaskFn>,
    w_mask16: Option<[av1oxide::dsp::mc::WMaskFn; 3]>,
    blend16: Option<av1oxide::dsp::mc::BlendFn>,
    blend_v16: Option<av1oxide::dsp::mc::BlendDirFn>,
    blend_h16: Option<av1oxide::dsp::mc::BlendDirFn>,
    warp8x8_16: Option<av1oxide::dsp::mc::Warp8x8Fn>,
    warp8x8t_16: Option<av1oxide::dsp::mc::Warp8x8tFn>,
    emu_edge16: Option<av1oxide::dsp::mc::EmuEdgeFn>,
    mc_scaled16: Option<[av1oxide::dsp::mc::McPutScaledFn; 10]>,
    mct_scaled16: Option<[av1oxide::dsp::mc::McPrepScaledFn; 10]>,
}

impl SimdYuvalari {
    fn aileler(&self, bpc: u8) -> &'static [u32] {
        if bpc == 8 {
            self.aileler8
        } else {
            self.aileler16
        }
    }
}

/// P34: 16 bpc AVX2 yuvası olan aileler (10/12 bit kümelerinde AVX2 geçişi).
#[cfg(target_arch = "x86_64")]
const AVX2_AILELER_16: &[u32] = &[
    FAM_PUT,
    FAM_PREP,
    FAM_AVG,
    FAM_W_AVG,
    FAM_MASK,
    FAM_W_MASK,
    FAM_BLEND,
    FAM_BLEND_V,
    FAM_BLEND_H,
    FAM_WARP,
    FAM_WARP_T,
    FAM_EMU_EDGE,
    FAM_PUT_SCALED,
    FAM_PREP_SCALED,
];

/// AVX2 yuvaları; AVX2 yoksa `None`.
fn avx2_yuvalari() -> Option<SimdYuvalari> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        use av1oxide::dsp::x86::mc::{prep_fn, put_fn};
        use av1oxide::dsp::x86::{mc_bilesik as b, mc_karisim as k, mc_kenar as kn, mc_warp as wp};
        return Some(SimdYuvalari {
            ad: " avx2",
            aileler8: AVX2_AILELER,
            aileler16: AVX2_AILELER_16,
            put: Some(Filter2d::ALL.map(|f| put_fn(t, f))),
            prep: Some(Filter2d::ALL.map(|f| prep_fn(t, f))),
            avg: Some(b::avg_fn(t)),
            w_avg: Some(b::w_avg_fn(t)),
            mask: Some(b::mask_fn(t)),
            w_mask: Some([b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)]),
            blend: Some(k::blend_fn(t)),
            blend_v: Some(k::blend_v_fn(t)),
            blend_h: Some(k::blend_h_fn(t)),
            warp8x8: Some(wp::warp8x8_fn(t)),
            warp8x8t: Some(wp::warp8x8t_fn(t)),
            emu_edge: Some(kn::emu_edge_fn(t)),
            mc_scaled: Some(
                Filter2d::ALL.map(|f| av1oxide::dsp::x86::mc_olcekli::put_scaled_fn(t, f)),
            ),
            mct_scaled: Some(
                Filter2d::ALL.map(|f| av1oxide::dsp::x86::mc_olcekli::prep_scaled_fn(t, f)),
            ),
            put16: Some(Filter2d::ALL.map(|f| av1oxide::dsp::x86::mc16::put16_fn(t, f))),
            prep16: Some(Filter2d::ALL.map(|f| av1oxide::dsp::x86::mc16::prep16_fn(t, f))),
            avg16: Some(av1oxide::dsp::x86::mc16_bilesik::avg_fn(t)),
            w_avg16: Some(av1oxide::dsp::x86::mc16_bilesik::w_avg_fn(t)),
            mask16: Some(av1oxide::dsp::x86::mc16_bilesik::mask_fn(t)),
            w_mask16: Some([0, 1, 2].map(|i| av1oxide::dsp::x86::mc16_bilesik::w_mask_fn(t, i))),
            blend16: Some(av1oxide::dsp::x86::mc16_karisim::blend_fn(t)),
            blend_v16: Some(av1oxide::dsp::x86::mc16_karisim::blend_v_fn(t)),
            blend_h16: Some(av1oxide::dsp::x86::mc16_karisim::blend_h_fn(t)),
            warp8x8_16: Some(av1oxide::dsp::x86::mc16_warp::warp8x8_fn(t)),
            warp8x8t_16: Some(av1oxide::dsp::x86::mc16_warp::warp8x8t_fn(t)),
            emu_edge16: Some(av1oxide::dsp::x86::mc16_kenar::emu_edge_fn(t)),
            mc_scaled16: Some(
                Filter2d::ALL.map(|f| av1oxide::dsp::x86::mc16_olcekli::put_scaled_fn(t, f)),
            ),
            mct_scaled16: Some(
                Filter2d::ALL.map(|f| av1oxide::dsp::x86::mc16_olcekli::prep_scaled_fn(t, f)),
            ),
        });
    }
    None
}

/// P35: SSSE3 yuvaları; SSSE3 yoksa `None`.
fn ssse3_yuvalari() -> Option<SimdYuvalari> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).ssse3_token()
    {
        use av1oxide::dsp::x86::{
            mc_bilesik_ssse3 as b8, mc_karisim_ssse3 as k8, mc_kenar_ssse3 as e8,
            mc_olcekli_ssse3 as o8, mc_ssse3 as m8, mc_warp_ssse3 as w8, mc16_bilesik_ssse3 as b16,
            mc16_karisim_ssse3 as k16, mc16_kenar_ssse3 as e16, mc16_olcekli_ssse3 as o16,
            mc16_ssse3 as m16, mc16_warp_ssse3 as w16,
        };
        return Some(SimdYuvalari {
            ad: " ssse3",
            aileler8: SSSE3_AILELER,
            aileler16: SSSE3_AILELER_16,
            put: Some(Filter2d::ALL.map(|f| m8::put_fn(t, f))),
            prep: Some(Filter2d::ALL.map(|f| m8::prep_fn(t, f))),
            put16: Some(Filter2d::ALL.map(|f| m16::put16_fn(t, f))),
            prep16: Some(Filter2d::ALL.map(|f| m16::prep16_fn(t, f))),
            avg: Some(b8::avg_fn(t)),
            w_avg: Some(b8::w_avg_fn(t)),
            mask: Some(b8::mask_fn(t)),
            w_mask: Some([0, 1, 2].map(|i| b8::w_mask_fn(t, i))),
            blend: Some(k8::blend_fn(t)),
            blend_v: Some(k8::blend_v_fn(t)),
            blend_h: Some(k8::blend_h_fn(t)),
            warp8x8: Some(w8::warp8x8_fn(t)),
            warp8x8t: Some(w8::warp8x8t_fn(t)),
            emu_edge: Some(e8::emu_edge_fn(t)),
            mc_scaled: Some(Filter2d::ALL.map(|f| o8::put_scaled_fn(t, f))),
            mct_scaled: Some(Filter2d::ALL.map(|f| o8::prep_scaled_fn(t, f))),
            avg16: Some(b16::avg_fn(t)),
            w_avg16: Some(b16::w_avg_fn(t)),
            mask16: Some(b16::mask_fn(t)),
            w_mask16: Some([0, 1, 2].map(|i| b16::w_mask_fn(t, i))),
            blend16: Some(k16::blend_fn(t)),
            blend_v16: Some(k16::blend_v_fn(t)),
            blend_h16: Some(k16::blend_h_fn(t)),
            warp8x8_16: Some(w16::warp8x8_fn(t)),
            warp8x8t_16: Some(w16::warp8x8t_fn(t)),
            emu_edge16: Some(e16::emu_edge_fn(t)),
            mc_scaled16: Some(Filter2d::ALL.map(|f| o16::put_scaled_fn(t, f))),
            mct_scaled16: Some(Filter2d::ALL.map(|f| o16::prep_scaled_fn(t, f))),
        });
    }
    None
}

/// P35: SSSE3 yuvası olan aileler (8 bpc ve 10/12 bit).
#[cfg(target_arch = "x86_64")]
const SSSE3_AILELER: &[u32] = &[
    FAM_PUT,
    FAM_PREP,
    FAM_AVG,
    FAM_W_AVG,
    FAM_MASK,
    FAM_W_MASK,
    FAM_BLEND,
    FAM_BLEND_V,
    FAM_BLEND_H,
    FAM_WARP,
    FAM_WARP_T,
    FAM_EMU_EDGE,
    FAM_PUT_SCALED,
    FAM_PREP_SCALED,
];
#[cfg(target_arch = "x86_64")]
const SSSE3_AILELER_16: &[u32] = &[
    FAM_PUT,
    FAM_PREP,
    FAM_AVG,
    FAM_W_AVG,
    FAM_MASK,
    FAM_W_MASK,
    FAM_BLEND,
    FAM_BLEND_V,
    FAM_BLEND_H,
    FAM_WARP,
    FAM_WARP_T,
    FAM_EMU_EDGE,
    FAM_PUT_SCALED,
    FAM_PREP_SCALED,
];

/// P35: SSE4.1 yuvaları (yalnız 8 bpc warp); SSE4.1 yoksa `None`.
fn sse41_yuvalari() -> Option<SimdYuvalari> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).sse41_token()
    {
        use av1oxide::dsp::x86::mc_warp_sse41 as w8;
        return Some(SimdYuvalari {
            ad: " sse41",
            aileler8: SSE41_AILELER,
            aileler16: &[],
            put: None,
            prep: None,
            avg: None,
            w_avg: None,
            mask: None,
            w_mask: None,
            blend: None,
            blend_v: None,
            blend_h: None,
            warp8x8: Some(w8::warp8x8_fn(t)),
            warp8x8t: Some(w8::warp8x8t_fn(t)),
            emu_edge: None,
            mc_scaled: None,
            mct_scaled: None,
            put16: None,
            prep16: None,
            avg16: None,
            w_avg16: None,
            mask16: None,
            w_mask16: None,
            blend16: None,
            blend_v16: None,
            blend_h16: None,
            warp8x8_16: None,
            warp8x8t_16: None,
            emu_edge16: None,
            mc_scaled16: None,
            mct_scaled16: None,
        });
    }
    None
}

/// P36: AVX512ICL yuvaları (8 bpc ve 10/12 bit put/prep, bileşik, karışım, warp);
/// AVX512ICL gövdesi olmayan aileler (`emu_edge`, ölçekli) yok. AVX512ICL yoksa `None`.
fn avx512_yuvalari() -> Option<SimdYuvalari> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx512icl_token()
    {
        use av1oxide::dsp::x86::{
            mc_avx512 as m8, mc_bilesik_avx512 as b8, mc_karisim_avx512 as k8,
            mc_warp_avx512 as w8, mc16_avx512 as m16, mc16_bilesik_avx512 as b16,
            mc16_karisim_avx512 as k16, mc16_warp_avx512 as w16,
        };
        return Some(SimdYuvalari {
            ad: " avx512icl",
            aileler8: AVX512_AILELER,
            aileler16: AVX512_AILELER,
            put: Some(Filter2d::ALL.map(|f| m8::put_fn(t, f))),
            prep: Some(Filter2d::ALL.map(|f| m8::prep_fn(t, f))),
            put16: Some(Filter2d::ALL.map(|f| m16::put16_fn(t, f))),
            prep16: Some(Filter2d::ALL.map(|f| m16::prep16_fn(t, f))),
            avg: Some(b8::avg_fn(t)),
            w_avg: Some(b8::w_avg_fn(t)),
            mask: Some(b8::mask_fn(t)),
            w_mask: Some([0, 1, 2].map(|i| b8::w_mask_fn(t, i))),
            blend: Some(k8::blend_fn(t)),
            blend_v: Some(k8::blend_v_fn(t)),
            blend_h: Some(k8::blend_h_fn(t)),
            warp8x8: Some(w8::warp8x8_fn(t)),
            warp8x8t: Some(w8::warp8x8t_fn(t)),
            emu_edge: None,
            mc_scaled: None,
            mct_scaled: None,
            avg16: Some(b16::avg_fn(t)),
            w_avg16: Some(b16::w_avg_fn(t)),
            mask16: Some(b16::mask_fn(t)),
            w_mask16: Some([0, 1, 2].map(|i| b16::w_mask_fn(t, i))),
            blend16: Some(k16::blend_fn(t)),
            blend_v16: Some(k16::blend_v_fn(t)),
            blend_h16: Some(k16::blend_h_fn(t)),
            warp8x8_16: Some(w16::warp8x8_fn(t)),
            warp8x8t_16: Some(w16::warp8x8t_fn(t)),
            emu_edge16: None,
            mc_scaled16: None,
            mct_scaled16: None,
        });
    }
    None
}

/// P36: AVX512ICL yuvası olan aileler (8 bpc ve 10/12 bit).
#[cfg(target_arch = "x86_64")]
const AVX512_AILELER: &[u32] = &[
    FAM_PUT,
    FAM_PREP,
    FAM_AVG,
    FAM_W_AVG,
    FAM_MASK,
    FAM_W_MASK,
    FAM_BLEND,
    FAM_BLEND_V,
    FAM_BLEND_H,
    FAM_WARP,
    FAM_WARP_T,
];

/// P35: SSE4.1 yuvası olan aileler (8 bpc).
#[cfg(target_arch = "x86_64")]
const SSE41_AILELER: &[u32] = &[FAM_WARP, FAM_WARP_T];

/// Bir kaydın bütün çağrılarını koşturur ve çıktıları sırayla `sink`e verir.
///
/// `avx2` verilirse 8 bpc `put`/`prep` (P32) ve kalan aileler (P33), 10/12
/// bitte P34 yuvası olan aileler (`AVX2_AILELER_16`) AVX2 yuvalarından geçer;
/// kaynak dilimine kopyasız sözleşmenin kuyruğu (`MC_KUYRUK`) test tarafında
/// eklenir.
fn run_record(rec: &Record, bpc: u8, avx2: Option<&SimdYuvalari>, sink: &mut dyn FnMut(&[u8])) {
    let bd = bit_depth(bpc);
    let bdmax = ((1u32 << bpc) - 1) as u16;
    let mut dsp: McDsp = McDsp::scalar_for(bd);
    let kuyruklu;
    let mut src_k: &[u8] = &rec.planes[0];
    if let Some(a) = avx2.filter(|_| bpc != 8) {
        if let Some(f) = a.put16 {
            dsp.mc = f;
        }
        if let Some(f) = a.prep16 {
            dsp.mct = f;
        }
        if let Some(f) = a.avg16 {
            dsp.avg = f;
        }
        if let Some(f) = a.w_avg16 {
            dsp.w_avg = f;
        }
        if let Some(f) = a.mask16 {
            dsp.mask = f;
        }
        if let Some(f) = a.w_mask16 {
            dsp.w_mask = f;
        }
        if let Some(f) = a.blend16 {
            dsp.blend = f;
        }
        if let Some(f) = a.blend_v16 {
            dsp.blend_v = f;
        }
        if let Some(f) = a.blend_h16 {
            dsp.blend_h = f;
        }
        if let Some(f) = a.warp8x8_16 {
            dsp.warp8x8 = f;
        }
        if let Some(f) = a.warp8x8t_16 {
            dsp.warp8x8t = f;
        }
        if let Some(f) = a.emu_edge16 {
            dsp.emu_edge = f;
        }
        if let Some(f) = a.mc_scaled16 {
            dsp.mc_scaled = f;
        }
        if let Some(f) = a.mct_scaled16 {
            dsp.mct_scaled = f;
        }
        let mut v = rec.planes[0].clone();
        v.resize(v.len() + MC_KUYRUK, 0);
        kuyruklu = v;
        src_k = &kuyruklu;
    } else if let Some(a) = avx2 {
        if let Some(f) = a.put {
            dsp.mc = f;
        }
        if let Some(f) = a.prep {
            dsp.mct = f;
        }
        if let Some(f) = a.avg {
            dsp.avg = f;
        }
        if let Some(f) = a.w_avg {
            dsp.w_avg = f;
        }
        if let Some(f) = a.mask {
            dsp.mask = f;
        }
        if let Some(f) = a.w_mask {
            dsp.w_mask = f;
        }
        if let Some(f) = a.blend {
            dsp.blend = f;
        }
        if let Some(f) = a.blend_v {
            dsp.blend_v = f;
        }
        if let Some(f) = a.blend_h {
            dsp.blend_h = f;
        }
        if let Some(f) = a.warp8x8 {
            dsp.warp8x8 = f;
        }
        if let Some(f) = a.warp8x8t {
            dsp.warp8x8t = f;
        }
        if let Some(f) = a.emu_edge {
            dsp.emu_edge = f;
        }
        if let Some(f) = a.mc_scaled {
            dsp.mc_scaled = f;
        }
        if let Some(f) = a.mct_scaled {
            dsp.mct_scaled = f;
        }
        let mut v = rec.planes[0].clone();
        v.resize(v.len() + MC_KUYRUK, 0);
        kuyruklu = v;
        src_k = &kuyruklu;
    }
    let bpp = if bpc == 8 { 1 } else { 2 };
    let w = rec.w;
    let h = rec.h;
    let dst_stride = rec.strides[1];
    let src_stride = rec.strides[0];
    let src = src_k;
    let dst_len = (h.saturating_sub(1)) * dst_stride + w * bpp;
    let mut out: Vec<u8> = vec![0u8; rec.out_len];
    let mut dst: Vec<u8> = vec![0u8; dst_len.max(rec.out_len)];

    for p in &rec.params {
        match rec.family {
            FAM_PUT => {
                let filter = Filter2d::from_index(rec.subtype as usize).expect("filtre");
                let _ = filter;
                dst.iter_mut().for_each(|b| *b = 0);
                dsp.mc[rec.subtype as usize](
                    &mut dst[..dst_len],
                    dst_stride,
                    src,
                    src_stride,
                    w,
                    h,
                    p[1] as u8,
                    p[2] as u8,
                    bd,
                    bdmax,
                );
                copy_rows(&dst, dst_stride, &mut out, w, h, bpp);
            }
            FAM_PUT_SCALED => {
                dst.iter_mut().for_each(|b| *b = 0);
                dsp.mc_scaled[rec.subtype as usize](
                    &mut dst[..dst_len],
                    dst_stride,
                    src,
                    src_stride,
                    w,
                    h,
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                    bd,
                    bdmax,
                );
                copy_rows(&dst, dst_stride, &mut out, w, h, bpp);
            }
            FAM_PREP => {
                let mut tmp = vec![0i16; w * h];
                dsp.mct[rec.subtype as usize](
                    &mut tmp, src, src_stride, w, h, p[1] as u8, p[2] as u8, bd, bdmax,
                );
                for (i, v) in tmp.iter().enumerate() {
                    out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            FAM_PREP_SCALED => {
                let mut tmp = vec![0i16; w * h];
                dsp.mct_scaled[rec.subtype as usize](
                    &mut tmp, src, src_stride, w, h, p[1], p[2], p[3], p[4], bd, bdmax,
                );
                for (i, v) in tmp.iter().enumerate() {
                    out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            FAM_AVG | FAM_W_AVG | FAM_MASK | FAM_W_MASK => {
                let tmp1 = tmp_plane(&rec.planes[0]);
                let tmp2 = tmp_plane(&rec.planes[1]);
                dst.iter_mut().for_each(|b| *b = 0);
                match rec.family {
                    FAM_AVG => {
                        (dsp.avg)(
                            &mut dst[..dst_len],
                            dst_stride,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            bd,
                            bdmax,
                        );
                    }
                    FAM_W_AVG => {
                        (dsp.w_avg)(
                            &mut dst[..dst_len],
                            dst_stride,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            p[0],
                            bd,
                            bdmax,
                        );
                    }
                    FAM_MASK => {
                        (dsp.mask)(
                            &mut dst[..dst_len],
                            dst_stride,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            &rec.planes[2],
                            bd,
                            bdmax,
                        );
                    }
                    _ => {
                        let ss_hor = rec.subtype != 0;
                        let ss_ver = rec.subtype == 2;
                        let mw = w >> usize::from(ss_hor);
                        let mh = h >> usize::from(ss_ver);
                        let mut mask = vec![0u8; mw * mh];
                        dsp.w_mask[rec.subtype as usize](
                            &mut dst[..dst_len],
                            dst_stride,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            &mut mask,
                            p[0],
                            ss_hor,
                            ss_ver,
                            bd,
                            bdmax,
                        );
                        copy_rows(&dst, dst_stride, &mut out, w, h, bpp);
                        let moff = w * h * bpp;
                        out[moff..moff + mw * mh].copy_from_slice(&mask);
                    }
                }
                if rec.family == FAM_AVG || rec.family == FAM_W_AVG || rec.family == FAM_MASK {
                    copy_rows(&dst, dst_stride, &mut out, w, h, bpp);
                }
            }
            FAM_BLEND | FAM_BLEND_V | FAM_BLEND_H => {
                // Mevcut hedef `plane0`dan kopyalanır (C oracle gibi).
                dst.iter_mut().for_each(|b| *b = 0);
                for y in 0..h {
                    let s = y * src_stride;
                    let d = y * dst_stride;
                    dst[d..d + w * bpp].copy_from_slice(&src[s..s + w * bpp]);
                }
                match rec.family {
                    FAM_BLEND => {
                        (dsp.blend)(
                            &mut dst,
                            dst_stride,
                            &rec.planes[1],
                            w,
                            h,
                            &rec.planes[2],
                            bd,
                            bdmax,
                        );
                    }
                    FAM_BLEND_V => {
                        (dsp.blend_v)(&mut dst, dst_stride, &rec.planes[1], w, h, bd, bdmax);
                    }
                    _ => {
                        (dsp.blend_h)(&mut dst, dst_stride, &rec.planes[1], w, h, bd, bdmax);
                    }
                }
                copy_rows(&dst, dst_stride, &mut out, w, h, bpp);
            }
            FAM_WARP => {
                let abcd = [p[2] as i16, p[3] as i16, p[4] as i16, p[5] as i16];
                dst.iter_mut().for_each(|b| *b = 0);
                (dsp.warp8x8)(
                    &mut dst, dst_stride, src, src_stride, &abcd, p[0], p[1], bd, bdmax,
                );
                copy_rows(&dst, dst_stride, &mut out, 8, 8, bpp);
            }
            FAM_WARP_T => {
                let abcd = [p[2] as i16, p[3] as i16, p[4] as i16, p[5] as i16];
                let mut tmp = vec![0i16; 64];
                (dsp.warp8x8t)(&mut tmp, 8, src, src_stride, &abcd, p[0], p[1], bd, bdmax);
                for (i, v) in tmp.iter().enumerate() {
                    out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            FAM_EMU_EDGE => {
                dst.iter_mut().for_each(|b| *b = 0);
                (dsp.emu_edge)(
                    w,
                    h,
                    rec.aux[0] as usize,
                    rec.aux[1] as usize,
                    rec.aux[2],
                    rec.aux[3],
                    &mut dst,
                    dst_stride,
                    src,
                    src_stride,
                    bd,
                    bdmax,
                );
                copy_rows(&dst, dst_stride, &mut out, w, h, bpp);
            }
            other => panic!("aile {other}"),
        }
        sink(&out);
    }
}

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
        .join("mc-family")
}

fn large_dir() -> PathBuf {
    workspace_root().join(".cache").join("mc-family")
}

/// Tablo bayt-eşitliği: dav1d'nin derlenmiş tablolarının ham dökümü.
#[test]
fn tablolar_dav1d_ile_bayt_esit() {
    use av1oxide::dsp::mc_tables::{MC_SUBPEL_FILTERS, MC_WARP_FILTER, OBMC_MASKS};
    let path = small_dir().join("mc-tables.bin");
    let data =
        std::fs::read(&path).unwrap_or_else(|e| panic!("tablo dökümü yok {}: {e}", path.display()));
    assert_eq!(&data[..8], b"AV1OMCTB", "tablo büyüsü");
    let count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    assert_eq!(count, 1, "tablo dökümü sürümü");
    let n = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    assert_eq!(n, 3, "tablo sayısı");
    let mut pos = 16usize;
    let mut tables = Vec::new();
    for _ in 0..n {
        let len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        tables.push(data[pos..pos + len].to_vec());
        pos += len;
    }
    assert_eq!(pos, data.len(), "döküm tamamen tüketilmeli");

    let mut subpel = Vec::new();
    for table in &MC_SUBPEL_FILTERS {
        for row in table {
            for v in row {
                subpel.push(*v as u8);
            }
        }
    }
    let mut warp = Vec::new();
    for row in &MC_WARP_FILTER {
        for v in row {
            warp.push(*v as u8);
        }
    }
    println!("  dav1d_mc_subpel_filters : {} bayt", tables[0].len());
    println!("  dav1d_mc_warp_filter    : {} bayt", tables[1].len());
    println!("  dav1d_obmc_masks        : {} bayt", tables[2].len());
    assert_eq!(tables[0].len(), 720);
    assert_eq!(tables[1].len(), 1544);
    assert_eq!(tables[2].len(), 64);
    assert_eq!(subpel, tables[0], "subpel tablosu bayt-eşit değil");
    assert_eq!(warp, tables[1], "warp tablosu bayt-eşit değil");
    assert_eq!(
        OBMC_MASKS.to_vec(),
        tables[2],
        "OBMC maskeleri bayt-eşit değil"
    );
    println!("  üç tablo dav1d ile bayt-eşit (720 + 1544 + 64 = 2 328 bayt)");
}

/// Küçük (git) küme — **tam bayt** karşılaştırması (`avx2`: P32 yuvaları).
fn compare_small(bpc: u8, avx2: Option<&SimdYuvalari>) -> usize {
    let input = small_dir().join(format!("mc-family-small-{bpc}-input.bin"));
    let expect = small_dir().join(format!("mc-family-small-{bpc}-expect.bin"));
    let mut inp = Fixture::open(&input, b"AV1OMCF1");
    let mut exp = Fixture::open(&expect, b"AV1OMCFO");
    assert_eq!(inp.bpc, bpc);
    assert_eq!(exp.bpc, bpc);
    let mut index = 0usize;
    let mut cells: BTreeMap<(u32, u32, usize), usize> = BTreeMap::new();
    loop {
        let Some(rec) = inp.next_record() else {
            break;
        };
        let total = exp.u32() as usize;
        assert_eq!(
            total,
            rec.n_calls * rec.out_len,
            "kayıt {index}: beklenen uzunluk"
        );
        let mut call = 0usize;
        let mut sink = |got: &[u8]| {
            let want = exp.bytes(rec.out_len);
            if got != want.as_slice() {
                let at = got
                    .iter()
                    .zip(want.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                panic!(
                    "küçük küme {bpc}-bit: kayıt {index} ({}) alt tür {}, w={}, h={}, çağrı \
                     {call}: ilk fark baytı {at}\n  skaler={:02x?}\n  dav1d ={:02x?}",
                    FAM_NAMES[rec.family as usize],
                    rec.subtype,
                    rec.w,
                    rec.h,
                    &got[at.saturating_sub(4)..(at + 8).min(got.len())],
                    &want[at.saturating_sub(4)..(at + 8).min(want.len())],
                );
            }
            call += 1;
        };
        run_record(&rec, bpc, avx2, &mut sink);
        assert_eq!(call, rec.n_calls);
        *cells.entry((rec.family, rec.subtype, rec.w)).or_insert(0) += rec.n_calls;
        index += 1;
    }
    println!(
        "  [small {bpc}-bit{}] {index} kayıt · {} hücre · tam bayt eşleşti",
        avx2.map_or("", |a| a.ad),
        cells.len()
    );
    index
}

/// 8(a) — küçük sınır kümesi (her koşuda, üç bit derinliği).
#[test]
fn kucuk_kume_bit_exact() {
    let mut total = 0;
    for bpc in [8u8, 10, 12] {
        total += compare_small(bpc, None);
    }
    println!("[small] TOPLAM {total} kayıt · tam bayt eşleşti");
    assert!(total > 1000, "küçük küme beklenenden küçük: {total}");
    // P32–P34 AVX2 ve P35 SSSE3/SSE4.1 yuvaları aynı kümede dav1d C ile.
    for (yuv, isim, neden) in [
        (avx2_yuvalari(), "avx2", Neden::avx2()),
        (ssse3_yuvalari(), "ssse3", Neden::isa()),
        (sse41_yuvalari(), "sse41", Neden::isa()),
        (avx512_yuvalari(), "avx512icl", Neden::isa()),
    ] {
        match yuv {
            Some(a) => {
                let n = compare_small(8, Some(&a));
                println!("[small {isim}] 8 bpc yuvaları: {n} kayıt · tam bayt eşleşti");
                let n16 = compare_small(10, Some(&a)) + compare_small(12, Some(&a));
                println!(
                    "[small {isim}] 16 bpc (10/12 bit) yuvaları: {n16} kayıt · tam bayt eşleşti"
                );
            }
            None => on_kosul::eksik(
                &format!("mc_family_oracle::kucuk_kume_bit_exact[{isim}]"),
                neden,
                "MC ISA yuvaları işlemcinin ISA düzeyini ister",
            ),
        }
    }
}

/// Büyük küme — hücre başına SHA-256 (`.cache/`, `--ignored`). `avx2`
/// verilirse yalnız `put`/`prep` hücreleri (AVX2 yuvaları) karşılaştırılır.
fn compare_large(bpc: u8, avx2: Option<&SimdYuvalari>) -> (usize, usize, usize) {
    let input = large_dir().join(format!("mc-family-large-{bpc}-input.bin"));
    let expect = large_dir().join(format!("mc-family-large-{bpc}-expect.bin"));
    let mut inp = Fixture::open(&input, b"AV1OMCF1");
    let mut exp = Fixture::open(&expect, b"AV1OMCF2");
    assert_eq!(inp.bpc, bpc);
    assert_eq!(exp.bpc, bpc);
    let n_cells = exp.u32() as usize;

    let mut hashes: BTreeMap<(u32, u32, usize), (usize, Sha256)> = BTreeMap::new();
    let mut records = 0usize;
    loop {
        let Some(rec) = inp.next_record() else {
            break;
        };
        if avx2.is_some_and(|a| !a.aileler(bpc).contains(&rec.family)) {
            continue;
        }
        let key = (rec.family, rec.subtype, rec.cell_w);
        let entry = hashes.entry(key).or_insert_with(|| (0, Sha256::new()));
        entry.0 += rec.n_calls;
        let mut sink = |got: &[u8]| entry.1.update(got);
        run_record(&rec, bpc, avx2, &mut sink);
        records += 1;
    }

    // Beklenen özetler (dosyadaki sıra: aile, alt tür, genişlik).
    let mut expected: BTreeMap<(u32, u32, usize), (usize, [u8; 32])> = BTreeMap::new();
    for _ in 0..n_cells {
        let family = exp.u32();
        let subtype = exp.u32();
        let w = exp.u32() as usize;
        let calls = exp.u32() as usize;
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&exp.bytes(32));
        if avx2.is_some_and(|a| !a.aileler(bpc).contains(&family)) {
            continue;
        }
        expected.insert((family, subtype, w), (calls, digest));
    }

    assert_eq!(
        hashes.len(),
        expected.len(),
        "{bpc}-bit: hücre sayısı iki tarafta farklı"
    );
    let mut calls_total = 0usize;
    let mut matched = 0usize;
    let mut cells_small = Vec::new();
    for (key, (calls, sha)) in hashes {
        let (want_calls, want_digest) = expected
            .remove(&key)
            .unwrap_or_else(|| panic!("{bpc}-bit: beklenmeyen hücre {key:?}"));
        assert_eq!(calls, want_calls, "{bpc}-bit: hücre {key:?} çağrı sayısı");
        calls_total += calls;
        if calls < 1000 {
            cells_small.push(format!("{key:?}={calls}"));
        }
        let got = sha.finish();
        if got != want_digest {
            panic!(
                "{bpc}-bit: hücre {key:?} ({}) SHA-256 uyuşmadı\n  rust ={}\n  dav1d={}",
                FAM_NAMES[key.0 as usize],
                hex(&got),
                hex(&want_digest)
            );
        }
        matched += 1;
    }
    assert!(
        cells_small.is_empty(),
        "{bpc}-bit: hücre başına < 1000 çağrı: {cells_small:?}"
    );
    assert!(expected.is_empty(), "iki tarafta hücre kümesi eşit olmalı");
    println!(
        "  [large {bpc}-bit{}] {records} kayıt · {matched} hücre · {calls_total} çağrı · \
         SHA-256 eşleşti",
        avx2.map_or("", |a| a.ad)
    );
    (records, matched, calls_total)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// 8(a) — büyük küme; hücre başına ≥ 1 000 çağrı ve SHA-256 eşitliği.
#[test]
#[ignore = "büyük küme: .cache/mc-family (--ignored ile koşulur)"]
fn buyuk_kume_bit_exact() {
    if !large_dir().join("mc-family-large-8-input.bin").exists() {
        on_kosul::eksik(
            "mc_family_oracle::buyuk_kume_bit_exact",
            Neden::OracleVerisiYok,
            ".cache/mc-family büyük küme; üretmek için: bash scripts/oracle/run_mc_oracle.sh",
        );
        return;
    }
    let mut records = 0;
    let mut cells = 0;
    let mut calls = 0;
    for bpc in [8u8, 10, 12] {
        let (r, c, n) = compare_large(bpc, None);
        records += r;
        cells += c;
        calls += n;
    }
    println!("[large] TOPLAM {records} kayıt · {cells} hücre · {calls} çağrı · SHA-256 eşleşti");
    assert_eq!(
        cells, 1041,
        "hücre sayısı 1 041 olmalı (aile × alt tür × genişlik)"
    );
    assert!(
        calls >= 1_041_000,
        "çağrı sayısı ≥ 1 041 000 olmalı: {calls}"
    );
    // P32–P34 AVX2 ve P35 SSSE3/SSE4.1 yuvaları büyük kümenin aynı hücrelerinde.
    for (yuv, isim, neden) in [
        (avx2_yuvalari(), "avx2", Neden::avx2()),
        (ssse3_yuvalari(), "ssse3", Neden::isa()),
        (sse41_yuvalari(), "sse41", Neden::isa()),
        (avx512_yuvalari(), "avx512icl", Neden::isa()),
    ] {
        match yuv {
            Some(a) => {
                let (r, c, n) = compare_large(8, Some(&a));
                println!(
                    "[large {isim}] 8 bpc yuvaları: {r} kayıt · {c} hücre · {n} çağrı · SHA-256 eşleşti"
                );
                // Her aile en az bir hücre (SSE4.1 yalnız warp).
                assert!(c >= a.aileler8.len(), "{isim} 8 bpc hücre sayısı: {c}");
                let (r10, c10, n10) = compare_large(10, Some(&a));
                let (r12, c12, n12) = compare_large(12, Some(&a));
                println!(
                    "[large {isim}] 16 bpc (10/12 bit) yuvaları: {} kayıt · {} hücre · {} çağrı · SHA-256 eşleşti",
                    r10 + r12,
                    c10 + c12,
                    n10 + n12
                );
                assert!(
                    c10 >= a.aileler16.len() && c12 >= a.aileler16.len(),
                    "{isim} 16 bpc hücre sayısı: {c10}/{c12}"
                );
            }
            None => on_kosul::eksik(
                &format!("mc_family_oracle::buyuk_kume_bit_exact[{isim}]"),
                neden,
                "MC ISA yuvaları işlemcinin ISA düzeyini ister",
            ),
        }
    }
    let _ = N_W_MASK;
    let _: &[usize; 7] = &WIDTHS;
    let _ = scalar::put_8tap::<8>;
}
