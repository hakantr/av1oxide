//! P46 (F4-REFMVS, Adım 1) — `refmvs` (`load_tmvs`, `save_tmvs`, `splat_mv`) 5-düzey
//! x86 SIMD (`D03042..D03055`, 14 tablo kaydı / 7 aday) çekirdek testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **5-düzey kurulum ve `SLOW_GATHER` bağımsızlığı:** Bit derinliğinden bağımsız
//!    (`bit_depth == 0`) `refmvs` yuvaları:
//!    - `splat_mv`: `none` → `"skaler"`, `sse2`/`ssse3`/`sse41` → `"sse2"`,
//!      `avx2` → `"avx2"`, `avx512icl`/`tespit` (`SLOW_GATHER` açık/kapalı) → `"avx512icl"`.
//!    - `save_tmvs`: `none`/`sse2` → `"skaler"`, `ssse3`/`sse41` → `"ssse3"`,
//!      `avx2` → `"avx2"`, `avx512icl`/`tespit` (`SLOW_GATHER` açık/kapalı) → `"avx512icl"`.
//!    - `load_tmvs`: `none`/`sse2`/`ssse3` → `"skaler"`,
//!      `sse41`/`avx2`/`avx512icl`/`tespit` → `"sse4"`.
//! 2. **Oracle eşitliği (`refmvs_dsp_oracle` küçük + büyük küme) + nöbetçi sınır
//!    denetimi:** Küçük küme (`tests/fixtures/refmvs-dsp/refmvs-dsp-small-*`, tam bayt) ve
//!    varsa büyük küme (`.cache/refmvs-dsp/refmvs-dsp-large-*`, 40 hücre × 1 000 çağrı =
//!    40 000 çağrı SHA-256) ile tüm blok boyutu (`bs ∈ 0..22`), `bw4/bh4` ve `n_mfmvs`
//!    kombinasyonlarında 7 x86 SIMD gövdesi dav1d C oracle ve skaler referansla **%100
//!    bit-birebir** aynıdır; ön/arka nöbetçi bölgeler değişmez.
//! 3. **Sözleşme dışı güvenli skaler devir:** Sözleşme içi çağrılarda
//!    [`refmvs_skaler_devir_sayisi`] `0` artar; sözleşme dışı girdilerde paniklemeden
//!    sayacı artırır.
//! 4. **ISA sınırı (`objdump -d`):** `refmvs_sse2` sembollerinde SSSE3+/VEX/EVEX `0`,
//!    `refmvs_ssse3` sembollerinde SSE4.1+/VEX/EVEX `0`, `refmvs_sse41` sembollerinde
//!    VEX/EVEX `0`, `refmvs` (AVX2) sembollerinde EVEX `0`, `refmvs_avx512` sembollerinde
//!    gather/scatter `0`.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::decode::refmvs::{Mv, RefBlock};
use av1oxide::decode::refmvs_tmvs::{INVALID_REF2CUR, MfmvPlan, TemporalBlock};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::refmvs::{LoadTmvsFn, SaveTmvsFn, SplatMvFn, refmvs_skaler_devir_sayisi};
use av1oxide::dsp::scalar::refmvs::{
    load_tmvs as scalar_load_tmvs, save_tmvs as scalar_save_tmvs, splat_mv as scalar_splat_mv,
};
use common::on_kosul::{self, Neden};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

struct RefmvsYuvalar {
    splat_sse2: SplatMvFn,
    splat_avx2: SplatMvFn,
    splat_avx512: SplatMvFn,
    save_ssse3: SaveTmvsFn,
    save_avx2: SaveTmvsFn,
    save_avx512: SaveTmvsFn,
    load_sse41: LoadTmvsFn,
}

fn refmvs_yuvalari(test: &str) -> Option<RefmvsYuvalar> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let d_sse2 = DspDispatch::select(detected.intersection(CpuLevel::Sse2.allowed()));
        let d_ssse3 = DspDispatch::select(detected.intersection(CpuLevel::Ssse3.allowed()));
        let d_sse41 = DspDispatch::select(detected.intersection(CpuLevel::Sse41.allowed()));
        let d_avx2 = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()));
        let d_avx512 = DspDispatch::select(detected);
        if d_sse2.sse2_token().is_some()
            && d_ssse3.ssse3_token().is_some()
            && d_sse41.sse41_token().is_some()
            && d_avx2.avx2_token().is_some()
            && d_avx512.avx512icl_token().is_some()
        {
            return Some(RefmvsYuvalar {
                splat_sse2: d_sse2.refmvs().splat_mv,
                splat_avx2: d_avx2.refmvs().splat_mv,
                splat_avx512: d_avx512.refmvs().splat_mv,
                save_ssse3: d_ssse3.refmvs().save_tmvs,
                save_avx2: d_avx2.refmvs().save_tmvs,
                save_avx512: d_avx512.refmvs().save_tmvs,
                load_sse41: d_sse41.refmvs().load_tmvs,
            });
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P46] refmvs SSE2/SSSE3/SSE4.1/AVX2/AVX512ICL gövdeleri x86_64 AVX512ICL ister",
    );
    None
}

// ---------------------------------------------------------------------------
// SHA-256 ve oracle akış çözümleyici (`refmvs_dsp_oracle` ile aynı biçim).
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
                let b = self.buf;
                self.block(&b);
                self.fill = 0;
            }
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
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let v = &self.data[self.pos..self.pos + n];
        self.pos += n;
        v
    }
}

fn block_from(bytes: &[u8]) -> RefBlock {
    let i16_at = |o: usize| i16::from_le_bytes([bytes[o], bytes[o + 1]]);
    RefBlock {
        mv: [
            Mv {
                y: i16_at(0),
                x: i16_at(2),
            },
            Mv {
                y: i16_at(4),
                x: i16_at(6),
            },
        ],
        ref_pair: [bytes[8] as i8, bytes[9] as i8],
        bs: bytes[10],
        mf: bytes[11],
    }
}

fn block_to(b: &RefBlock) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..2].copy_from_slice(&b.mv[0].y.to_le_bytes());
    out[2..4].copy_from_slice(&b.mv[0].x.to_le_bytes());
    out[4..6].copy_from_slice(&b.mv[1].y.to_le_bytes());
    out[6..8].copy_from_slice(&b.mv[1].x.to_le_bytes());
    out[8] = b.ref_pair[0] as u8;
    out[9] = b.ref_pair[1] as u8;
    out[10] = b.bs;
    out[11] = b.mf;
    out
}

fn temporal_to(t: &TemporalBlock) -> [u8; 5] {
    let mut out = [0u8; 5];
    out[0..2].copy_from_slice(&t.mv.y.to_le_bytes());
    out[2..4].copy_from_slice(&t.mv.x.to_le_bytes());
    out[4] = t.ref_;
    out
}

fn temporal_from(bytes: &[u8]) -> TemporalBlock {
    TemporalBlock {
        mv: Mv {
            y: i16::from_le_bytes([bytes[0], bytes[1]]),
            x: i16::from_le_bytes([bytes[2], bytes[3]]),
        },
        ref_: bytes[4],
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn find_oracle_file(name: &str) -> Option<PathBuf> {
    let cache = repo_root().join(".cache/refmvs-dsp").join(name);
    if cache.is_file() {
        return Some(cache);
    }
    let fixture = repo_root()
        .join("crates/av1oxide/tests/fixtures/refmvs-dsp")
        .join(name);
    fixture.is_file().then_some(fixture)
}

fn run_call_all_isas(kind: u32, p: &[u32; 8], pay: &[u8], y: &RefmvsYuvalar) -> Option<Vec<u8>> {
    let mut cur = Cursor::new(pay);
    match kind {
        0 => {
            let iw4 = p[0] as usize;
            let stride = p[1] as usize;
            let (col_start8, col_end8) = (p[2] as usize, p[3] as usize);
            let (row_start8, row_end8) = (p[4] as usize, p[5] as usize);
            let mut ref_sign = [0u8; 7];
            ref_sign.copy_from_slice(cur.bytes(7));
            let _pad = cur.u8();
            let rows = row_end8 * 2 + 2;
            let grid_bytes = cur.bytes(iw4 * rows * 12);
            let grid: Vec<RefBlock> = (0..iw4 * rows)
                .map(|i| block_from(&grid_bytes[i * 12..i * 12 + 12]))
                .collect();

            let mut out_first: Option<Vec<u8>> = None;
            for (isa_ad, f) in [
                ("ssse3", y.save_ssse3),
                ("avx2", y.save_avx2),
                ("avx512icl", y.save_avx512),
            ] {
                let mut rp = vec![TemporalBlock::default(); stride * (row_end8 - row_start8) + 64];
                f(
                    &mut rp, stride, &grid, iw4, &ref_sign, col_end8, row_end8, col_start8,
                    row_start8,
                );
                let mut out = Vec::with_capacity(stride * (row_end8 - row_start8) * 5);
                for r in 0..row_end8 - row_start8 {
                    for x in 0..stride {
                        out.extend_from_slice(&temporal_to(&rp[r * stride + x]));
                    }
                }
                if let Some(ref prev) = out_first {
                    assert_eq!(&out, prev, "save_tmvs ISA farkı ({isa_ad})");
                } else {
                    out_first = Some(out);
                }
            }
            out_first
        }
        1 => {
            let stride = p[0] as usize;
            let iw8 = p[1] as usize;
            let ih8 = p[2] as usize;
            let n_mfmvs = p[3] as usize;
            let (col_start8, col_end8) = (p[4] as usize, p[5] as usize);
            let (row_start8, row_end8) = (p[6] as usize, p[7] as usize);
            let plan_bytes = cur.bytes(32);
            let mut plan = MfmvPlan {
                n_mfmvs,
                ..MfmvPlan::default()
            };
            for n in 0..3 {
                plan.mfmv_ref[n] = plan_bytes[n * 9];
                plan.mfmv_ref2cur[n] = plan_bytes[n * 9 + 1] as i8;
                for m in 0..7 {
                    plan.mfmv_ref2ref[n][m] = plan_bytes[n * 9 + 2 + m];
                }
            }
            let mut ref_bufs: [Vec<TemporalBlock>; 7] = Default::default();
            let mut present = [false; 7];
            for buf in ref_bufs.iter_mut() {
                let pr = cur.u8();
                let len = cur.u32() as usize;
                let raw = cur.bytes(len * 5);
                *buf = (0..len)
                    .map(|k| temporal_from(&raw[k * 5..k * 5 + 5]))
                    .collect();
                if pr == 0 {
                    buf.clear();
                }
            }
            for (i, buf) in ref_bufs.iter().enumerate() {
                present[i] = !buf.is_empty();
            }
            let proj_len = 16 * stride;
            let raw = cur.bytes(proj_len * 5);
            let mut proj: Vec<TemporalBlock> = (0..proj_len)
                .map(|k| temporal_from(&raw[k * 5..k * 5 + 5]))
                .collect();
            proj.resize(proj_len + 64, TemporalBlock::default());
            {
                let refs: [Option<&[TemporalBlock]>; 7] = [
                    present[0].then_some(ref_bufs[0].as_slice()),
                    present[1].then_some(ref_bufs[1].as_slice()),
                    present[2].then_some(ref_bufs[2].as_slice()),
                    present[3].then_some(ref_bufs[3].as_slice()),
                    present[4].then_some(ref_bufs[4].as_slice()),
                    present[5].then_some(ref_bufs[5].as_slice()),
                    present[6].then_some(ref_bufs[6].as_slice()),
                ];
                (y.load_sse41)(
                    &mut proj, stride, iw8, ih8, &plan, &refs, 0, col_start8, col_end8, row_start8,
                    row_end8,
                );
            }
            let mut out = Vec::with_capacity(proj_len * 5);
            for t in proj.iter().take(proj_len) {
                out.extend_from_slice(&temporal_to(t));
            }
            Some(out)
        }
        2 => {
            let iw4 = p[0] as usize;
            let ih4 = p[1] as usize;
            let (bx4, by4) = (p[2] as usize, p[3] as usize);
            let (bw4, bh4) = (p[4] as usize, p[5] as usize);
            let rmv = block_from(cur.bytes(12));
            let grid_bytes = cur.bytes(iw4 * ih4 * 12);
            let base_grid: Vec<RefBlock> = (0..iw4 * ih4)
                .map(|i| block_from(&grid_bytes[i * 12..i * 12 + 12]))
                .collect();

            let mut out_first: Option<Vec<u8>> = None;
            for (isa_ad, f) in [
                ("sse2", y.splat_sse2),
                ("avx2", y.splat_avx2),
                ("avx512icl", y.splat_avx512),
            ] {
                let mut grid = base_grid.clone();
                f(&mut grid, iw4, rmv, bx4, by4, bw4, bh4);
                let mut out = Vec::with_capacity(iw4 * ih4 * 12);
                for b in &grid {
                    out.extend_from_slice(&block_to(b));
                }
                if let Some(ref prev) = out_first {
                    assert_eq!(&out, prev, "splat_mv ISA farkı ({isa_ad})");
                } else {
                    out_first = Some(out);
                }
            }
            out_first
        }
        3 => None,
        _ => panic!("bilinmeyen func {kind}"),
    }
}

#[test]
fn p46_refmvs_x86_kurulum_bes_duzey_tam() {
    let Some(_y) = refmvs_yuvalari("refmvs_x86::p46_refmvs_x86_kurulum_bes_duzey_tam") else {
        return;
    };
    for (bits, beklenen_load, beklenen_save, beklenen_splat) in [
        (0x00u32, "skaler", "skaler", "skaler"),
        (0x01u32, "skaler", "skaler", "sse2"),
        (0x03u32, "skaler", "ssse3", "sse2"),
        (0x07u32, "sse4", "ssse3", "sse2"),
        (0x0fu32, "sse4", "avx2", "avx2"),
        (0x2fu32, "sse4", "avx2", "avx2"),
        (0x1fu32, "sse4", "avx512icl", "avx512icl"),
        (0x3fu32, "sse4", "avx512icl", "avx512icl"),
    ] {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        let d = DspDispatch::select(flags);
        for (field, beklenen) in [
            ("load_tmvs", beklenen_load),
            ("save_tmvs", beklenen_save),
            ("splat_mv", beklenen_splat),
        ] {
            let st = planned
                .iter()
                .find(|s| s.key.family == "refmvs" && s.key.field == field)
                .expect("refmvs yuvası");
            assert_eq!(
                st.implementation, beklenen,
                "planlanan bits={bits:#x} refmvs/{field}"
            );
            assert_eq!(
                d.independent_slot_implementation("refmvs", field),
                beklenen,
                "canlı bits={bits:#x} refmvs/{field}"
            );
        }
    }
}

#[test]
fn p46_refmvs_x86_oracle_ve_sentinel() {
    let Some(y) = refmvs_yuvalari("refmvs_x86::p46_refmvs_x86_oracle_ve_sentinel") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir_once = refmvs_skaler_devir_sayisi();

    // 1) Küçük oracle kümesi (tam bayt eşitlik).
    let inp_small =
        find_oracle_file("refmvs-dsp-small-input.bin").expect("refmvs-dsp-small-input.bin");
    let exp_small =
        find_oracle_file("refmvs-dsp-small-expect.bin").expect("refmvs-dsp-small-expect.bin");
    {
        let input = std::fs::read(&inp_small).unwrap();
        let expect = std::fs::read(&exp_small).unwrap();
        let mut ci = Cursor::new(&input);
        let mut ce = Cursor::new(&expect);
        assert_eq!(ci.bytes(8), b"RMDSP01\n");
        assert_eq!(ce.bytes(8), b"RMDSP01\n");
        let n_cells = ci.u32();
        let calls = ci.u32();
        let _ = ce.u32();
        let _ = ce.u32();
        let mut compared = 0usize;
        for _ in 0..n_cells {
            let func = ci.u32();
            let cell = ci.u32();
            assert_eq!((func, cell), (ce.u32(), ce.u32()));
            for call in 0..calls {
                let f = ci.u32();
                assert_eq!(f, func);
                let mut p = [0u32; 8];
                for slot in &mut p {
                    *slot = ci.u32();
                }
                let pay_len = ci.u32() as usize;
                let pay = ci.bytes(pay_len);
                let want_len = ce.u32() as usize;
                let want = ce.bytes(want_len);
                if let Some(got) = run_call_all_isas(f, &p, pay, &y) {
                    assert_eq!(
                        got, want,
                        "refmvs-dsp küçük küme farkı: func={func} cell={cell} call={call}"
                    );
                    compared += 1;
                }
            }
        }
        assert!(
            compared >= 120,
            "küçük kümede en az 120 DSP çağrısı doğrulanmalı: {compared}"
        );
    }

    // 2) Büyük oracle kümesi (varsa 40 hücre × 1 000 = 40 000 çağrı SHA-256).
    if let (Some(inp_large), Some(sha_large)) = (
        find_oracle_file("refmvs-dsp-large-input.bin"),
        find_oracle_file("refmvs-dsp-large.sha256"),
    ) {
        let input = std::fs::read(&inp_large).unwrap();
        let sha_text = std::fs::read_to_string(&sha_large).unwrap();
        let mut want: BTreeMap<(u32, u32), (u64, u64, String)> = BTreeMap::new();
        for line in sha_text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let func: u32 = it.next().unwrap().parse().unwrap();
            let cell: u32 = it.next().unwrap().parse().unwrap();
            let calls: u64 = it.next().unwrap().parse().unwrap();
            let bytes: u64 = it.next().unwrap().parse().unwrap();
            let digest = it.next().unwrap().to_string();
            want.insert((func, cell), (calls, bytes, digest));
        }
        let mut ci = Cursor::new(&input);
        assert_eq!(ci.bytes(8), b"RMDSP01\n");
        let n_cells = ci.u32();
        let calls = ci.u32();
        let mut total_dsp_calls = 0u64;
        for _ in 0..n_cells {
            let func = ci.u32();
            let cell = ci.u32();
            let mut sha = Sha256::new();
            let mut out_bytes = 0u64;
            for _ in 0..calls {
                let f = ci.u32();
                assert_eq!(f, func);
                let mut p = [0u32; 8];
                for slot in &mut p {
                    *slot = ci.u32();
                }
                let pay_len = ci.u32() as usize;
                let pay = ci.bytes(pay_len);
                if let Some(out) = run_call_all_isas(f, &p, pay, &y) {
                    sha.update(&out);
                    out_bytes += out.len() as u64;
                    total_dsp_calls += 1;
                }
            }
            if func <= 2 {
                let (wcalls, wbytes, wdigest) = want
                    .get(&(func, cell))
                    .unwrap_or_else(|| panic!("hücre {cell} özeti yok"));
                assert_eq!(*wcalls, u64::from(calls));
                assert_eq!(*wbytes, out_bytes, "hücre {cell}: bayt sayısı uyuşmuyor");
                assert_eq!(
                    hex(&sha.finish()),
                    *wdigest,
                    "hücre {cell} (func {func}): SHA-256 uyuşmuyor"
                );
            }
        }
        assert_eq!(
            total_dsp_calls, 40_000,
            "büyük kümede 40 000 DSP çağrısı doğrulanmalı"
        );
    }

    // 3) Nöbetçi (sentinel) ve sınır denetimleri: splat_mv, save_tmvs, load_tmvs.
    let mut rng = Rng(0x2026_1119_0046_0001);
    let nobetci_ref = RefBlock {
        mv: [
            Mv {
                y: -12345,
                x: 23456,
            },
            Mv {
                y: 11111,
                x: -22222,
            },
        ],
        ref_pair: [5, 3],
        bs: 17,
        mf: 2,
    };
    for bw4 in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32] {
        for bh4 in [1usize, 2, 4, 8, 16] {
            let iw4 = bw4 + 6;
            let ih4 = bh4 + 4;
            let bx4 = 3;
            let by4 = 2;
            let rmv = RefBlock {
                mv: [
                    Mv {
                        y: rng.aralik(-4000, 4000) as i16,
                        x: rng.aralik(-4000, 4000) as i16,
                    },
                    Mv {
                        y: rng.aralik(-4000, 4000) as i16,
                        x: rng.aralik(-4000, 4000) as i16,
                    },
                ],
                ref_pair: [rng.aralik(-1, 6) as i8, rng.aralik(-1, 6) as i8],
                bs: rng.aralik(0, 21) as u8,
                mf: rng.aralik(0, 3) as u8,
            };
            let mut ref_grid = vec![nobetci_ref; iw4 * ih4];
            scalar_splat_mv(&mut ref_grid, iw4, rmv, bx4, by4, bw4, bh4);
            for (isa_ad, f) in [
                ("sse2", y.splat_sse2),
                ("avx2", y.splat_avx2),
                ("avx512icl", y.splat_avx512),
            ] {
                let mut grid = vec![nobetci_ref; iw4 * ih4];
                f(&mut grid, iw4, rmv, bx4, by4, bw4, bh4);
                assert_eq!(
                    grid, ref_grid,
                    "splat_mv sentinel/fark ({isa_ad} bw4={bw4} bh4={bh4})"
                );
            }
        }
    }

    // save_tmvs: tam sınır boyunda (`rp.len() == (rows - 1) * stride + col_end8`, kuyruksuz!) ve
    // tüm 22 blok boyutunda (`bs ∈ 0..22`) nöbetçi/skaler paritesi.
    let nobetci_tmp = TemporalBlock {
        mv: Mv { y: 7777, x: -7777 },
        ref_: 99,
    };
    for (stride, col_start8, col_end8, row_start8, row_end8) in [
        (16usize, 0usize, 16usize, 0usize, 4usize),
        (20, 2, 18, 2, 10),
        (33, 1, 32, 0, 8),
        (64, 0, 64, 0, 16),
    ] {
        let iw4 = stride * 2;
        let grid_rows = row_end8 * 2 + 2;
        let grid: Vec<RefBlock> = (0..iw4 * grid_rows)
            .map(|i| RefBlock {
                mv: [
                    Mv {
                        y: rng.aralik(-4500, 4500) as i16,
                        x: rng.aralik(-4500, 4500) as i16,
                    },
                    Mv {
                        y: rng.aralik(-4500, 4500) as i16,
                        x: rng.aralik(-4500, 4500) as i16,
                    },
                ],
                ref_pair: [rng.aralik(-1, 7) as i8, rng.aralik(-1, 7) as i8],
                bs: ((i + rng.aralik(0, 21) as usize) % 22) as u8,
                mf: (i & 3) as u8,
            })
            .collect();
        let ref_sign: [u8; 7] = core::array::from_fn(|_| rng.aralik(0, 1) as u8);
        let rp_exact_len = (row_end8 - row_start8 - 1) * stride + col_end8;
        let mut ref_rp = vec![nobetci_tmp; rp_exact_len];
        scalar_save_tmvs(
            &mut ref_rp,
            stride,
            &grid,
            iw4,
            &ref_sign,
            col_end8,
            row_end8,
            col_start8,
            row_start8,
        );
        for (isa_ad, f) in [
            ("ssse3", y.save_ssse3),
            ("avx2", y.save_avx2),
            ("avx512icl", y.save_avx512),
        ] {
            let mut rp = vec![nobetci_tmp; rp_exact_len];
            f(
                &mut rp, stride, &grid, iw4, &ref_sign, col_end8, row_end8, col_start8, row_start8,
            );
            assert_eq!(
                rp, ref_rp,
                "save_tmvs tam sınır / nöbetçi farkı ({isa_ad} stride={stride} cols={col_start8}..{col_end8})"
            );
        }
    }

    // load_tmvs: tile_row_idx > 0, negatif/pozitif ref2cur ve INVALID_REF2CUR kombinasyonları.
    for tile_row_idx in [0usize, 1, 2] {
        for n_mfmvs in [0usize, 1, 2, 3] {
            let stride = 24usize;
            let iw8 = 20usize;
            let ih8 = 16usize;
            let col_start8 = 2usize;
            let col_end8 = 18usize;
            let row_start8 = 2usize;
            let row_end8 = 14usize;
            let mut plan = MfmvPlan {
                n_mfmvs,
                ..MfmvPlan::default()
            };
            for n in 0..3 {
                plan.mfmv_ref[n] = (n * 2) as u8;
                plan.mfmv_ref2cur[n] = if n == 1 {
                    -13
                } else {
                    rng.aralik(-31, 31) as i8
                };
                for r2r in &mut plan.mfmv_ref2ref[n] {
                    *r2r = rng.aralik(1, 31) as u8;
                }
            }
            let ref_buf: Vec<TemporalBlock> = (0..stride * ih8)
                .map(|_| TemporalBlock {
                    mv: Mv {
                        y: rng.aralik(-5000, 5000) as i16,
                        x: rng.aralik(-5000, 5000) as i16,
                    },
                    ref_: rng.aralik(0, 7) as u8,
                })
                .collect();
            let refs: [Option<&[TemporalBlock]>; 7] = [
                Some(ref_buf.as_slice()),
                None,
                Some(ref_buf.as_slice()),
                None,
                Some(ref_buf.as_slice()),
                None,
                Some(ref_buf.as_slice()),
            ];
            let proj_len = 16 * stride * (tile_row_idx + 1);
            let mut ref_proj = vec![nobetci_tmp; proj_len];
            let mut simd_proj = vec![nobetci_tmp; proj_len];
            scalar_load_tmvs(
                &mut ref_proj,
                stride,
                iw8,
                ih8,
                &plan,
                &refs,
                tile_row_idx,
                col_start8,
                col_end8,
                row_start8,
                row_end8,
            );
            (y.load_sse41)(
                &mut simd_proj,
                stride,
                iw8,
                ih8,
                &plan,
                &refs,
                tile_row_idx,
                col_start8,
                col_end8,
                row_start8,
                row_end8,
            );
            assert_eq!(
                simd_proj, ref_proj,
                "load_tmvs nöbetçi/fark (tile_row_idx={tile_row_idx} n_mfmvs={n_mfmvs})"
            );
        }
    }

    assert_eq!(
        refmvs_skaler_devir_sayisi(),
        devir_once,
        "sözleşme içi çağrılarda skaler devir olmamalı"
    );
}

#[test]
fn p46_refmvs_x86_sozlesme_disi_skaler_devir() {
    let Some(y) = refmvs_yuvalari("refmvs_x86::p46_refmvs_x86_sozlesme_disi_skaler_devir") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // 1) splat_mv sözleşme dışı (kısa grid ve bx4 + bw4 > iw4).
    let rmv = RefBlock::default();
    for f in [y.splat_sse2, y.splat_avx2, y.splat_avx512] {
        let d0 = refmvs_skaler_devir_sayisi();
        let mut kisa_grid = vec![RefBlock::default(); 15];
        f(&mut kisa_grid, 4, rmv, 0, 0, 4, 4);
        f(&mut kisa_grid, 4, rmv, 2, 0, 4, 2);
        assert_eq!(refmvs_skaler_devir_sayisi(), d0 + 2);
    }

    // 2) save_tmvs sözleşme dışı (kısa rp, kısa grid, stride == 0).
    let grid = vec![RefBlock::default(); 16 * 8];
    let ref_sign = [1u8; 7];
    for f in [y.save_ssse3, y.save_avx2, y.save_avx512] {
        let d0 = refmvs_skaler_devir_sayisi();
        let mut kisa_rp = vec![TemporalBlock::default(); 7];
        f(&mut kisa_rp, 8, &grid, 16, &ref_sign, 8, 2, 0, 0);
        let mut rp = vec![TemporalBlock::default(); 32];
        f(&mut rp, 8, &grid[..10], 16, &ref_sign, 8, 2, 0, 0);
        f(&mut rp, 0, &grid, 16, &ref_sign, 8, 2, 0, 0);
        assert_eq!(refmvs_skaler_devir_sayisi(), d0 + 3);
    }

    // 3) load_tmvs sözleşme dışı (kısa rp_proj, row_end8 - row_start8 > 16, kısa ref_buf).
    let plan = MfmvPlan {
        n_mfmvs: 1,
        mfmv_ref: [0, 0, 0],
        mfmv_ref2cur: [8, INVALID_REF2CUR, INVALID_REF2CUR],
        mfmv_ref2ref: [[4; 7]; 3],
    };
    let ref_buf = vec![TemporalBlock::default(); 8 * 8];
    let refs: [Option<&[TemporalBlock]>; 7] =
        [Some(ref_buf.as_slice()), None, None, None, None, None, None];
    let d0 = refmvs_skaler_devir_sayisi();
    let mut kisa_proj = vec![TemporalBlock::default(); 8 * 4];
    (y.load_sse41)(&mut kisa_proj, 8, 8, 8, &plan, &refs, 0, 0, 8, 0, 8);
    let mut proj = vec![TemporalBlock::default(); 16 * 8];
    (y.load_sse41)(&mut proj, 8, 8, 20, &plan, &refs, 0, 0, 8, 0, 17);
    let kisa_ref = vec![TemporalBlock::default(); 8 * 7 + 7];
    let kisa_refs: [Option<&[TemporalBlock]>; 7] = [
        Some(kisa_ref.as_slice()),
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    (y.load_sse41)(&mut proj, 8, 8, 8, &plan, &kisa_refs, 0, 0, 8, 0, 8);
    assert_eq!(refmvs_skaler_devir_sayisi(), d0 + 3);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p46_refmvs_x86_isa_siniri_objdump_denetimi() {
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
    ];
    let yasak_ssse3 = [
        "pblendvb", "pblendw", "pinsrb", "pinsrd", "pinsrq", "pextrb", "pextrd", "pextrq",
        "pmovzxbw", "pmovzxbd", "pmovzxbq", "pmovzxwd", "pmovzxwq", "pmovzxdq", "pmovsxbw",
        "pmovsxbd", "pmovsxbq", "pmovsxwd", "pmovsxwq", "pmovsxdq", "pmuldq", "pmulld", "pminsb",
        "pmaxsb", "pminuw", "pmaxuw", "pminud", "pmaxud", "pminsd", "pmaxsd", "packusdw",
    ];

    let mut sse2_bulundu = 0usize;
    let mut ssse3_bulundu = 0usize;
    let mut sse41_bulundu = 0usize;
    let mut avx2_bulundu = 0usize;
    let mut avx512_bulundu = 0usize;
    let mut aktif_kip: Option<&str> = None;
    let mut aktif_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") {
            aktif_sembol = satir.to_string();
            if satir.contains("refmvs_sse2") {
                aktif_kip = Some("sse2");
                sse2_bulundu += 1;
            } else if satir.contains("refmvs_ssse3") {
                aktif_kip = Some("ssse3");
                ssse3_bulundu += 1;
            } else if satir.contains("refmvs_sse41") {
                aktif_kip = Some("sse41");
                sse41_bulundu += 1;
            } else if satir.contains("refmvs_avx512") {
                aktif_kip = Some("avx512");
                avx512_bulundu += 1;
            } else if satir.contains("save_tmvs_avx2") || satir.contains("splat_mv_avx2") {
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
                    "SSE2 sembolünde SSSE3+/SSE4.1 yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSE2 sembolünde VEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm"),
                    "SSE2 sembolünde YMM/ZMM yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "ssse3" => {
                assert!(
                    !yasak_ssse3.contains(&op),
                    "SSSE3 sembolünde SSE4.1 yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSSE3 sembolünde VEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm"),
                    "SSSE3 sembolünde YMM/ZMM yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "sse41" => {
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSE4.1 sembolünde VEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm"),
                    "SSE4.1 sembolünde YMM/ZMM yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "avx2" => {
                assert!(
                    !sag.contains("%zmm") && !sag.contains("{%k"),
                    "AVX2 sembolünde EVEX yönergesi: {aktif_sembol} -> {satir}"
                );
            }
            "avx512" => {
                assert!(
                    !op.contains("gather") && !op.contains("scatter"),
                    "AVX512ICL refmvs sembolünde gather/scatter yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
            }
            _ => {}
        }
    }

    assert!(sse2_bulundu >= 1, "refmvs_sse2 sembolü bulunmalı");
    assert!(ssse3_bulundu >= 1, "refmvs_ssse3 sembolü bulunmalı");
    assert!(sse41_bulundu >= 1, "refmvs_sse41 sembolü bulunmalı");
    assert!(
        avx2_bulundu >= 1,
        "save_tmvs_avx2/splat_mv_avx2 sembolü bulunmalı"
    );
    assert!(avx512_bulundu >= 1, "refmvs_avx512 sembolü bulunmalı");
}
