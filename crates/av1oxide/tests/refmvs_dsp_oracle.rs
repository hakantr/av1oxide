//! Refmvs DSP differential'ı — skaler Rust ↔ dav1d C (`save_tmvs`, `load_tmvs`,
//! `splat_mv`) bit-exact.
//!
//! Oracle, dav1d `src/refmvs.c`'yi (asm=OFF, kaynak **değiştirilmeden**) ayrı
//! bir süreçte derler (`scripts/oracle/refmvs_dsp_oracle.c`) ve beklenen
//! çıktıyı üretir. Bu test üretim crate'inin çekirdeklerini aynı girdiyle
//! koşturup karşılaştırır; üretim crate'i oracle'a **bağlanmaz** (SRC-11).
//!
//! | Küme | Konum | Karşılaştırma | Git |
//! |---|---|---|---|
//! | Küçük küme (hücre başına 3 çağrı) | `tests/fixtures/refmvs-dsp/` → `.cache/refmvs-dsp/` | **tam bayt** | ✅ |
//! | Büyük küme (hücre başına 1 000 çağrı) | `.cache/refmvs-dsp/` | **hücre SHA-256** + sayaç | ⬜ |
//!
//! 40 hücre: `save_tmvs` 12, `load_tmvs` 12, `splat_mv` 16. Büyük küme
//! **40 000 çağrı** eder (hücre başına 1 000).
//!
//! Yeniden üretim (tek komut): `bash scripts/oracle/run_refmvs_dsp_oracle.sh`

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::decode::refmvs::{Mv, RefBlock};
use av1oxide::decode::refmvs_tmvs::{INVALID_REF2CUR, MfmvPlan, TemporalBlock, mv_projection};
use av1oxide::dsp::scalar::refmvs::{load_tmvs, save_tmvs, splat_mv};
use common::on_kosul::{self, Neden};
use std::collections::BTreeMap;
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

// ---------------------------------------------------------------------------
// Akış çözümleyici — `scripts/oracle/refmvs_dsp_oracle.c` biçimi.
// ---------------------------------------------------------------------------

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

/// Tek çağrıyı koşar; çıktı baytlarını döndürür.
fn run_call(kind: u32, p: &[u32; 8], pay: &[u8]) -> Vec<u8> {
    let mut cur = Cursor::new(pay);
    match kind {
        0 => {
            // save_tmvs
            let iw4 = p[0] as usize;
            let stride = p[1] as usize;
            let (col_start8, col_end8) = (p[2] as usize, p[3] as usize);
            let (row_start8, row_end8) = (p[4] as usize, p[5] as usize);
            let mut ref_sign = [0u8; 7];
            ref_sign.copy_from_slice(cur.bytes(7));
            let _pad = cur.u8();
            // C tarafı `2*row_end8 + 2` satır üretir (son satırda `bw8`
            // taşması için pay; bkz. `scripts/oracle/refmvs_dsp_oracle.c`).
            let rows = row_end8 * 2 + 2;
            let grid_bytes = cur.bytes(iw4 * rows * 12);
            let grid: Vec<RefBlock> = (0..iw4 * rows)
                .map(|i| block_from(&grid_bytes[i * 12..i * 12 + 12]))
                .collect();
            let mut rp = vec![TemporalBlock::default(); stride * (row_end8 - row_start8) + 64];
            save_tmvs(
                &mut rp, stride, &grid, iw4, &ref_sign, col_end8, row_end8, col_start8, row_start8,
            );
            let mut out = Vec::with_capacity(stride * (row_end8 - row_start8) * 5);
            for y in 0..row_end8 - row_start8 {
                for x in 0..stride {
                    out.extend_from_slice(&temporal_to(&rp[y * stride + x]));
                }
            }
            out
        }
        1 => {
            // load_tmvs
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
                load_tmvs(
                    &mut proj, stride, iw8, ih8, &plan, &refs, 0, col_start8, col_end8, row_start8,
                    row_end8,
                );
            }
            let mut out = Vec::with_capacity(proj_len * 5);
            for t in proj.iter().take(proj_len) {
                out.extend_from_slice(&temporal_to(t));
            }
            out
        }
        2 => {
            // splat_mv
            let iw4 = p[0] as usize;
            let ih4 = p[1] as usize;
            let (bx4, by4) = (p[2] as usize, p[3] as usize);
            let (bw4, bh4) = (p[4] as usize, p[5] as usize);
            let rmv = block_from(cur.bytes(12));
            let grid_bytes = cur.bytes(iw4 * ih4 * 12);
            let mut grid: Vec<RefBlock> = (0..iw4 * ih4)
                .map(|i| block_from(&grid_bytes[i * 12..i * 12 + 12]))
                .collect();
            splat_mv(&mut grid, iw4, rmv, bx4, by4, bw4, bh4);
            let mut out = Vec::with_capacity(iw4 * ih4 * 12);
            for b in &grid {
                out.extend_from_slice(&block_to(b));
            }
            out
        }
        3 => {
            // mv_projection: p = { den, num, mv.y (u16), mv.x (u16) }
            let den = p[0] as i32;
            let num = p[1] as i32;
            let mv = Mv {
                y: p[2] as u16 as i16,
                x: p[3] as u16 as i16,
            };
            let r = mv_projection(mv, num, den);
            let mut out = Vec::with_capacity(4);
            out.extend_from_slice(&r.y.to_le_bytes());
            out.extend_from_slice(&r.x.to_le_bytes());
            out
        }
        _ => panic!("bilinmeyen func {kind}"),
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn find(name: &str) -> Option<PathBuf> {
    let cache = root().join(".cache/refmvs-dsp").join(name);
    if cache.is_file() {
        return Some(cache);
    }
    let fixture = root()
        .join("crates/av1oxide/tests/fixtures/refmvs-dsp")
        .join(name);
    fixture.is_file().then_some(fixture)
}

/// Küçük küme: çağrı çağrı **tam bayt** eşitlik.
#[test]
fn refmvs_dsp_kucuk_kume_tam_bayt() {
    let Some(inp) = find("refmvs-dsp-small-input.bin") else {
        on_kosul::eksik(
            "refmvs_dsp_oracle::refmvs_dsp_kucuk_kume_tam_bayt",
            Neden::OracleVerisiYok,
            "refmvs-dsp küçük küme yok (bash scripts/oracle/run_refmvs_dsp_oracle.sh)",
        );
        return;
    };
    let Some(exp) = find("refmvs-dsp-small-expect.bin") else {
        on_kosul::eksik(
            "refmvs_dsp_oracle::refmvs_dsp_kucuk_kume_tam_bayt",
            Neden::OracleVerisiYok,
            "refmvs-dsp küçük beklenti yok",
        );
        return;
    };
    let input = std::fs::read(&inp).unwrap();
    let expect = std::fs::read(&exp).unwrap();
    let mut ci = Cursor::new(&input);
    let mut ce = Cursor::new(&expect);
    assert_eq!(ci.bytes(8), b"RMDSP01\n");
    assert_eq!(ce.bytes(8), b"RMDSP01\n");
    let n_cells = ci.u32();
    let calls = ci.u32();
    let _ = ce.u32();
    let _ = ce.u32();
    let mut compared = 0usize;
    let mut per_kind = BTreeMap::new();
    for _ in 0..n_cells {
        let func = ci.u32();
        let cell = ci.u32();
        let efunc = ce.u32();
        let ecell = ce.u32();
        assert_eq!((func, cell), (efunc, ecell), "hücre başlığı uyuşmuyor");
        for call in 0..calls {
            let f = ci.u32();
            assert_eq!(f, func);
            let mut p = [0u32; 8];
            for slot in &mut p {
                *slot = ci.u32();
            }
            let pay_len = ci.u32() as usize;
            let pay = ci.bytes(pay_len);
            let got = run_call(f, &p, pay);
            let want_len = ce.u32() as usize;
            let want = ce.bytes(want_len);
            assert_eq!(
                got.len(),
                want.len(),
                "hücre {cell} çağrı {call}: uzunluk {} != {}",
                got.len(),
                want.len()
            );
            assert!(
                got == want,
                "hücre {cell} çağrı {call}: baytlar farklı (ilk fark {:?})\ngot {:?}\nwant {:?}",
                got.iter().zip(want.iter()).position(|(a, b)| a != b),
                &got[1240..1280.min(got.len())],
                &want[1240..1280.min(want.len())]
            );
            compared += 1;
            *per_kind.entry(func).or_insert(0usize) += 1;
        }
    }
    eprintln!("refmvs-dsp küçük küme: {compared} çağrı tam bayt eşit · tür dağılımı {per_kind:?}");
}

/// `mv_projection`: **bütün referans mesafeleri** (den 1..=31) ve uç/işaret
/// durumları C ile karşılaştırır (S2b.7 §5).
///
/// P15'te `den = 3` bozulması oracle'dan kaçmıştı. Kök neden ölçüldü:
/// `load_tmvs` çıktısında pencere içine düşen projeksiyonlarda `div_mult[den]`
/// değerindeki ±1 tamsayı ofseti değiştirmez (12 000 çağrıda 0 fark), çünkü
/// pencere içi ofset |offset| ≤ 512 iken `mv*num` ≤ 1536 olur ve tamsayı
/// yuvarlaması aynı kalır. Bu yüzden gerçek C `mv_projection` işlevi ayrı
/// hücrelerle (func 3) doğrudan sınanır.
#[test]
fn mv_projeksiyonu_butun_mesafeler() {
    let Some(inp) = find("refmvs-dsp-small-input.bin") else {
        on_kosul::eksik(
            "refmvs_dsp_oracle::mv_projeksiyonu_butun_mesafeler",
            Neden::OracleVerisiYok,
            "refmvs-dsp küçük küme yok",
        );
        return;
    };
    let Some(exp) = find("refmvs-dsp-small-expect.bin") else {
        on_kosul::eksik(
            "refmvs_dsp_oracle::mv_projeksiyonu_butun_mesafeler",
            Neden::OracleVerisiYok,
            "refmvs-dsp küçük beklenti yok",
        );
        return;
    };
    let input = std::fs::read(&inp).unwrap();
    let expect = std::fs::read(&exp).unwrap();
    let mut ci = Cursor::new(&input);
    let mut ce = Cursor::new(&expect);
    assert_eq!(ci.bytes(8), b"RMDSP01\n");
    assert_eq!(ce.bytes(8), b"RMDSP01\n");
    let n_cells = ci.u32();
    let calls = ci.u32();
    let _ = ce.u32();
    let _ = ce.u32();
    let mut dens = std::collections::BTreeSet::new();
    let mut nums = std::collections::BTreeSet::new();
    let mut mvs = std::collections::BTreeSet::new();
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
            let got = run_call(f, &p, pay);
            let want_len = ce.u32() as usize;
            let want = ce.bytes(want_len);
            if func == 3 {
                assert_eq!(got, want, "mv_projection hücre {cell} çağrı {call}");
                dens.insert(p[0]);
                nums.insert(p[1] as i32);
                mvs.insert((p[2] as u16 as i16, p[3] as u16 as i16));
                compared += 1;
            }
        }
    }
    // Kapsam: 31 mesafenin tamamı, her iki işaret, sıfır ve uç mv'ler.
    assert_eq!(dens.len(), 31, "den kapsamı: {dens:?}");
    assert_eq!(dens.iter().next(), Some(&1));
    assert_eq!(dens.iter().next_back(), Some(&31));
    assert!(dens.contains(&3), "den = 3 kapsamda olmalı");
    assert!(
        nums.contains(&-31) && nums.contains(&31) && nums.contains(&0),
        "num uçları: {nums:?}"
    );
    assert!(mvs.contains(&(0, 0)));
    assert!(mvs.contains(&(32767, -32768)), "uç mv kapsamda olmalı");
    assert!(mvs.iter().any(|(y, x)| *y > 0 && *x < 0));
    assert!(mvs.iter().any(|(y, x)| *y < 0 && *x > 0));
    eprintln!("mv_projection: {compared} çağrı · 31 mesafe · den=3 dâhil");
}

/// Büyük küme: hücre başına SHA-256 (1 000 çağrı) — C ile aynı özet.
#[test]
fn refmvs_dsp_buyuk_kume_ozet() {
    let Some(inp) = find("refmvs-dsp-large-input.bin") else {
        on_kosul::eksik(
            "refmvs_dsp_oracle::refmvs_dsp_buyuk_kume_ozet",
            Neden::OracleVerisiYok,
            "refmvs-dsp büyük küme yok",
        );
        return;
    };
    let Some(sha_path) = find("refmvs-dsp-large.sha256") else {
        on_kosul::eksik(
            "refmvs_dsp_oracle::refmvs_dsp_buyuk_kume_ozet",
            Neden::OracleVerisiYok,
            "refmvs-dsp büyük küme özeti yok",
        );
        return;
    };
    let input = std::fs::read(&inp).unwrap();
    let sha_text = std::fs::read_to_string(&sha_path).unwrap();
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
    let mut total_calls = 0u64;
    let mut per_kind: BTreeMap<u32, u64> = BTreeMap::new();
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
            let out = run_call(f, &p, pay);
            sha.update(&out);
            out_bytes += out.len() as u64;
            total_calls += 1;
            *per_kind.entry(f).or_insert(0) += 1;
        }
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
    assert!(total_calls >= 40_000, "toplam çağrı {total_calls} < 40 000");
    eprintln!(
        "refmvs-dsp büyük küme: {n_cells} hücre · {total_calls} çağrı · \
         hücre başına {calls} çağrı · tür dağılımı {per_kind:?}"
    );
}

/// `mv_projection` bağımsız kontrolü (dav1d `refmvs.c:176-192`).
#[test]
fn mv_projeksiyonu_temel_ozellikler() {
    assert_eq!(mv_projection(Mv { y: 12, x: -12 }, 0, 3), Mv::ZERO);
    assert_eq!(
        mv_projection(Mv { y: 4000, x: -4000 }, 16, 16),
        Mv { y: 4000, x: -4000 }
    );
    assert_eq!(INVALID_REF2CUR, -32);
}
