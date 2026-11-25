//! P45 (F4-IPRED, Adım 2 / Paket A) — 8 bpc intra tahmin (`intra_pred[14]`,
//! `cfl_ac[3]`, `cfl_pred[4]`, `pal_pred`; 59 kayıt = 22 SSSE3 + 22 AVX2 + 15 AVX512ICL)
//! çekirdek testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **Kurulum ve `SLOW_GATHER` bağımsızlığı:** 8 bpc 22 IPRED yuvasının
//!    `CpuFlags::from_dav1d_x86_bits` (`0x00`, `0x01`, `0x03`, `0x07`, `0x0f`,
//!    `0x2f`, `0x1f`, `0x3f`) altında beklenen ISA etiketlerini (`intra_pred[14]` +
//!    `pal_pred` için `0x1f`/`0x3f`'te `"avx512icl"`, `cfl_ac[3]` + `cfl_pred[4]`
//!    için `0x1f`/`0x3f`'te `"avx2"`, `0x0f`/`0x2f`'te tüm 22 yuva `"avx2"`,
//!    `0x03`/`0x07`'de tüm 22 yuva `"ssse3"`, `0x00`/`0x01`'de `"skaler"`) kurduğunu
//!    doğrular (`SLOW_GATHER` açık/kapalı fark etmez).
//! 2. **Oracle eşitliği (`ipred-small-8` ve `ipred-large-8`):** `kind 0..=21`
//!    (`intra_pred[14]`, `cfl_ac[3]`, `cfl_pred[4]`, `pal_pred`) kayıtlarında
//!    SSSE3, AVX2 ve AVX512ICL tabloları dav1d C oracle ile **%100 bit-birebir**
//!    aynı çıktıyı üretir.
//! 3. **`SentinelBuffer` + `OverReadProbe` + tekil `dst_stride` (`w + 1`, `w + 3`, `w + 7`):**
//!    Tüm 22 yuva ve 3 ISA için ön/arka koruyucu taşması `0`, `0x00` vs `0xFF`
//!    dolgu farkı `0` ve tekil satır adımlarında skaler referansla %100 eşitlik.
//! 4. **Sözleşme dışı güvenli skaler devir:** Normal testlerden sonra
//!    [`ipred_skaler_devir_sayisi`] artışının `0` olduğunu; bozuk/kısa
//!    `dst`/`edge`/`ac`/`pal`/`idx`, yanlış `BitDepth::Ten`, yanlış `bitdepth_max`,
//!    geçersiz `w`/`h` girdilerinde paniklemeden sayacı artırdığını doğrular.
//! 5. **ISA sınırı (`objdump -d`):** `ipred_ssse3` sembollerinde SSE4.1+/VEX/EVEX `0`,
//!    `x86::ipred::` (AVX2) sembollerinde EVEX `0`, `ipred_avx512` sembollerinde
//!    `vpgather*`/`vpscatter*` `0`.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::ipred::{
    CflPredMode, Edge, IntraPredDsp, IntraPredMode, ipred_skaler_devir_sayisi,
};
use av1oxide::dsp::scalar::ipred::required_range;
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

const INTRA_PRED_FIELDS: [&str; 14] = [
    "intra_pred[DC_PRED]",
    "intra_pred[VERT_PRED]",
    "intra_pred[HOR_PRED]",
    "intra_pred[LEFT_DC_PRED]",
    "intra_pred[TOP_DC_PRED]",
    "intra_pred[DC_128_PRED]",
    "intra_pred[Z1_PRED]",
    "intra_pred[Z2_PRED]",
    "intra_pred[Z3_PRED]",
    "intra_pred[SMOOTH_PRED]",
    "intra_pred[SMOOTH_V_PRED]",
    "intra_pred[SMOOTH_H_PRED]",
    "intra_pred[PAETH_PRED]",
    "intra_pred[FILTER_PRED]",
];

const CFL_AC_FIELDS: [&str; 3] = [
    "cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]",
    "cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]",
    "cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]",
];

const CFL_PRED_FIELDS: [&str; 4] = [
    "cfl_pred[DC_PRED]",
    "cfl_pred[LEFT_DC_PRED]",
    "cfl_pred[TOP_DC_PRED]",
    "cfl_pred[DC_128_PRED]",
];

struct Ipred8Tablolar {
    skaler: IntraPredDsp,
    ssse3: IntraPredDsp,
    avx2: IntraPredDsp,
    avx512: IntraPredDsp,
}

fn ipred8_tablolari(test: &str) -> Option<Ipred8Tablolar> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let d_ssse3 = DspDispatch::select(detected.intersection(CpuLevel::Ssse3.allowed()));
        let d_avx2 = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()));
        let d_avx512 = DspDispatch::select(detected);
        if d_ssse3.ssse3_token().is_some()
            && d_avx2.avx2_token().is_some()
            && d_avx512.avx512icl_token().is_some()
        {
            return Some(Ipred8Tablolar {
                skaler: IntraPredDsp::scalar::<8>(),
                ssse3: *d_ssse3.for_bit_depth(BitDepth::Eight).ipred(),
                avx2: *d_avx2.for_bit_depth(BitDepth::Eight).ipred(),
                avx512: *d_avx512.for_bit_depth(BitDepth::Eight).ipred(),
            });
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P45] 8 bpc IPRED SSSE3/AVX2/AVX512ICL gövdeleri x86_64 AVX512ICL ister",
    );
    None
}

fn read_u8<R: Read>(r: &mut R) -> u8 {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).expect("u8 okuma");
    b[0]
}

fn read_u16<R: Read>(r: &mut R) -> u16 {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).expect("u16 okuma");
    u16::from_le_bytes(b)
}

fn read_i16<R: Read>(r: &mut R) -> i16 {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).expect("i16 okuma");
    i16::from_le_bytes(b)
}

fn read_u32<R: Read>(r: &mut R) -> u32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).expect("u32 okuma");
    u32::from_le_bytes(b)
}

fn skip_bytes<R: Read>(r: &mut R, mut n: usize) {
    let mut buf = [0u8; 4096];
    while n > 0 {
        let chunk = n.min(buf.len());
        r.read_exact(&mut buf[..chunk]).expect("payload atlama");
        n -= chunk;
    }
}

fn find_large_oracle_dir(root: &Path) -> Option<PathBuf> {
    let local = root
        .join("..")
        .join("..")
        .join(".cache")
        .join("ipred-oracle");
    if local.join("ipred-large-8-input.bin").exists() {
        return Some(local);
    }
    let shared = PathBuf::from("/home/hakanbiris/github/av1oxide/.cache/ipred-oracle");
    if shared.join("ipred-large-8-input.bin").exists() {
        return Some(shared);
    }
    None
}

fn verify_ipred_oracle_8bpc(dir: &Path, set_name: &str, t: &Ipred8Tablolar) -> usize {
    let in_path = dir.join(format!("ipred-{set_name}-8-input.bin"));
    let exp_path = dir.join(format!("ipred-{set_name}-8-expect.bin"));
    let in_f = File::open(&in_path).unwrap_or_else(|e| panic!("{}: {e}", in_path.display()));
    let exp_f = File::open(&exp_path).unwrap_or_else(|e| panic!("{}: {e}", exp_path.display()));
    let mut inp = BufReader::new(in_f);
    let mut exp = BufReader::new(exp_f);

    let mut magic = [0u8; 4];
    inp.read_exact(&mut magic).expect("IPRI magic");
    assert_eq!(&magic, b"IPRI");
    assert_eq!(read_u32(&mut inp), 1);
    assert_eq!(read_u8(&mut inp), 8);
    let mut pad = [0u8; 3];
    inp.read_exact(&mut pad).expect("pad");
    let count = read_u32(&mut inp);

    exp.read_exact(&mut magic).expect("IPRE magic");
    assert_eq!(&magic, b"IPRE");
    assert_eq!(read_u32(&mut exp), 1);
    assert_eq!(read_u8(&mut exp), 8);
    exp.read_exact(&mut pad).expect("exp pad");
    assert_eq!(read_u32(&mut exp), count);

    let mut tested_records = 0usize;
    let bd = BitDepth::Eight;

    for rec_idx in 0..count {
        let mut hdr_in = [0u8; 40];
        let mut hdr_ex = [0u8; 40];
        inp.read_exact(&mut hdr_in).expect("kayıt başlığı in");
        exp.read_exact(&mut hdr_ex).expect("kayıt başlığı ex");
        assert_eq!(hdr_in, hdr_ex, "kayıt başlığı uyuşmazlığı rec={rec_idx}");

        let kind = hdr_in[0];
        let w = usize::from(u16::from_le_bytes([hdr_in[4], hdr_in[5]]));
        let h = usize::from(u16::from_le_bytes([hdr_in[6], hdr_in[7]]));
        let bw = usize::from(u16::from_le_bytes([hdr_in[8], hdr_in[9]]));
        let bh = usize::from(u16::from_le_bytes([hdr_in[10], hdr_in[11]]));
        let back = usize::from(u16::from_le_bytes([hdr_in[12], hdr_in[13]]));
        let front = usize::from(u16::from_le_bytes([hdr_in[14], hdr_in[15]]));
        let angle = u32::from_le_bytes([hdr_in[16], hdr_in[17], hdr_in[18], hdr_in[19]]) as usize;
        let max_width =
            u32::from_le_bytes([hdr_in[20], hdr_in[21], hdr_in[22], hdr_in[23]]) as usize;
        let max_height =
            u32::from_le_bytes([hdr_in[24], hdr_in[25], hdr_in[26], hdr_in[27]]) as usize;
        let bitdepth_max = u16::from_le_bytes([hdr_in[28], hdr_in[29]]);
        let alpha = i32::from(hdr_in[30] as i8);
        let w_pad = usize::from(hdr_in[31]);
        let h_pad = usize::from(hdr_in[32]);
        let payload_len =
            u32::from_le_bytes([hdr_in[36], hdr_in[37], hdr_in[38], hdr_in[39]]) as usize;

        match kind {
            0..=13 => {
                let mode_idx = usize::from(kind);
                let count_edge = back + front + 1;
                assert_eq!(payload_len, count_edge * 2);
                let mut edge_buf = vec![0u8; count_edge];
                for slot in &mut edge_buf {
                    *slot = read_u16(&mut inp) as u8;
                }
                let mut want = vec![0u8; w * h];
                for slot in &mut want {
                    *slot = read_u16(&mut exp) as u8;
                }
                let stride = w + 8;
                for (isa_ad, dsp) in [
                    ("ssse3", &t.ssse3),
                    ("avx2", &t.avx2),
                    ("avx512icl", &t.avx512),
                ] {
                    let mut dst = vec![0xA5u8; (h - 1) * stride + w];
                    (dsp.intra_pred[mode_idx])(
                        &mut dst,
                        stride,
                        Edge::new(&edge_buf, back),
                        w,
                        h,
                        angle,
                        max_width,
                        max_height,
                        bd,
                        bitdepth_max,
                    );
                    for y in 0..h {
                        assert_eq!(
                            &dst[y * stride..y * stride + w],
                            &want[y * w..(y + 1) * w],
                            "intra_pred[{mode_idx}] oracle farkı ({set_name} rec={rec_idx} {isa_ad} {w}x{h} angle={angle} y={y})"
                        );
                    }
                }
                tested_records += 1;
            }
            14..=16 => {
                let layout = usize::from(kind - 14);
                let ss_hor = usize::from(layout != 2);
                let ss_ver = usize::from(layout == 0);
                let luma_w = w << ss_hor;
                let luma_h = h << ss_ver;
                assert_eq!(payload_len, luma_w * luma_h * 2);
                let luma_stride = luma_w + 4;
                let mut luma = vec![0u8; luma_stride * luma_h];
                for y in 0..luma_h {
                    for x in 0..luma_w {
                        luma[y * luma_stride + x] = read_u16(&mut inp) as u8;
                    }
                }
                let mut want = vec![0i16; w * h];
                for slot in &mut want {
                    *slot = read_i16(&mut exp);
                }
                for (isa_ad, dsp) in [("ssse3", &t.ssse3), ("avx2", &t.avx2)] {
                    let mut ac = vec![0i16; w * h];
                    (dsp.cfl_ac[layout])(&mut ac, &luma, luma_stride, w_pad, h_pad, w, h, bd);
                    assert_eq!(
                        ac, want,
                        "cfl_ac[{layout}] oracle farkı ({set_name} rec={rec_idx} {isa_ad} {w}x{h} pad={w_pad}/{h_pad})"
                    );
                }
                tested_records += 1;
            }
            17..=20 => {
                let cfl_idx = usize::from(kind - 17);
                let count_edge = back + front + 1;
                assert_eq!(payload_len, count_edge * 2 + w * h * 2);
                let mut edge_buf = vec![0u8; count_edge];
                for slot in &mut edge_buf {
                    *slot = read_u16(&mut inp) as u8;
                }
                let mut ac = vec![0i16; w * h];
                for slot in &mut ac {
                    *slot = read_i16(&mut inp);
                }
                let mut want = vec![0u8; w * h];
                for slot in &mut want {
                    *slot = read_u16(&mut exp) as u8;
                }
                let stride = w + 8;
                for (isa_ad, dsp) in [("ssse3", &t.ssse3), ("avx2", &t.avx2)] {
                    let mut dst = vec![0xA5u8; (h - 1) * stride + w];
                    (dsp.cfl_pred[cfl_idx])(
                        &mut dst,
                        stride,
                        Edge::new(&edge_buf, back),
                        w,
                        h,
                        &ac,
                        alpha,
                        bd,
                        bitdepth_max,
                    );
                    for y in 0..h {
                        assert_eq!(
                            &dst[y * stride..y * stride + w],
                            &want[y * w..(y + 1) * w],
                            "cfl_pred[{cfl_idx}] oracle farkı ({set_name} rec={rec_idx} {isa_ad} {w}x{h} alpha={alpha} y={y})"
                        );
                    }
                }
                tested_records += 1;
            }
            21 => {
                assert_eq!(payload_len, 16 + w * h / 2);
                let mut pal = [0u8; 8];
                for slot in &mut pal {
                    *slot = read_u16(&mut inp) as u8;
                }
                let mut idx = vec![0u8; w * h / 2];
                inp.read_exact(&mut idx).expect("pal_pred idx");
                let mut want = vec![0u8; w * h];
                for slot in &mut want {
                    *slot = read_u16(&mut exp) as u8;
                }
                let stride = w + 8;
                for (isa_ad, dsp) in [
                    ("ssse3", &t.ssse3),
                    ("avx2", &t.avx2),
                    ("avx512icl", &t.avx512),
                ] {
                    let mut dst = vec![0xA5u8; (h - 1) * stride + w];
                    (dsp.pal_pred)(&mut dst, stride, &pal, &idx, w, h, bd);
                    for y in 0..h {
                        assert_eq!(
                            &dst[y * stride..y * stride + w],
                            &want[y * w..(y + 1) * w],
                            "pal_pred oracle farkı ({set_name} rec={rec_idx} {isa_ad} {w}x{h} y={y})"
                        );
                    }
                }
                tested_records += 1;
            }
            22 => {
                skip_bytes(&mut inp, payload_len);
                skip_bytes(&mut exp, (bw / 2) * bh);
            }
            _ => panic!("bilinmeyen kind: {kind}"),
        }
    }
    tested_records
}

#[test]
fn p45_ipred_8bpc_x86_kurulum_uc_duzey_tam() {
    let Some(_t) = ipred8_tablolari("ipred_8bpc_x86::p45_ipred_8bpc_x86_kurulum_uc_duzey_tam")
    else {
        return;
    };

    for (bits, beklenen_intra_pal, beklenen_cfl) in [
        (0x00u32, "skaler", "skaler"),
        (0x01u32, "skaler", "skaler"),
        (0x03u32, "ssse3", "ssse3"),
        (0x07u32, "ssse3", "ssse3"),
        (0x0fu32, "avx2", "avx2"),
        (0x2fu32, "avx2", "avx2"),
        (0x1fu32, "avx512icl", "avx2"),
        (0x3fu32, "avx512icl", "avx2"),
    ] {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        let d = DspDispatch::select(flags);

        for field in INTRA_PRED_FIELDS.iter().copied().chain(["pal_pred"]) {
            let st = planned
                .iter()
                .find(|s| {
                    s.key.family == "ipred" && s.key.field == field && s.key.bit_depth == Some(8)
                })
                .unwrap_or_else(|| panic!("ipred/{field}@8 yuvası"));
            assert_eq!(
                st.implementation, beklenen_intra_pal,
                "planlanan bits={bits:#x} ipred/{field}@8"
            );
            assert_eq!(
                d.slot_implementation(BitDepth::Eight, "ipred", field),
                beklenen_intra_pal,
                "canlı bits={bits:#x} ipred/{field}@8"
            );
        }

        for field in CFL_AC_FIELDS
            .iter()
            .copied()
            .chain(CFL_PRED_FIELDS.iter().copied())
        {
            let st = planned
                .iter()
                .find(|s| {
                    s.key.family == "ipred" && s.key.field == field && s.key.bit_depth == Some(8)
                })
                .unwrap_or_else(|| panic!("ipred/{field}@8 yuvası"));
            assert_eq!(
                st.implementation, beklenen_cfl,
                "planlanan bits={bits:#x} ipred/{field}@8"
            );
            assert_eq!(
                d.slot_implementation(BitDepth::Eight, "ipred", field),
                beklenen_cfl,
                "canlı bits={bits:#x} ipred/{field}@8"
            );
        }
    }
}

#[test]
fn p45_ipred_8bpc_x86_oracle_small_ve_large() {
    let Some(t) = ipred8_tablolari("ipred_8bpc_x86::p45_ipred_8bpc_x86_oracle_small_ve_large")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir_once = ipred_skaler_devir_sayisi();

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let small_dir = root.join("tests").join("fixtures").join("ipred");
    let small_n = verify_ipred_oracle_8bpc(&small_dir, "small", &t);
    assert!(
        small_n >= 1000,
        "ipred-small-8 içinde en az 1000 kayıt beklenir: {small_n}"
    );

    if let Some(large_dir) = find_large_oracle_dir(root) {
        let large_n = verify_ipred_oracle_8bpc(&large_dir, "large", &t);
        assert!(
            large_n >= 370_000,
            "ipred-large-8 içinde en az 370 000 kayıt beklenir: {large_n}"
        );
    }

    assert_eq!(
        ipred_skaler_devir_sayisi(),
        devir_once,
        "oracle koşusunda skaler devir olmamalı"
    );
}

#[test]
fn p45_ipred_8bpc_x86_sentinel_overread_ve_tekil_stride() {
    let Some(t) =
        ipred8_tablolari("ipred_8bpc_x86::p45_ipred_8bpc_x86_sentinel_overread_ve_tekil_stride")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir_once = ipred_skaler_devir_sayisi();
    let bd = BitDepth::Eight;

    let sizes = [
        (4usize, 4usize),
        (4, 16),
        (16, 4),
        (8, 8),
        (8, 32),
        (32, 8),
        (16, 16),
        (16, 64),
        (64, 16),
        (32, 32),
        (64, 64),
    ];

    // 1. intra_pred[14]
    for mode in IntraPredMode::ALL {
        let angles: &[(usize, usize, usize)] = match mode {
            IntraPredMode::Z1 => &[
                (3, 0, 0),
                (45 | (1 << 10), 0, 0),
                (87 | (1 << 10) | (1 << 9), 0, 0),
            ],
            IntraPredMode::Z2 => &[
                (93, 1, 65536),
                (135 | (1 << 10), 65536, 1),
                (177 | (1 << 10) | (1 << 9), 8, 8),
            ],
            IntraPredMode::Z3 => &[
                (183, 0, 0),
                (225 | (1 << 10), 0, 0),
                (267 | (1 << 10) | (1 << 9), 0, 0),
            ],
            IntraPredMode::Filter => &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            _ => &[(0, 0, 0)],
        };

        for &(w, h) in &sizes {
            if mode == IntraPredMode::Filter && (w > 32 || h > 32) {
                continue;
            }
            for &(angle, mw, mh) in angles {
                let (back, front) = required_range(mode, w, h, angle);
                let data_len = back + front + 1;
                for stride_add in [0usize, 1, 3, 7] {
                    let stride = w + stride_add;
                    let out_len = (h - 1) * stride + w;
                    let probe = OverReadProbe::new(data_len, 64, out_len);
                    let topleft = probe.data_offset() + back;

                    let mut ref_sb = SentinelBuffer::with_defaults(out_len);
                    (t.skaler.intra_pred[mode.index()])(
                        ref_sb.data_mut(),
                        stride,
                        Edge::new(&probe.payload(), back),
                        w,
                        h,
                        angle,
                        mw,
                        mh,
                        bd,
                        255,
                    );
                    ref_sb.verify().expect("skaler sentinel");

                    for (isa_ad, dsp) in [
                        ("ssse3", &t.ssse3),
                        ("avx2", &t.avx2),
                        ("avx512icl", &t.avx512),
                    ] {
                        let mut out_a = SentinelBuffer::with_defaults(out_len);
                        let mut out_b = SentinelBuffer::with_defaults(out_len);
                        let res = probe.run_pair(out_a.data_mut(), out_b.data_mut(), |inp, out| {
                            (dsp.intra_pred[mode.index()])(
                                out,
                                stride,
                                Edge::new(inp, topleft),
                                w,
                                h,
                                angle,
                                mw,
                                mh,
                                bd,
                                255,
                            );
                        });
                        assert!(
                            res.is_ok(),
                            "{isa_ad} {mode:?} {w}x{h} stride={stride} over-read: {res:?}"
                        );
                        out_a.verify().expect("out_a sentinel");
                        out_b.verify().expect("out_b sentinel");
                        assert_eq!(
                            out_a.data(),
                            ref_sb.data(),
                            "{isa_ad} {mode:?} {w}x{h} stride={stride} skaler farkı"
                        );
                    }
                }
            }
        }
    }

    // 2. cfl_ac[3]
    for layout in 0..3usize {
        let ss_hor = usize::from(layout != 2);
        let ss_ver = usize::from(layout == 0);
        let w_max = 32 >> ss_hor;
        let h_max = 32 >> ss_ver;
        for &(cw, ch) in &sizes {
            if cw > w_max || ch > h_max {
                continue;
            }
            for &(w_pad, h_pad) in &[(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
                let h_step = 2 >> ss_hor;
                let v_step = 2 >> ss_ver;
                if !w_pad.is_multiple_of(h_step)
                    || !h_pad.is_multiple_of(v_step)
                    || w_pad * 4 >= cw
                    || h_pad * 4 >= ch
                {
                    continue;
                }
                let luma_w = cw << ss_hor;
                let luma_h = ch << ss_ver;
                for stride_add in [0usize, 1, 3, 7] {
                    let y_stride = luma_w + stride_add;
                    let y_len = (luma_h - 1) * y_stride + luma_w;
                    let ac_bytes_len = cw * ch * 2;
                    let probe = OverReadProbe::new(y_len, 64, ac_bytes_len);

                    let mut ref_ac = vec![0i16; cw * ch];
                    (t.skaler.cfl_ac[layout])(
                        &mut ref_ac,
                        &probe.payload(),
                        y_stride,
                        w_pad,
                        h_pad,
                        cw,
                        ch,
                        bd,
                    );

                    for (isa_ad, dsp) in [("ssse3", &t.ssse3), ("avx2", &t.avx2)] {
                        let mut out_a = vec![0u8; ac_bytes_len];
                        let mut out_b = vec![0u8; ac_bytes_len];
                        let res = probe.run_pair(&mut out_a, &mut out_b, |inp, out| {
                            let y_slice = &inp[probe.data_offset()..probe.data_offset() + y_len];
                            let mut ac = vec![0i16; cw * ch];
                            (dsp.cfl_ac[layout])(
                                &mut ac, y_slice, y_stride, w_pad, h_pad, cw, ch, bd,
                            );
                            for (i, &v) in ac.iter().enumerate() {
                                out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
                            }
                        });
                        assert!(
                            res.is_ok(),
                            "{isa_ad} cfl_ac[{layout}] {cw}x{ch} stride={y_stride} over-read: {res:?}"
                        );
                        let mut got_ac = vec![0i16; cw * ch];
                        for i in 0..cw * ch {
                            got_ac[i] = i16::from_le_bytes([out_a[i * 2], out_a[i * 2 + 1]]);
                        }
                        assert_eq!(
                            got_ac, ref_ac,
                            "{isa_ad} cfl_ac[{layout}] {cw}x{ch} pad={w_pad}/{h_pad} stride={y_stride}"
                        );
                    }
                }
            }
        }
    }

    // 3. cfl_pred[4]
    for mode in CflPredMode::ALL {
        for &(w, h) in &sizes {
            if w > 32 || h > 32 {
                continue;
            }
            let data_len = h + w + 1;
            let mut ac = vec![0i16; w * h];
            for (i, slot) in ac.iter_mut().enumerate() {
                *slot = (((i as i32 * 73) % 4081) - 2040) as i16;
            }
            for alpha in [-16i32, -7, 0, 5, 16] {
                for stride_add in [0usize, 1, 3, 7] {
                    let stride = w + stride_add;
                    let out_len = (h - 1) * stride + w;
                    let probe = OverReadProbe::new(data_len, 64, out_len);
                    let topleft = probe.data_offset() + h;

                    let mut ref_sb = SentinelBuffer::with_defaults(out_len);
                    (t.skaler.cfl_pred[mode.index()])(
                        ref_sb.data_mut(),
                        stride,
                        Edge::new(&probe.payload(), h),
                        w,
                        h,
                        &ac,
                        alpha,
                        bd,
                        255,
                    );
                    ref_sb.verify().expect("skaler cfl_pred sentinel");

                    for (isa_ad, dsp) in [("ssse3", &t.ssse3), ("avx2", &t.avx2)] {
                        let mut out_a = SentinelBuffer::with_defaults(out_len);
                        let mut out_b = SentinelBuffer::with_defaults(out_len);
                        let res = probe.run_pair(out_a.data_mut(), out_b.data_mut(), |inp, out| {
                            (dsp.cfl_pred[mode.index()])(
                                out,
                                stride,
                                Edge::new(inp, topleft),
                                w,
                                h,
                                &ac,
                                alpha,
                                bd,
                                255,
                            );
                        });
                        assert!(
                            res.is_ok(),
                            "{isa_ad} cfl_pred[{mode:?}] {w}x{h} over-read: {res:?}"
                        );
                        out_a.verify().expect("out_a sentinel");
                        out_b.verify().expect("out_b sentinel");
                        assert_eq!(
                            out_a.data(),
                            ref_sb.data(),
                            "{isa_ad} cfl_pred[{mode:?}] {w}x{h} alpha={alpha} stride={stride}"
                        );
                    }
                }
            }
        }
    }

    // 4. pal_pred
    let pal = [17u8, 42, 89, 128, 173, 200, 231, 254];
    for &(w, h) in &sizes {
        let idx_len = (w * h) / 2;
        for stride_add in [0usize, 1, 3, 7] {
            let stride = w + stride_add;
            let out_len = (h - 1) * stride + w;
            let probe = OverReadProbe::new(idx_len, 64, out_len);
            let mut clamped_ref = probe.payload();
            for b in &mut clamped_ref {
                *b &= 0x77;
            }
            let mut ref_sb = SentinelBuffer::with_defaults(out_len);
            (t.skaler.pal_pred)(ref_sb.data_mut(), stride, &pal, &clamped_ref, w, h, bd);
            ref_sb.verify().expect("skaler pal_pred sentinel");

            for (isa_ad, dsp) in [
                ("ssse3", &t.ssse3),
                ("avx2", &t.avx2),
                ("avx512icl", &t.avx512),
            ] {
                let mut out_a = SentinelBuffer::with_defaults(out_len);
                let mut out_b = SentinelBuffer::with_defaults(out_len);
                let res = probe.run_pair(out_a.data_mut(), out_b.data_mut(), |inp, out| {
                    let mut clamped =
                        inp[probe.data_offset()..probe.data_offset() + idx_len].to_vec();
                    for b in &mut clamped {
                        *b &= 0x77;
                    }
                    (dsp.pal_pred)(out, stride, &pal, &clamped, w, h, bd);
                });
                assert!(
                    res.is_ok(),
                    "{isa_ad} pal_pred {w}x{h} stride={stride} over-read: {res:?}"
                );
                out_a.verify().expect("out_a sentinel");
                out_b.verify().expect("out_b sentinel");
                assert_eq!(
                    out_a.data(),
                    ref_sb.data(),
                    "{isa_ad} pal_pred {w}x{h} stride={stride}"
                );
            }
        }
    }

    assert_eq!(
        ipred_skaler_devir_sayisi(),
        devir_once,
        "sözleşme içi çağrılarda skaler devir olmamalı"
    );
}

#[test]
fn p45_ipred_8bpc_x86_sozlesme_disi_skaler_devir() {
    let Some(t) = ipred8_tablolari("ipred_8bpc_x86::p45_ipred_8bpc_x86_sozlesme_disi_skaler_devir")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let edge_bytes = [128u8; 256];
    let edge = Edge::new(&edge_bytes, 128);
    let mut dst = [0u8; 64 * 64];
    let mut ac = [0i16; 32 * 32];
    let luma = [64u8; 64 * 64];
    let pal = [10u8; 8];
    let idx = [0x12u8; 64 * 64 / 2];

    for dsp in [&t.ssse3, &t.avx2, &t.avx512] {
        // 1) intra_pred: kısa dst, kısa edge, yanlış BitDepth::Ten, yanlış bitdepth_max, geçersiz w
        let d0 = ipred_skaler_devir_sayisi();
        (dsp.intra_pred[0])(&mut dst[..15], 4, edge, 4, 4, 0, 0, 0, BitDepth::Eight, 255);
        (dsp.intra_pred[0])(
            &mut dst,
            4,
            Edge::new(&edge_bytes[..4], 2),
            4,
            4,
            0,
            0,
            0,
            BitDepth::Eight,
            255,
        );
        (dsp.intra_pred[0])(&mut dst, 4, edge, 4, 4, 0, 0, 0, BitDepth::Ten, 255);
        (dsp.intra_pred[0])(&mut dst, 4, edge, 4, 4, 0, 0, 0, BitDepth::Eight, 1023);
        (dsp.intra_pred[0])(&mut dst, 6, edge, 6, 4, 0, 0, 0, BitDepth::Eight, 255);
        assert_eq!(ipred_skaler_devir_sayisi(), d0 + 5);

        // 2) pal_pred: kısa dst, kısa pal, kısa idx, yanlış BitDepth::Ten
        let d1 = ipred_skaler_devir_sayisi();
        (dsp.pal_pred)(&mut dst[..15], 4, &pal, &idx, 4, 4, BitDepth::Eight);
        (dsp.pal_pred)(&mut dst, 4, &pal[..7], &idx, 4, 4, BitDepth::Eight);
        (dsp.pal_pred)(&mut dst, 4, &pal, &idx[..7], 4, 4, BitDepth::Eight);
        (dsp.pal_pred)(&mut dst, 4, &pal, &idx, 4, 4, BitDepth::Ten);
        assert_eq!(ipred_skaler_devir_sayisi(), d1 + 4);
    }

    for dsp in [&t.ssse3, &t.avx2] {
        // 3) cfl_ac: kısa ac, kısa y, yanlış BitDepth::Ten, geçersiz cw
        let d2 = ipred_skaler_devir_sayisi();
        (dsp.cfl_ac[0])(&mut ac[..15], &luma, 8, 0, 0, 4, 4, BitDepth::Eight);
        (dsp.cfl_ac[0])(&mut ac, &luma[..15], 8, 0, 0, 4, 4, BitDepth::Eight);
        (dsp.cfl_ac[0])(&mut ac, &luma, 8, 0, 0, 4, 4, BitDepth::Ten);
        (dsp.cfl_ac[0])(&mut ac, &luma, 12, 0, 0, 6, 4, BitDepth::Eight);
        assert_eq!(ipred_skaler_devir_sayisi(), d2 + 4);

        // 4) cfl_pred: kısa dst, kısa ac, geçersiz alpha, yanlış BitDepth::Ten
        let d3 = ipred_skaler_devir_sayisi();
        (dsp.cfl_pred[0])(&mut dst[..15], 4, edge, 4, 4, &ac, 4, BitDepth::Eight, 255);
        (dsp.cfl_pred[0])(&mut dst, 4, edge, 4, 4, &ac[..15], 4, BitDepth::Eight, 255);
        (dsp.cfl_pred[0])(&mut dst, 4, edge, 4, 4, &ac, 17, BitDepth::Eight, 255);
        (dsp.cfl_pred[0])(&mut dst, 4, edge, 4, 4, &ac, 4, BitDepth::Ten, 255);
        assert_eq!(ipred_skaler_devir_sayisi(), d3 + 4);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p45_ipred_8bpc_x86_isa_siniri_objdump_denetimi() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let yasak_ssse3 = [
        "pblendvb", "pblendw", "pinsrb", "pinsrd", "pinsrq", "pextrb", "pextrd", "pextrq",
        "pmovzxbw", "pmovzxbd", "pmovzxbq", "pmovzxwd", "pmovzxwq", "pmovzxdq", "pmovsxbw",
        "pmovsxbd", "pmovsxbq", "pmovsxwd", "pmovsxwq", "pmovsxdq", "pmuldq", "pmulld", "pminsb",
        "pmaxsb", "pminuw", "pmaxuw", "pminud", "pmaxud", "pminsd", "pmaxsd", "packusdw",
    ];

    let mut ssse3_bulundu = 0usize;
    let mut avx2_bulundu = 0usize;
    let mut avx512_bulundu = 0usize;
    let mut aktif_kip: Option<&str> = None;
    let mut aktif_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") && satir.contains("x86") && satir.contains("ipred") {
            aktif_sembol = satir.to_string();
            if satir.contains("ipred_ssse3") {
                aktif_kip = Some("ssse3");
                ssse3_bulundu += 1;
            } else if satir.contains("ipred_avx512") {
                aktif_kip = Some("avx512");
                avx512_bulundu += 1;
            } else if satir.contains("ipred")
                && !satir.contains("ipred16")
                && (satir.contains("_avx2") || satir.contains("x865ipred"))
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
            "avx2" => {
                assert!(
                    !sag.contains("%zmm") && !sag.contains("{%k"),
                    "AVX2 sembolünde EVEX yönergesi: {aktif_sembol} -> {satir}"
                );
            }
            "avx512" => {
                assert!(
                    !op.contains("gather") && !op.contains("scatter"),
                    "AVX512ICL ipred_avx512 sembolünde gather/scatter yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
            }
            _ => {}
        }
    }

    assert!(ssse3_bulundu >= 1, "ipred_ssse3 sembolü bulunmalı");
    assert!(avx2_bulundu >= 1, "x86::ipred (AVX2) sembolü bulunmalı");
    assert!(avx512_bulundu >= 1, "ipred_avx512 sembolü bulunmalı");
}
