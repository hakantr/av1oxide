//! P54 (`F5d` — `F5-IPRED`) `aarch64` NEON intra tahmin (`44 / 44` matris kaydı = `66`
//! yuva @ `8 / 10 / 12` bpc: `intra_pred[0..13]`, `cfl_ac[0..2]`, `cfl_pred[0..3]`,
//! `pal_pred`) C oracle (`ipred-small-{8,10,12}` + `ipred-large-{8,10,12}` = `1 170 288`
//! kayıt) bit-birebir eşitliği, `SentinelBuffer`, `OverReadProbe`, tekil `dst_stride` ve
//! sözleşme dışı / bozuk nibble (`0x88`) güvenli skaler devir testleri.

#![cfg_attr(
    not(target_arch = "aarch64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::ipred::{Edge, IntraPredDsp, IntraPredMode, ipred_skaler_devir_sayisi};
use av1oxide::dsp::scalar::ipred::required_range;
use av1oxide::dsp::{ARM_ISA_CANDIDATES, DspDispatch};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};
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

fn all_ipred_fields() -> Vec<&'static str> {
    let mut v = Vec::with_capacity(22);
    v.extend_from_slice(&INTRA_PRED_FIELDS);
    v.extend_from_slice(&CFL_AC_FIELDS);
    v.extend_from_slice(&CFL_PRED_FIELDS);
    v.push("pal_pred");
    v
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ipred")
}

fn large_oracle_dir() -> Option<PathBuf> {
    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".cache")
        .join("ipred-oracle");
    if local.join("ipred-large-8-input.bin").is_file()
        && local.join("ipred-large-10-input.bin").is_file()
        && local.join("ipred-large-12-input.bin").is_file()
    {
        return Some(local);
    }
    None
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Record {
    kind: u8,
    aux: u8,
    w: usize,
    h: usize,
    bw: usize,
    bh: usize,
    back: usize,
    front: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bitdepth_max: u16,
    alpha: i8,
    w_pad: usize,
    h_pad: usize,
    pal_sz: usize,
    payload_len: usize,
}

fn read_u8(b: &[u8], off: &mut usize) -> u8 {
    let v = b[*off];
    *off += 1;
    v
}

fn read_u16(b: &[u8], off: &mut usize) -> u16 {
    let v = u16::from_le_bytes([b[*off], b[*off + 1]]);
    *off += 2;
    v
}

fn read_i16(b: &[u8], off: &mut usize) -> i16 {
    let v = i16::from_le_bytes([b[*off], b[*off + 1]]);
    *off += 2;
    v
}

fn read_u32(b: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes([b[*off], b[*off + 1], b[*off + 2], b[*off + 3]]);
    *off += 4;
    v
}

fn read_record(b: &[u8], off: &mut usize) -> Record {
    let kind = read_u8(b, off);
    let aux = read_u8(b, off);
    assert_eq!(read_u8(b, off), 0);
    assert_eq!(read_u8(b, off), 0);
    let w = usize::from(read_u16(b, off));
    let h = usize::from(read_u16(b, off));
    let bw = usize::from(read_u16(b, off));
    let bh = usize::from(read_u16(b, off));
    let back = usize::from(read_u16(b, off));
    let front = usize::from(read_u16(b, off));
    let angle = read_u32(b, off) as usize;
    let max_width = read_u32(b, off) as usize;
    let max_height = read_u32(b, off) as usize;
    let bitdepth_max = read_u16(b, off);
    let alpha = read_u8(b, off) as i8;
    let w_pad = usize::from(read_u8(b, off));
    let h_pad = usize::from(read_u8(b, off));
    let pal_sz = usize::from(read_u8(b, off));
    assert_eq!(read_u16(b, off), 0);
    let payload_len = read_u32(b, off) as usize;
    Record {
        kind,
        aux,
        w,
        h,
        bw,
        bh,
        back,
        front,
        angle,
        max_width,
        max_height,
        bitdepth_max,
        alpha,
        w_pad,
        h_pad,
        pal_sz,
        payload_len,
    }
}

fn verify_ipred_oracle_pair(
    ipred: &IntraPredDsp,
    bd_expected: u8,
    input_path: &Path,
    expect_path: &Path,
    set_label: &str,
) -> usize {
    let inp = std::fs::read(input_path).unwrap_or_else(|e| panic!("{}: {e}", input_path.display()));
    let exp =
        std::fs::read(expect_path).unwrap_or_else(|e| panic!("{}: {e}", expect_path.display()));

    assert!(inp.len() >= 16 && &inp[0..4] == b"IPRI");
    assert!(exp.len() >= 16 && &exp[0..4] == b"IPRE");
    let mut ioff = 4usize;
    let mut eoff = 4usize;
    assert_eq!(read_u32(&inp, &mut ioff), 1);
    assert_eq!(read_u32(&exp, &mut eoff), 1);
    assert_eq!(read_u8(&inp, &mut ioff), bd_expected);
    assert_eq!(read_u8(&exp, &mut eoff), bd_expected);
    ioff += 3;
    eoff += 3;
    let n_in = read_u32(&inp, &mut ioff) as usize;
    let n_ex = read_u32(&exp, &mut eoff) as usize;
    assert_eq!(n_in, n_ex);

    let bit_depth = BitDepth::from_bits(u32::from(bd_expected)).expect("bitdepth");
    let bpp = if bit_depth == BitDepth::Eight { 1 } else { 2 };
    let mut tested_blocks = 0usize;
    let mut edge8 = Vec::with_capacity(512);
    let mut luma = Vec::with_capacity(128 * 128 * 2);
    let mut ac = Vec::with_capacity(64 * 64);
    let mut dst = Vec::with_capacity(128 * 128 * 2);

    for case_idx in 0..n_in {
        let rec = read_record(&inp, &mut ioff);
        let exp_rec = read_record(&exp, &mut eoff);
        assert_eq!(rec, exp_rec);

        match rec.kind {
            0..=13 => {
                let count = rec.back + rec.front + 1;
                assert_eq!(rec.payload_len, count * 2);
                let mode = IntraPredMode::from_index(usize::from(rec.kind)).unwrap();
                let stride = (rec.w + 8) * bpp;
                let dst_len = (rec.h - 1) * stride + rec.w * bpp;
                dst.clear();
                dst.resize(dst_len, 0xA5);

                if bpp == 1 {
                    edge8.clear();
                    for _ in 0..count {
                        edge8.push(read_u16(&inp, &mut ioff) as u8);
                    }
                    ipred.intra_pred[mode.index()](
                        &mut dst,
                        stride,
                        Edge::new(&edge8, rec.back),
                        rec.w,
                        rec.h,
                        rec.angle,
                        rec.max_width,
                        rec.max_height,
                        bit_depth,
                        rec.bitdepth_max,
                    );
                    for y in 0..rec.h {
                        for x in 0..rec.w {
                            let want = read_u16(&exp, &mut eoff) as u8;
                            let got = dst[y * stride + x];
                            assert_eq!(
                                got, want,
                                "{set_label} bd=8 #{case_idx} {:?} {}x{} angle={} ({x},{y})",
                                mode, rec.w, rec.h, rec.angle
                            );
                        }
                    }
                } else {
                    let edge_slice = &inp[ioff..ioff + rec.payload_len];
                    ioff += rec.payload_len;
                    ipred.intra_pred[mode.index()](
                        &mut dst,
                        stride,
                        Edge::new(edge_slice, rec.back * 2),
                        rec.w,
                        rec.h,
                        rec.angle,
                        rec.max_width,
                        rec.max_height,
                        bit_depth,
                        rec.bitdepth_max,
                    );
                    for y in 0..rec.h {
                        let row_got = &dst[y * stride..y * stride + rec.w * 2];
                        let row_want = &exp[eoff..eoff + rec.w * 2];
                        eoff += rec.w * 2;
                        assert_eq!(
                            row_got, row_want,
                            "{set_label} bd={bd_expected} #{case_idx} {:?} {}x{} angle={} y={y}",
                            mode, rec.w, rec.h, rec.angle
                        );
                    }
                }
                tested_blocks += 1;
            }
            14..=16 => {
                let layout = usize::from(rec.kind - 14);
                let ss_hor = usize::from(layout != 2);
                let ss_ver = usize::from(layout == 0);
                let luma_w = rec.w << ss_hor;
                let luma_h = rec.h << ss_ver;
                assert_eq!(rec.payload_len, luma_w * luma_h * 2);
                let luma_stride = (luma_w + 4) * bpp;
                luma.clear();
                luma.resize(luma_stride * luma_h, 0);

                if bpp == 1 {
                    for y in 0..luma_h {
                        for x in 0..luma_w {
                            luma[y * luma_stride + x] = read_u16(&inp, &mut ioff) as u8;
                        }
                    }
                } else {
                    for y in 0..luma_h {
                        luma[y * luma_stride..y * luma_stride + luma_w * 2]
                            .copy_from_slice(&inp[ioff..ioff + luma_w * 2]);
                        ioff += luma_w * 2;
                    }
                }

                ac.clear();
                ac.resize(rec.w * rec.h, 0);
                ipred.cfl_ac[layout](
                    &mut ac,
                    &luma,
                    luma_stride,
                    rec.w_pad,
                    rec.h_pad,
                    rec.w,
                    rec.h,
                    bit_depth,
                );
                for (k, &got) in ac.iter().enumerate() {
                    let want = read_i16(&exp, &mut eoff);
                    assert_eq!(
                        got, want,
                        "{set_label} bd={bd_expected} #{case_idx} cfl_ac[{layout}] {}x{} pad=({},{}) k={k}",
                        rec.w, rec.h, rec.w_pad, rec.h_pad
                    );
                }
                tested_blocks += 1;
            }
            17..=20 => {
                let cfl_idx = usize::from(rec.kind - 17);
                let count = rec.back + rec.front + 1;
                assert_eq!(rec.payload_len, count * 2 + rec.w * rec.h * 2);
                let stride = (rec.w + 8) * bpp;
                let dst_len = (rec.h - 1) * stride + rec.w * bpp;
                dst.clear();
                dst.resize(dst_len, 0xA5);

                if bpp == 1 {
                    edge8.clear();
                    for _ in 0..count {
                        edge8.push(read_u16(&inp, &mut ioff) as u8);
                    }
                    ac.clear();
                    for _ in 0..rec.w * rec.h {
                        ac.push(read_i16(&inp, &mut ioff));
                    }
                    ipred.cfl_pred[cfl_idx](
                        &mut dst,
                        stride,
                        Edge::new(&edge8, rec.back),
                        rec.w,
                        rec.h,
                        &ac,
                        i32::from(rec.alpha),
                        bit_depth,
                        rec.bitdepth_max,
                    );
                    for y in 0..rec.h {
                        for x in 0..rec.w {
                            let want = read_u16(&exp, &mut eoff) as u8;
                            let got = dst[y * stride + x];
                            assert_eq!(
                                got, want,
                                "{set_label} bd=8 #{case_idx} cfl_pred[{cfl_idx}] {}x{} alpha={} ({x},{y})",
                                rec.w, rec.h, rec.alpha
                            );
                        }
                    }
                } else {
                    let edge_bytes = count * 2;
                    let edge_slice = &inp[ioff..ioff + edge_bytes];
                    ioff += edge_bytes;
                    ac.clear();
                    for _ in 0..rec.w * rec.h {
                        ac.push(read_i16(&inp, &mut ioff));
                    }
                    ipred.cfl_pred[cfl_idx](
                        &mut dst,
                        stride,
                        Edge::new(edge_slice, rec.back * 2),
                        rec.w,
                        rec.h,
                        &ac,
                        i32::from(rec.alpha),
                        bit_depth,
                        rec.bitdepth_max,
                    );
                    for y in 0..rec.h {
                        let row_got = &dst[y * stride..y * stride + rec.w * 2];
                        let row_want = &exp[eoff..eoff + rec.w * 2];
                        eoff += rec.w * 2;
                        assert_eq!(
                            row_got, row_want,
                            "{set_label} bd={bd_expected} #{case_idx} cfl_pred[{cfl_idx}] {}x{} alpha={} y={y}",
                            rec.w, rec.h, rec.alpha
                        );
                    }
                }
                tested_blocks += 1;
            }
            21 => {
                let idx_len = rec.w * rec.h / 2;
                assert_eq!(rec.payload_len, 16 + idx_len);
                let stride = (rec.w + 8) * bpp;
                let dst_len = (rec.h - 1) * stride + rec.w * bpp;
                dst.clear();
                dst.resize(dst_len, 0xA5);

                if bpp == 1 {
                    let mut pal8 = [0u8; 8];
                    for p in &mut pal8 {
                        *p = read_u16(&inp, &mut ioff) as u8;
                    }
                    let idx_slice = &inp[ioff..ioff + idx_len];
                    ioff += idx_len;
                    (ipred.pal_pred)(&mut dst, stride, &pal8, idx_slice, rec.w, rec.h, bit_depth);
                    for y in 0..rec.h {
                        for x in 0..rec.w {
                            let want = read_u16(&exp, &mut eoff) as u8;
                            let got = dst[y * stride + x];
                            assert_eq!(
                                got, want,
                                "{set_label} bd=8 #{case_idx} pal_pred {}x{} ({x},{y})",
                                rec.w, rec.h
                            );
                        }
                    }
                } else {
                    let pal_slice = &inp[ioff..ioff + 16];
                    ioff += 16;
                    let idx_slice = &inp[ioff..ioff + idx_len];
                    ioff += idx_len;
                    (ipred.pal_pred)(
                        &mut dst, stride, pal_slice, idx_slice, rec.w, rec.h, bit_depth,
                    );
                    for y in 0..rec.h {
                        let row_got = &dst[y * stride..y * stride + rec.w * 2];
                        let row_want = &exp[eoff..eoff + rec.w * 2];
                        eoff += rec.w * 2;
                        assert_eq!(
                            row_got, row_want,
                            "{set_label} bd={bd_expected} #{case_idx} pal_pred {}x{} y={y}",
                            rec.w, rec.h
                        );
                    }
                }
                tested_blocks += 1;
            }
            22 => {
                ioff += rec.payload_len;
                eoff += (rec.bw / 2) * rec.bh;
            }
            _ => panic!("bilinmeyen kind={}", rec.kind),
        }
    }

    assert_eq!(ioff, inp.len());
    assert_eq!(eoff, exp.len());
    tested_blocks
}

#[test]
fn f5d_ipred_arm64_aday_ve_66_yuva_kurulumu() {
    let ipred_adaylar: Vec<_> = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "ipred")
        .collect();
    assert_eq!(ipred_adaylar.len(), 66);
    let fields = all_ipred_fields();
    for bd in [8u8, 10, 12] {
        for &field in &fields {
            assert!(
                ipred_adaylar.iter().any(|c| c.bit_depth == bd
                    && c.field == field
                    && c.implementation == "neon"
                    && c.requires == CpuFlags::NEON
                    && c.default),
                "eksik ARM64 IPRED adayı: ipred/{field}@{bd}"
            );
        }
    }

    let d_none = DspDispatch::select(CpuLevel::None.allowed());
    for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
        for &field in &fields {
            assert_eq!(d_none.slot_implementation(bd, "ipred", field), "skaler");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let d_neon = DspDispatch::select(CpuFlags::detect());
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            for &field in &fields {
                assert_eq!(d_neon.slot_implementation(bd, "ipred", field), "neon");
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5d_ipred_arm64_small_ve_large_c_oracle_sentinel_overread_ve_skaler_devir() {
    #[cfg(target_arch = "aarch64")]
    {
        let _kilit = SAYAC
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let devir_once = ipred_skaler_devir_sayisi();

        let d_neon = DspDispatch::select(CpuFlags::detect());
        let small_dir = fixture_dir();
        let Some(large_dir) = large_oracle_dir() else {
            on_kosul::eksik(
                "ipred_arm64::f5d_ipred_arm64_small_ve_large_c_oracle_sentinel_overread_ve_skaler_devir",
                Neden::OracleVerisiYok,
                "ipred-large-{8,10,12}-input.bin yok (.cache/ipred-oracle)",
            );
            return;
        };

        let mut total_blocks = 0usize;
        for (bd_num, bd_enum) in [
            (8u8, BitDepth::Eight),
            (10u8, BitDepth::Ten),
            (12u8, BitDepth::Twelve),
        ] {
            let ipred = d_neon.for_bit_depth(bd_enum).ipred();
            let s_in = small_dir.join(format!("ipred-small-{bd_num}-input.bin"));
            let s_ex = small_dir.join(format!("ipred-small-{bd_num}-expect.bin"));
            let n_s = verify_ipred_oracle_pair(ipred, bd_num, &s_in, &s_ex, "small");
            assert!(n_s >= 900, "small oracle bd={bd_num}: {n_s}");

            let l_in = large_dir.join(format!("ipred-large-{bd_num}-input.bin"));
            let l_ex = large_dir.join(format!("ipred-large-{bd_num}-expect.bin"));
            let n_l = verify_ipred_oracle_pair(ipred, bd_num, &l_in, &l_ex, "large");
            assert!(n_l >= 350_000, "large oracle bd={bd_num}: {n_l}");

            total_blocks += n_s + n_l;
        }
        assert!(
            total_blocks >= 1_100_000,
            "toplam IPRED oracle bloğu >= 1.1M olmalı: {total_blocks}"
        );

        // SentinelBuffer + OverReadProbe + tekil dst_stride (`w + 1`, `w + 3` piksel):
        for (bd_enum, bd_max) in [
            (BitDepth::Eight, 255u16),
            (BitDepth::Ten, 1023u16),
            (BitDepth::Twelve, 4095u16),
        ] {
            let bpp = if bd_enum == BitDepth::Eight { 1 } else { 2 };
            let neon = d_neon.for_bit_depth(bd_enum).ipred();
            let skaler = match bd_enum {
                BitDepth::Eight => IntraPredDsp::scalar::<8>(),
                BitDepth::Ten => IntraPredDsp::scalar::<10>(),
                BitDepth::Twelve => IntraPredDsp::scalar::<12>(),
            };
            for (w, h) in [(4usize, 4usize), (8, 8), (16, 8), (32, 16)] {
                let stride = (w + 3) * bpp;
                let dst_len = (h - 1) * stride + w * bpp;
                for mode_idx in 0..14usize {
                    let mode = IntraPredMode::from_index(mode_idx).unwrap();
                    if mode == IntraPredMode::Filter && (w > 32 || h > 32) {
                        continue;
                    }
                    let angle = match mode {
                        IntraPredMode::Z1 => 45,
                        IntraPredMode::Z2 => 135,
                        IntraPredMode::Z3 => 225,
                        IntraPredMode::Filter => 2,
                        _ => 0,
                    };
                    let (back, front) = required_range(mode, w, h, angle);
                    let edge_len = (back + front + 1) * bpp;
                    let probe = OverReadProbe::new(edge_len, 64, dst_len);
                    let mut out_a = vec![0u8; dst_len];
                    let mut out_b = vec![0u8; dst_len];
                    probe
                        .run_pair(&mut out_a, &mut out_b, |inp, out| {
                            out.fill(0x5A);
                            let mut ebuf =
                                inp[probe.data_offset()..probe.data_offset() + edge_len].to_vec();
                            if bpp == 2 {
                                for ch in ebuf.as_chunks_mut::<2>().0 {
                                    let v = u16::from_le_bytes(*ch) & bd_max;
                                    *ch = v.to_le_bytes();
                                }
                            }
                            let e_slice = if bpp == 1 {
                                &inp[probe.data_offset()..probe.data_offset() + edge_len]
                            } else {
                                &ebuf[..]
                            };
                            let mut sb = SentinelBuffer::with_defaults(dst_len);
                            sb.data_mut().copy_from_slice(out);
                            neon.intra_pred[mode_idx](
                                sb.data_mut(),
                                stride,
                                Edge::new(e_slice, back * bpp),
                                w,
                                h,
                                angle,
                                128,
                                128,
                                bd_enum,
                                bd_max,
                            );
                            sb.verify().expect("IPRED SentinelBuffer ihlali");
                            out.copy_from_slice(sb.data());

                            let mut ref_out = vec![0x5Au8; dst_len];
                            skaler.intra_pred[mode_idx](
                                &mut ref_out,
                                stride,
                                Edge::new(e_slice, back * bpp),
                                w,
                                h,
                                angle,
                                128,
                                128,
                                bd_enum,
                                bd_max,
                            );
                            assert_eq!(out, &ref_out[..], "IPRED tekil stride farkı {mode:?}");
                        })
                        .expect("IPRED OverReadProbe taşması");
                }
            }
        }

        // Normal çalışmada skaler devir 0 olmalı:
        assert_eq!(ipred_skaler_devir_sayisi(), devir_once);

        // Bozuk palet nibble (`0x88` -> indeks >= 8) ve sözleşme dışı kısa `dst` güvenli skaler devir:
        let neon8 = d_neon.for_bit_depth(BitDepth::Eight).ipred();
        let pal8 = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let bozuk_idx = [0x88u8; 8]; // nibble 8 >= 8 -> skaler devir!
        let mut dst_pal = [0u8; 16];
        (neon8.pal_pred)(&mut dst_pal, 4, &pal8, &bozuk_idx, 4, 4, BitDepth::Eight);
        assert!(
            ipred_skaler_devir_sayisi() > devir_once,
            "bozuk pal_pred nibble 0x88 skalere devretmeli"
        );
    }
}
