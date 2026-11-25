//! P45 Adım 2: 16-bpc (`10/12` bit ortak) IPRED x86_64 SIMD (`SSSE3`, `AVX2`,
//! `AVX512ICL`) kurulum, oracle, `SentinelBuffer`, `OverReadProbe`, sözleşme dışı
//! skaler devir ve `objdump -d` ISA sınırı testleri.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use std::path::{Path, PathBuf};

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::ipred::{
    CflPredMode, Edge, IntraPredMode, N_CFL_AC_LAYOUTS, N_IMPL_INTRA_PRED_MODES,
    ipred_skaler_devir_sayisi, kenar_okuma_araligi,
};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};

const MAGIC_IN: &[u8; 4] = b"IPRI";
const MAGIC_EX: &[u8; 4] = b"IPRE";

static DEVIR_KILIDI: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

fn all_22_ipred_fields() -> Vec<&'static str> {
    let mut fields = Vec::with_capacity(22);
    fields.extend_from_slice(&INTRA_PRED_FIELDS);
    fields.extend_from_slice(&CFL_AC_FIELDS);
    fields.extend_from_slice(&CFL_PRED_FIELDS);
    fields.push("pal_pred");
    fields
}

fn is_avx512_slot(field: &str) -> bool {
    matches!(
        field,
        "intra_pred[Z1_PRED]"
            | "intra_pred[Z2_PRED]"
            | "intra_pred[Z3_PRED]"
            | "intra_pred[SMOOTH_PRED]"
            | "intra_pred[SMOOTH_V_PRED]"
            | "intra_pred[SMOOTH_H_PRED]"
            | "intra_pred[PAETH_PRED]"
            | "intra_pred[FILTER_PRED]"
            | "pal_pred"
    )
}

struct Ipred16Dispatch {
    ssse3: DspDispatch,
    avx2: DspDispatch,
    avx512: DspDispatch,
}

fn ipred16_tablolari(test: &str) -> Option<Ipred16Dispatch> {
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
            return Some(Ipred16Dispatch {
                ssse3: d_ssse3,
                avx2: d_avx2,
                avx512: d_avx512,
            });
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P45] 16 bpc IPRED SSSE3/AVX2/AVX512ICL gövdeleri x86_64 AVX512ICL ister",
    );
    None
}

#[test]
fn ipred16_kurulum_ve_slow_gather_bagimsizligi() {
    let Some(_t) =
        ipred16_tablolari("ipred_16bpc_x86::ipred16_kurulum_ve_slow_gather_bagimsizligi")
    else {
        return;
    };
    let fields = all_22_ipred_fields();
    assert_eq!(fields.len(), 22);

    let cases: &[(u32, &str, &str)] = &[
        (0x00, "skaler", "skaler"),
        (0x01, "skaler", "skaler"),
        (0x03, "ssse3", "ssse3"),
        (0x07, "ssse3", "ssse3"),
        (0x0f, "avx2", "avx2"),
        (0x2f, "avx2", "avx2"),
        (0x1f, "avx512icl", "avx2"),
        (0x3f, "avx512icl", "avx2"),
    ];

    for &(bits, exp_icl, exp_other) in cases {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        for bd in [10u8, 12u8] {
            for &field in &fields {
                let want = if is_avx512_slot(field) {
                    exp_icl
                } else {
                    exp_other
                };
                let state = planned
                    .iter()
                    .find(|s| {
                        s.key.family == "ipred"
                            && s.key.field == field
                            && s.key.bit_depth == Some(bd)
                    })
                    .unwrap_or_else(|| panic!("ipred/{field}@{bd} yuvası bulunamadı"));
                assert_eq!(
                    state.implementation, want,
                    "bits=0x{bits:02x} ipred/{field}@{bd} planned beklenen={want}, bulunan={}",
                    state.implementation
                );
            }
        }

        let dsp = DspDispatch::select(flags);
        for (bd_enum, bd_num) in [(BitDepth::Ten, 10u8), (BitDepth::Twelve, 12u8)] {
            for &field in &fields {
                let want = if is_avx512_slot(field) {
                    exp_icl
                } else {
                    exp_other
                };
                assert_eq!(
                    dsp.slot_implementation(bd_enum, "ipred", field),
                    want,
                    "bits=0x{bits:02x} select ipred/{field}@{bd_num}"
                );
            }
        }
    }
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ipred")
}

fn cache_dir() -> PathBuf {
    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".cache")
        .join("ipred-oracle");
    if local.exists() {
        return local;
    }
    let fallback = PathBuf::from("/home/hakanbiris/github/av1oxide/.cache/ipred-oracle");
    if fallback.exists() {
        return fallback;
    }
    local
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
    let v = u16::from_le_bytes(b[*off..*off + 2].try_into().unwrap());
    *off += 2;
    v
}

fn read_i16(b: &[u8], off: &mut usize) -> i16 {
    let v = i16::from_le_bytes(b[*off..*off + 2].try_into().unwrap());
    *off += 2;
    v
}

fn read_u32(b: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*off..*off + 4].try_into().unwrap());
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

fn run_oracle_pair_with_dsp(
    dsp: &DspDispatch,
    bd_expected: u8,
    input_path: &Path,
    expect_path: &Path,
    isa_label: &str,
) -> usize {
    let inp = std::fs::read(input_path).unwrap_or_else(|e| panic!("{}: {e}", input_path.display()));
    let exp =
        std::fs::read(expect_path).unwrap_or_else(|e| panic!("{}: {e}", expect_path.display()));

    assert!(inp.len() >= 16 && &inp[0..4] == MAGIC_IN);
    assert!(exp.len() >= 16 && &exp[0..4] == MAGIC_EX);
    let mut ioff = 4;
    let mut eoff = 4;
    assert_eq!(read_u32(&inp, &mut ioff), 1);
    assert_eq!(read_u32(&exp, &mut eoff), 1);
    assert_eq!(read_u8(&inp, &mut ioff), bd_expected);
    assert_eq!(read_u8(&exp, &mut eoff), bd_expected);
    ioff += 3;
    eoff += 3;
    let n_in = read_u32(&inp, &mut ioff) as usize;
    let n_ex = read_u32(&exp, &mut eoff) as usize;
    assert_eq!(n_in, n_ex);

    let bit_depth = BitDepth::from_bits(u32::from(bd_expected)).unwrap();
    let bpp = 2usize;
    let ipred = dsp.for_bit_depth(bit_depth).ipred();
    let mut tested_blocks = 0usize;

    for case_idx in 0..n_in {
        let rec = read_record(&inp, &mut ioff);
        let exp_rec = read_record(&exp, &mut eoff);
        assert_eq!(rec, exp_rec);

        if rec.kind < N_IMPL_INTRA_PRED_MODES as u8 {
            let count = rec.back + rec.front + 1;
            assert_eq!(rec.payload_len, count * 2);
            let edge_slice = &inp[ioff..ioff + rec.payload_len];
            ioff += rec.payload_len;

            let mode = IntraPredMode::from_index(usize::from(rec.kind)).unwrap();
            let stride = (rec.w + 8) * bpp;
            let mut dst = vec![0xA5u8; (rec.h - 1) * stride + rec.w * bpp];
            ipred.intra_pred[mode.index()](
                &mut dst,
                stride,
                Edge::new(edge_slice, rec.back * bpp),
                rec.w,
                rec.h,
                rec.angle,
                rec.max_width,
                rec.max_height,
                bit_depth,
                rec.bitdepth_max,
            );
            for y in 0..rec.h {
                let row_got = &dst[y * stride..y * stride + rec.w * bpp];
                let row_want = &exp[eoff..eoff + rec.w * bpp];
                eoff += rec.w * bpp;
                assert_eq!(
                    row_got, row_want,
                    "{isa_label} bd={bd_expected} olgu #{case_idx} ({:?} {}x{} angle={} y={y})",
                    mode, rec.w, rec.h, rec.angle
                );
            }
            tested_blocks += 1;
        } else if (14..=16).contains(&rec.kind) {
            let layout = usize::from(rec.kind - 14);
            let (ss_hor, ss_ver) = match layout {
                0 => (1usize, 1usize),
                1 => (1, 0),
                2 => (0, 0),
                _ => unreachable!(),
            };
            let luma_w = rec.w << ss_hor;
            let luma_h = rec.h << ss_ver;
            assert_eq!(rec.payload_len, luma_w * luma_h * bpp);
            let luma_stride = (luma_w + 4) * bpp;
            let mut luma = vec![0u8; luma_stride * luma_h];
            for y in 0..luma_h {
                luma[y * luma_stride..y * luma_stride + luma_w * bpp]
                    .copy_from_slice(&inp[ioff..ioff + luma_w * bpp]);
                ioff += luma_w * bpp;
            }

            let mut ac = vec![0i16; rec.w * rec.h];
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
                    "{isa_label} bd={bd_expected} olgu #{case_idx} (cfl_ac[{layout}] {}x{} pad=({},{})) k={k}",
                    rec.w, rec.h, rec.w_pad, rec.h_pad
                );
            }
            tested_blocks += 1;
        } else if (17..=20).contains(&rec.kind) {
            let count = rec.back + rec.front + 1;
            assert_eq!(rec.payload_len, count * 2 + rec.w * rec.h * 2);
            let edge_bytes = count * bpp;
            let edge_slice = &inp[ioff..ioff + edge_bytes];
            ioff += edge_bytes;
            let mut ac = Vec::with_capacity(rec.w * rec.h);
            for _ in 0..rec.w * rec.h {
                ac.push(read_i16(&inp, &mut ioff));
            }
            let cfl_idx = usize::from(rec.kind - 17);
            let mode = CflPredMode::ALL[cfl_idx];
            let stride = (rec.w + 8) * bpp;
            let mut dst = vec![0xA5u8; (rec.h - 1) * stride + rec.w * bpp];
            ipred.cfl_pred[cfl_idx](
                &mut dst,
                stride,
                Edge::new(edge_slice, rec.back * bpp),
                rec.w,
                rec.h,
                &ac,
                i32::from(rec.alpha),
                bit_depth,
                rec.bitdepth_max,
            );
            for y in 0..rec.h {
                let row_got = &dst[y * stride..y * stride + rec.w * bpp];
                let row_want = &exp[eoff..eoff + rec.w * bpp];
                eoff += rec.w * bpp;
                assert_eq!(
                    row_got, row_want,
                    "{isa_label} bd={bd_expected} olgu #{case_idx} (cfl_pred[{:?}] {}x{} alpha={} y={y})",
                    mode, rec.w, rec.h, rec.alpha
                );
            }
            tested_blocks += 1;
        } else if rec.kind == 21 {
            let pal_bytes = 8 * bpp;
            let idx_len = rec.w * rec.h / 2;
            assert_eq!(rec.payload_len, pal_bytes + idx_len);
            let pal_slice = &inp[ioff..ioff + pal_bytes];
            ioff += pal_bytes;
            let idx_slice = &inp[ioff..ioff + idx_len];
            ioff += idx_len;

            let stride = (rec.w + 8) * bpp;
            let mut dst = vec![0xA5u8; (rec.h - 1) * stride + rec.w * bpp];
            (ipred.pal_pred)(
                &mut dst, stride, pal_slice, idx_slice, rec.w, rec.h, bit_depth,
            );
            for y in 0..rec.h {
                let row_got = &dst[y * stride..y * stride + rec.w * bpp];
                let row_want = &exp[eoff..eoff + rec.w * bpp];
                eoff += rec.w * bpp;
                assert_eq!(
                    row_got, row_want,
                    "{isa_label} bd={bd_expected} olgu #{case_idx} (pal_pred {}x{} y={y})",
                    rec.w, rec.h
                );
            }
            tested_blocks += 1;
        } else if rec.kind == 22 {
            ioff += rec.payload_len;
            eoff += (rec.bw / 2) * rec.bh;
        } else {
            panic!("bilinmeyen kind={} (olgu #{case_idx})", rec.kind);
        }
    }

    assert_eq!(ioff, inp.len());
    assert_eq!(eoff, exp.len());
    tested_blocks
}

#[test]
fn ipred16_small_ve_large_oracle_10_12() {
    let Some(t) = ipred16_tablolari("ipred_16bpc_x86::ipred16_small_ve_large_oracle_10_12") else {
        return;
    };
    let _guard = DEVIR_KILIDI.lock().unwrap_or_else(|e| e.into_inner());
    let before = ipred_skaler_devir_sayisi();

    let small_dir = fixture_dir();
    let large_dir = cache_dir();

    for (dsp, label) in [
        (&t.ssse3, "ssse3"),
        (&t.avx2, "avx2"),
        (&t.avx512, "avx512icl"),
    ] {
        for bd in [10u8, 12u8] {
            let s_in = small_dir.join(format!("ipred-small-{bd}-input.bin"));
            let s_ex = small_dir.join(format!("ipred-small-{bd}-expect.bin"));
            let n_small = run_oracle_pair_with_dsp(dsp, bd, &s_in, &s_ex, label);
            assert!(n_small > 100);

            let l_in = large_dir.join(format!("ipred-large-{bd}-input.bin"));
            let l_ex = large_dir.join(format!("ipred-large-{bd}-expect.bin"));
            if l_in.exists() && l_ex.exists() {
                let n_large = run_oracle_pair_with_dsp(dsp, bd, &l_in, &l_ex, label);
                assert!(n_large > 300_000);
            }
        }
    }

    assert_eq!(
        ipred_skaler_devir_sayisi(),
        before,
        "Oracle testleri sırasında skaler devir sayacı artmamalı"
    );
}

#[test]
fn ipred16_sentinel_overread_ve_buyuk_stride() {
    let Some(t) = ipred16_tablolari("ipred_16bpc_x86::ipred16_sentinel_overread_ve_buyuk_stride")
    else {
        return;
    };
    let _guard = DEVIR_KILIDI.lock().unwrap_or_else(|e| e.into_inner());
    let before = ipred_skaler_devir_sayisi();
    let scalar_dsp = DspDispatch::select(CpuFlags::from_dav1d_x86_bits(0));

    let dims: &[(usize, usize)] = &[
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (8, 16),
        (16, 8),
        (16, 16),
        (16, 32),
        (32, 16),
        (32, 32),
        (64, 64),
    ];

    for (simd_dsp, label) in [
        (&t.ssse3, "ssse3"),
        (&t.avx2, "avx2"),
        (&t.avx512, "avx512icl"),
    ] {
        for (bit_depth, bd_max) in [(BitDepth::Ten, 1023u16), (BitDepth::Twelve, 4095u16)] {
            let ipred_simd = simd_dsp.for_bit_depth(bit_depth).ipred();
            let ipred_ref = scalar_dsp.for_bit_depth(bit_depth).ipred();

            for &(w, h) in dims {
                for extra_stride in [0usize, 2, 6, 14] {
                    let stride = w * 2 + extra_stride;
                    let dst_len = (h - 1) * stride + w * 2;

                    for mode in IntraPredMode::ALL {
                        if mode == IntraPredMode::Filter && (w > 32 || h > 32) {
                            continue;
                        }
                        let angles: &[usize] = match mode {
                            IntraPredMode::Z1 => &[3, 45, 87, 3 | 1024, 45 | 1536],
                            IntraPredMode::Z2 => &[93, 135, 177, 93 | 1024, 135 | 1536],
                            IntraPredMode::Z3 => &[183, 225, 267, 183 | 1024, 225 | 1536],
                            IntraPredMode::Filter => &[0, 1, 2, 3, 4],
                            _ => &[0],
                        };

                        for &angle in angles {
                            let (back, front) = kenar_okuma_araligi(mode, w, h, angle).unwrap();
                            let data_len = (back + 1 + front) * 2;
                            let back_bytes = back * 2;
                            let probe = OverReadProbe::new(data_len, 64, dst_len);
                            let topleft = probe.data_offset() + back_bytes;
                            let mut dst_a = SentinelBuffer::with_defaults(dst_len);
                            let mut dst_b = SentinelBuffer::with_defaults(dst_len);

                            let res =
                                probe.run_pair(dst_a.data_mut(), dst_b.data_mut(), |input, out| {
                                    out.fill(0x55);
                                    let mut masked = input.to_vec();
                                    let start = probe.data_offset();
                                    for c in masked[start..start + data_len].as_chunks_mut::<2>().0
                                    {
                                        let v = u16::from_le_bytes(*c) & bd_max;
                                        *c = v.to_le_bytes();
                                    }
                                    ipred_simd.intra_pred[mode.index()](
                                        out,
                                        stride,
                                        Edge::new(&masked, topleft),
                                        w,
                                        h,
                                        angle,
                                        w,
                                        h,
                                        bit_depth,
                                        bd_max,
                                    );
                                });
                            assert!(
                                res.is_ok(),
                                "{label} {:?} {:?} {w}x{h} angle={angle} OverReadProbe: {res:?}",
                                bit_depth,
                                mode
                            );
                            dst_a.verify().expect("ön/arka koruyucu bozuldu");
                            dst_b.verify().expect("ön/arka koruyucu bozuldu");

                            let mut payload_edge = probe.payload();
                            for c in payload_edge.as_chunks_mut::<2>().0 {
                                let v = u16::from_le_bytes(*c) & bd_max;
                                *c = v.to_le_bytes();
                            }
                            let mut dst_ref = vec![0x55u8; dst_len];
                            ipred_ref.intra_pred[mode.index()](
                                &mut dst_ref,
                                stride,
                                Edge::new(&payload_edge, back_bytes),
                                w,
                                h,
                                angle,
                                w,
                                h,
                                bit_depth,
                                bd_max,
                            );
                            assert_eq!(
                                dst_a.data(),
                                &dst_ref[..],
                                "{label} {:?} {:?} {w}x{h} stride={stride} referans farkı!",
                                bit_depth,
                                mode
                            );
                        }
                    }

                    // cfl_ac + cfl_pred + pal_pred (yalnız <= 32x32 CfL, <= 64x64 pal_pred)
                    if w <= 32 && h <= 32 {
                        for layout in 0..N_CFL_AC_LAYOUTS {
                            let (ss_hor, ss_ver) = match layout {
                                0 => (1usize, 1usize),
                                1 => (1, 0),
                                _ => (0, 0),
                            };
                            let w_max = 32 >> ss_hor;
                            let h_max = 32 >> ss_ver;
                            if w > w_max || h > h_max {
                                continue;
                            }
                            let h_step = 2 >> ss_hor;
                            let v_step = 2 >> ss_ver;
                            let w_pad = if w > 4 * h_step { h_step } else { 0 };
                            let h_pad = if h > 4 * v_step { v_step } else { 0 };
                            let luma_w = w << ss_hor;
                            let luma_h = h << ss_ver;
                            let luma_stride = luma_w * 2 + extra_stride;
                            let luma_len = (luma_h - 1) * luma_stride + luma_w * 2;
                            let probe_y = OverReadProbe::new(luma_len, 64, w * h * 2);
                            let mut out_a = SentinelBuffer::with_defaults(w * h * 2);
                            let mut out_b = SentinelBuffer::with_defaults(w * h * 2);

                            let res = probe_y.run_pair(
                                out_a.data_mut(),
                                out_b.data_mut(),
                                |input, out| {
                                    let start = probe_y.data_offset();
                                    let mut masked = input.to_vec();
                                    for c in masked[start..start + luma_len].as_chunks_mut::<2>().0
                                    {
                                        let v = u16::from_le_bytes(*c) & bd_max;
                                        *c = v.to_le_bytes();
                                    }
                                    let mut ac = vec![0i16; w * h];
                                    ipred_simd.cfl_ac[layout](
                                        &mut ac,
                                        &masked[start..],
                                        luma_stride,
                                        w_pad,
                                        h_pad,
                                        w,
                                        h,
                                        bit_depth,
                                    );
                                    for (c, &v) in out.as_chunks_mut::<2>().0.iter_mut().zip(&ac) {
                                        *c = v.to_le_bytes();
                                    }
                                },
                            );
                            assert!(res.is_ok(), "{label} cfl_ac[{layout}] {w}x{h}: {res:?}");
                            out_a.verify().expect("ön/arka koruyucu bozuldu");
                            out_b.verify().expect("ön/arka koruyucu bozuldu");

                            let mut y_ref = probe_y.payload();
                            for c in y_ref.as_chunks_mut::<2>().0 {
                                let v = u16::from_le_bytes(*c) & bd_max;
                                *c = v.to_le_bytes();
                            }
                            let mut ac_ref = vec![0i16; w * h];
                            ipred_ref.cfl_ac[layout](
                                &mut ac_ref,
                                &y_ref,
                                luma_stride,
                                w_pad,
                                h_pad,
                                w,
                                h,
                                bit_depth,
                            );
                            let mut out_ref = vec![0u8; w * h * 2];
                            for (c, &v) in out_ref.as_chunks_mut::<2>().0.iter_mut().zip(&ac_ref) {
                                *c = v.to_le_bytes();
                            }
                            assert_eq!(out_a.data(), &out_ref[..]);
                        }

                        let mut ac_in = vec![0i16; w * h];
                        for (i, slot) in ac_in.iter_mut().enumerate() {
                            *slot = ((i as i16 * 13) % 511) - 255;
                        }
                        for mode in CflPredMode::ALL {
                            let (back, front) = (h, w);
                            let data_len = (back + 1 + front) * 2;
                            let back_bytes = back * 2;
                            let probe_e = OverReadProbe::new(data_len, 64, dst_len);
                            let topleft = probe_e.data_offset() + back_bytes;
                            let mut dst_a = SentinelBuffer::with_defaults(dst_len);
                            let mut dst_b = SentinelBuffer::with_defaults(dst_len);

                            let res = probe_e.run_pair(
                                dst_a.data_mut(),
                                dst_b.data_mut(),
                                |input, out| {
                                    out.fill(0x33);
                                    let mut masked = input.to_vec();
                                    let start = probe_e.data_offset();
                                    for c in masked[start..start + data_len].as_chunks_mut::<2>().0
                                    {
                                        let v = u16::from_le_bytes(*c) & bd_max;
                                        *c = v.to_le_bytes();
                                    }
                                    ipred_simd.cfl_pred[mode.index()](
                                        out,
                                        stride,
                                        Edge::new(&masked, topleft),
                                        w,
                                        h,
                                        &ac_in,
                                        -11,
                                        bit_depth,
                                        bd_max,
                                    );
                                },
                            );
                            assert!(res.is_ok());
                            dst_a.verify().expect("ön/arka koruyucu bozuldu");
                            dst_b.verify().expect("ön/arka koruyucu bozuldu");

                            let mut payload_edge = probe_e.payload();
                            for c in payload_edge.as_chunks_mut::<2>().0 {
                                let v = u16::from_le_bytes(*c) & bd_max;
                                *c = v.to_le_bytes();
                            }
                            let mut dst_ref = vec![0x33u8; dst_len];
                            ipred_ref.cfl_pred[mode.index()](
                                &mut dst_ref,
                                stride,
                                Edge::new(&payload_edge, back_bytes),
                                w,
                                h,
                                &ac_in,
                                -11,
                                bit_depth,
                                bd_max,
                            );
                            assert_eq!(dst_a.data(), &dst_ref[..]);
                        }
                    }

                    // pal_pred
                    let mut pal = [0u8; 16];
                    for (c, slot) in pal.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                        let v = ((c * 131 + 7) as u16) & bd_max;
                        *slot = v.to_le_bytes();
                    }
                    let idx_len = w * h / 2;
                    let probe_idx = OverReadProbe::new(idx_len, 64, dst_len);
                    let mut dst_a = SentinelBuffer::with_defaults(dst_len);
                    let mut dst_b = SentinelBuffer::with_defaults(dst_len);

                    let res =
                        probe_idx.run_pair(dst_a.data_mut(), dst_b.data_mut(), |input, out| {
                            out.fill(0x77);
                            let mut masked = input.to_vec();
                            let start = probe_idx.data_offset();
                            for b in &mut masked[start..start + idx_len] {
                                *b &= 0x77;
                            }
                            (ipred_simd.pal_pred)(
                                out,
                                stride,
                                &pal,
                                &masked[start..],
                                w,
                                h,
                                bit_depth,
                            );
                        });
                    assert!(res.is_ok());
                    dst_a.verify().expect("ön/arka koruyucu bozuldu");
                    dst_b.verify().expect("ön/arka koruyucu bozuldu");

                    let mut idx_ref = probe_idx.payload();
                    for b in &mut idx_ref {
                        *b &= 0x77;
                    }
                    let mut dst_ref = vec![0x77u8; dst_len];
                    (ipred_ref.pal_pred)(&mut dst_ref, stride, &pal, &idx_ref, w, h, bit_depth);
                    assert_eq!(dst_a.data(), &dst_ref[..]);
                }
            }
        }
    }

    assert_eq!(
        ipred_skaler_devir_sayisi(),
        before,
        "Normal Sentinel/OverRead testlerinde skaler devir sayacı artmamalı"
    );
}

#[test]
fn ipred16_sozlesme_disi_skaler_devir() {
    let Some(t) = ipred16_tablolari("ipred_16bpc_x86::ipred16_sozlesme_disi_skaler_devir") else {
        return;
    };
    let _guard = DEVIR_KILIDI.lock().unwrap_or_else(|e| e.into_inner());
    let dsp = &t.avx512;
    let ipred10 = dsp.for_bit_depth(BitDepth::Ten).ipred();
    let ipred12 = dsp.for_bit_depth(BitDepth::Twelve).ipred();

    let mut dst = vec![0u8; 64];
    let edge = vec![0u8; 64];

    // 1. Kısa dst tamponu — skaler devir sayacı artmalı, paniklemesi beklenen
    // durum `catch_unwind` ile yakalanmalı.
    let c0 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut short_dst = [0u8; 4];
        ipred10.intra_pred[IntraPredMode::Dc.index()](
            &mut short_dst,
            8,
            Edge::new(&edge, 16),
            4,
            4,
            0,
            4,
            4,
            BitDepth::Ten,
            1023,
        );
    }));
    assert!(ipred_skaler_devir_sayisi() > c0);

    // 2. Tek baytlık dst_stride (dst_stride % 2 != 0)
    let c1 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ipred10.intra_pred[IntraPredMode::Vert.index()](
            &mut dst,
            9,
            Edge::new(&edge, 16),
            4,
            4,
            0,
            4,
            4,
            BitDepth::Ten,
            1023,
        );
    }));
    assert!(ipred_skaler_devir_sayisi() > c1);

    // 3. Tek baytlık edge.topleft (topleft % 2 != 0)
    let c2 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ipred10.intra_pred[IntraPredMode::Paeth.index()](
            &mut dst,
            8,
            Edge::new(&edge, 15),
            4,
            4,
            0,
            4,
            4,
            BitDepth::Ten,
            1023,
        );
    }));
    assert!(ipred_skaler_devir_sayisi() > c2);

    // 4. Yanlış BitDepth::Eight
    let c3 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ipred10.intra_pred[IntraPredMode::Smooth.index()](
            &mut dst,
            8,
            Edge::new(&edge, 16),
            4,
            4,
            0,
            4,
            4,
            BitDepth::Eight,
            1023,
        );
    }));
    assert!(ipred_skaler_devir_sayisi() > c3);

    // 5. Yanlış bitdepth_max (10 bit için 4095, 12 bit için 1023)
    let c4 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ipred10.intra_pred[IntraPredMode::Dc128.index()](
            &mut dst,
            8,
            Edge::new(&edge, 16),
            4,
            4,
            0,
            4,
            4,
            BitDepth::Ten,
            4095,
        );
    }));
    assert!(ipred_skaler_devir_sayisi() > c4);

    let c5 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ipred12.intra_pred[IntraPredMode::Dc128.index()](
            &mut dst,
            8,
            Edge::new(&edge, 16),
            4,
            4,
            0,
            4,
            4,
            BitDepth::Twelve,
            1023,
        );
    }));
    assert!(ipred_skaler_devir_sayisi() > c5);

    // 6. cfl_ac, cfl_pred ve pal_pred bozuk girdiler
    let c6 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut short_ac = [0i16; 4];
        ipred10.cfl_ac[0](&mut short_ac, &edge, 16, 0, 0, 4, 4, BitDepth::Ten);
    }));
    assert!(ipred_skaler_devir_sayisi() > c6);

    let c7 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let short_ac = [0i16; 4];
        ipred10.cfl_pred[0](
            &mut dst,
            8,
            Edge::new(&edge, 16),
            4,
            4,
            &short_ac,
            1,
            BitDepth::Ten,
            1023,
        );
    }));
    assert!(ipred_skaler_devir_sayisi() > c7);

    let c8 = ipred_skaler_devir_sayisi();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let short_pal = [0u8; 8];
        let idx = [0u8; 8];
        (ipred10.pal_pred)(&mut dst, 8, &short_pal, &idx, 4, 4, BitDepth::Ten);
    }));
    assert!(ipred_skaler_devir_sayisi() > c8);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn ipred16_objdump_isa_siniri() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "-C"])
        .arg(&exe)
        .output()
        .expect("objdump çalıştırılamadı");
    assert!(out.status.success(), "objdump başarısız oldu");
    let text = String::from_utf8_lossy(&out.stdout);

    let sse41_forbidden = [
        "packusdw", "pminuw", "pmaxuw", "pminsd", "pmaxsd", "pminsb", "pmaxsb", "pminud", "pmaxud",
        "pmulld", "pmuldq", "pmovsxbw", "pmovzxbw", "pmovsxwd", "pmovzxwd", "pmovsxbd", "pmovzxbd",
        "pblendw", "pblendvb", "pinsrb", "pinsrd", "pinsrq", "pextrb", "pextrd", "pextrq", "ptest",
    ];

    let mut mode = 0u8; // 0: dış, 1: ssse3, 2: avx2, 3: avx512
    let mut ssse3_syms = 0usize;
    let mut avx2_syms = 0usize;
    let mut avx512_syms = 0usize;

    for line in text.lines() {
        if line.ends_with(">:") {
            if line.contains("dsp::x86::ipred16_ssse3::") {
                mode = 1;
                ssse3_syms += 1;
            } else if line.contains("dsp::x86::ipred16::") {
                mode = 2;
                avx2_syms += 1;
            } else if line.contains("dsp::x86::ipred16_avx512::") {
                mode = 3;
                avx512_syms += 1;
            } else {
                mode = 0;
            }
            continue;
        }
        if mode == 0 {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            continue;
        }
        let insn = parts[2].trim();
        let mnemonic = insn.split_whitespace().next().unwrap_or("");
        if mnemonic.is_empty() {
            continue;
        }

        match mode {
            1 => {
                assert!(
                    !mnemonic.starts_with('v')
                        && !insn.contains("%ymm")
                        && !insn.contains("%zmm")
                        && !insn.contains("{%k"),
                    "ipred16_ssse3 içinde VEX/EVEX komutu bulundu: {line}"
                );
                for &bad in &sse41_forbidden {
                    assert!(
                        mnemonic != bad,
                        "ipred16_ssse3 içinde yasak SSE4.1 komutu ({bad}) bulundu: {line}"
                    );
                }
            }
            2 => {
                assert!(
                    !insn.contains("%zmm") && !insn.contains("{%k"),
                    "ipred16 (AVX2) içinde EVEX komutu bulundu: {line}"
                );
            }
            3 => {
                assert!(
                    !mnemonic.starts_with("vpgather") && !mnemonic.starts_with("vpscatter"),
                    "ipred16_avx512 içinde gather/scatter komutu bulundu: {line}"
                );
            }
            _ => {}
        }
    }

    assert!(ssse3_syms > 0, "ipred16_ssse3 sembolleri bulunamadı");
    assert!(avx2_syms > 0, "ipred16 sembolleri bulunamadı");
    assert!(avx512_syms > 0, "ipred16_avx512 sembolleri bulunamadı");
}
