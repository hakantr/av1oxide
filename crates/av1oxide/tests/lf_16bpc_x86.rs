//! P46 (F4-LF16) — 16-bpc (`10/12` bit) Loop Filter (Deblocking) SSSE3 (`D02708`,
//! `D02714`, `D02720`, `D02726`), AVX2 (`D02704`, `D02710`, `D02716`, `D02722`) ve
//! AVX512ICL (`D02706`, `D02712`, `D02718`, `D02724`) çekirdek testleri (`12` kayıt /
//! `24` yuva `@10+@12`).
//!
//! Doğrulanan sözleşme:
//! 1. **Kurulum ve `SLOW_GATHER` paritesi:** `BitDepth::Ten` ve `BitDepth::Twelve` için
//!    `loop_filter_sb[0..2][0..2]` yuvaları `ssse3`/`sse41` düzeylerinde `"ssse3"`,
//!    `avx2` düzeyinde `"avx2"`, `0x3f` (`AVX512ICL | SLOW_GATHER`) altında yatay
//!    süzgeçler (`[0][0]`, `[1][0]`) `"avx2"` ve dikey süzgeçler (`[0][1]`, `[1][1]`)
//!    `"avx512icl"`, `0x1f` (`!SLOW_GATHER`) altında 4 yuvanın tamamı `"avx512icl"`,
//!    `sse2`/`none` düzeylerinde `"skaler"` olarak kurulur.
//! 2. **Oracle eşitliği + tüm genişlik/dallanma kombinasyonları + `SentinelBuffer`:**
//!    Varsa `.cache/loopfilter-oracle/loopfilter-{input,expect}.bin` içindeki tüm
//!    10-bit ve 12-bit C oracle kayıtlarında ve sentetik tüm `wd ∈ {4, 6, 8, 16}`,
//!    `flat8in`, `flat8out`, `hev`, `level_or` yedeklemesi, `gy == 0` atlaması,
//!    12-bit uç değerleri (`0`, `4095`), tek piksel adımı (`stride = w4 * 8 + 2`) ve
//!    en sıkı `SentinelBuffer` sınırı durumlarında SSSE3, AVX2 ve AVX512ICL gövdeleri
//!    dav1d C oracle / skaler referansla **%100 bit-birebir** aynıdır; `SentinelBuffer`
//!    ön/arka koruyucu taşması `0`'dır.
//! 3. **Sözleşme dışı güvenli skaler devir:** sözleşme içi çağrılarda
//!    [`lf_skaler_devir_sayisi`] `0` artar; kısa `dst`, kısa `mask`, kısa `level`,
//!    `lvl_idx >= 4` veya tek bayt `stride` (`stride = 129`) girdilerinde paniklemeden
//!    sayacı artırır.
//! 4. **ISA sınırı (`objdump -d`):** `lf16_ssse3` sembollerinde SSE4.1+/VEX/EVEX `0`,
//!    `lf16::` (AVX2) sembollerinde EVEX `0`, `lf16_avx512` sembollerinde
//!    `vpgather*`/`vpscatter*` `0`.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

mod common;

use av1oxide::BitDepth;
use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::lf::{LimLut, LoopFilterDsp, lf_skaler_devir_sayisi};
use common::on_kosul::{self, Neden};
use common::sentinel::SentinelBuffer;
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

const FIELDS: [(&str, usize, usize); 4] = [
    ("loop_filter_sb[0][0]", 0, 0),
    ("loop_filter_sb[0][1]", 0, 1),
    ("loop_filter_sb[1][0]", 1, 0),
    ("loop_filter_sb[1][1]", 1, 1),
];

struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 16) as u32
    }
}

struct Lf16Yuvalar {
    ssse3_10: LoopFilterDsp,
    avx2_10: LoopFilterDsp,
    avx512_10: LoopFilterDsp,
    ssse3_12: LoopFilterDsp,
    avx2_12: LoopFilterDsp,
    avx512_12: LoopFilterDsp,
}

fn lf16_yuvalari(test: &str) -> Option<Lf16Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let f_ssse3 = detected.intersection(CpuLevel::Ssse3.allowed());
        let f_avx2 = detected.intersection(CpuLevel::Avx2.allowed());
        let f_avx512 = detected.difference(CpuFlags::SLOW_GATHER);
        let d_ssse3 = DspDispatch::select(f_ssse3);
        let d_avx2 = DspDispatch::select(f_avx2);
        let d_avx512 = DspDispatch::select(f_avx512);
        if d_ssse3.ssse3_token().is_some()
            && d_avx2.avx2_token().is_some()
            && d_avx512.avx512icl_token().is_some()
        {
            return Some(Lf16Yuvalar {
                ssse3_10: *d_ssse3.for_bit_depth(BitDepth::Ten).lf(),
                avx2_10: *d_avx2.for_bit_depth(BitDepth::Ten).lf(),
                avx512_10: *d_avx512.for_bit_depth(BitDepth::Ten).lf(),
                ssse3_12: *d_ssse3.for_bit_depth(BitDepth::Twelve).lf(),
                avx2_12: *d_avx2.for_bit_depth(BitDepth::Twelve).lf(),
                avx512_12: *d_avx512.for_bit_depth(BitDepth::Twelve).lf(),
            });
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P46] 16-bpc loopfilter SSSE3/AVX2/AVX512ICL gövdeleri x86_64 AVX512ICL ister",
    );
    None
}

fn build_lut(sharp: i32) -> LimLut {
    let mut lut = LimLut::default();
    for level in 0..64i32 {
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

fn find_oracle_dir() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = root.join("../..").join(".cache/loopfilter-oracle");
    if local.join("loopfilter-input.bin").exists() && local.join("loopfilter-expect.bin").exists() {
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
            let shared = repo_root.join(".cache/loopfilter-oracle");
            if shared.join("loopfilter-input.bin").exists()
                && shared.join("loopfilter-expect.bin").exists()
            {
                return Some(shared);
            }
        }
    }
    None
}

fn verify_loopfilter_c_oracle(dir: &Path, y: &Lf16Yuvalar) -> usize {
    let input = std::fs::read(dir.join("loopfilter-input.bin")).expect("loopfilter-input.bin");
    let expect = std::fs::read(dir.join("loopfilter-expect.bin")).expect("loopfilter-expect.bin");
    assert_eq!(
        input.len(),
        expect.len(),
        "loopfilter oracle girdi/beklenen boyutu eşit olmalı"
    );

    let lut = oracle_lut();
    let mut p = 0usize;
    let mut verified_16bpc = 0usize;

    while p + 12 <= input.len() {
        let h0u = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
        let h1u = u32::from_le_bytes([input[p + 4], input[p + 5], input[p + 6], input[p + 7]]);
        let stride_px = usize::from(u16::from_le_bytes([input[p + 8], input[p + 9]]));
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

        if bd != 10 && bd != 12 {
            continue;
        }

        let plane_idx = usize::from(func >= 2);
        let dir = func & 1;
        let stride_bytes = stride_px;

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

        let tables: [(&str, &LoopFilterDsp); 3] = if bd == 10 {
            [
                ("ssse3", &y.ssse3_10),
                ("avx2", &y.avx2_10),
                ("avx512icl", &y.avx512_10),
            ]
        } else {
            [
                ("ssse3", &y.ssse3_12),
                ("avx2", &y.avx2_12),
                ("avx512icl", &y.avx512_12),
            ]
        };

        for (isa_ad, dsp) in tables {
            let mut work = in_plane.to_vec();
            dsp.loop_filter_sb[plane_idx][dir](
                &mut work,
                stride_bytes,
                &mask,
                32,
                &level,
                32,
                0,
                &lut,
                w4,
                y0,
                count,
            );
            if has_second_half {
                dsp.loop_filter_sb[plane_idx][dir](
                    &mut work[stride_bytes..],
                    stride_bytes,
                    &mask_hi,
                    32,
                    &level_hi,
                    32,
                    0,
                    &lut,
                    n - 8,
                    y0,
                    count,
                );
            }
            assert_eq!(
                work.as_slice(),
                exp_plane,
                "C oracle farkı: bd={bd} func={func} ({isa_ad}) rec={verified_16bpc}"
            );
        }

        verified_16bpc += 1;
    }
    verified_16bpc
}

#[test]
fn p46_lf_16bpc_x86_kurulum_ve_slow_gather_paritesi() {
    let Some(_y) = lf16_yuvalari("lf_16bpc_x86::p46_lf_16bpc_x86_kurulum_ve_slow_gather_paritesi")
    else {
        return;
    };

    for bd in [BitDepth::Ten, BitDepth::Twelve] {
        for (bits, beklenen_h, beklenen_v) in [
            (0x00u32, "skaler", "skaler"),
            (0x01u32, "skaler", "skaler"),
            (0x03u32, "ssse3", "ssse3"),
            (0x07u32, "ssse3", "ssse3"),
            (0x0fu32, "avx2", "avx2"),
            (0x2fu32, "avx2", "avx2"),
            (0x1fu32, "avx512icl", "avx512icl"),
            (0x3fu32, "avx2", "avx512icl"),
        ] {
            let flags = CpuFlags::from_dav1d_x86_bits(bits);
            let planned = DspDispatch::planned_slot_states(flags);
            let d = DspDispatch::select(flags);

            for (field, _plane, dir) in FIELDS {
                let beklenen = if dir == 0 { beklenen_h } else { beklenen_v };
                let st = planned
                    .iter()
                    .find(|s| {
                        s.key.bit_depth == Some(bd.bits() as u8)
                            && s.key.family == "loopfilter"
                            && s.key.field == field
                    })
                    .expect("loopfilter 16-bpc yuvası bulunmalı");
                assert_eq!(
                    st.implementation, beklenen,
                    "planlanan bd={bd:?} bits={bits:#x} {field}"
                );
                assert_eq!(
                    d.slot_implementation(bd, "loopfilter", field),
                    beklenen,
                    "canlı bd={bd:?} bits={bits:#x} {field}"
                );
            }
        }
    }
}

#[test]
fn p46_lf_16bpc_x86_oracle_ve_sentinel() {
    let Some(y) = lf16_yuvalari("lf_16bpc_x86::p46_lf_16bpc_x86_oracle_ve_sentinel") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir_once = lf_skaler_devir_sayisi();

    if let Some(oracle_dir) = find_oracle_dir() {
        let n = verify_loopfilter_c_oracle(&oracle_dir, &y);
        assert!(
            n >= 400,
            "loopfilter C oracle içinde en az 400 adet 16-bpc kaydı doğrulanmalı: {n}"
        );
    }

    let scalar_10 = LoopFilterDsp::scalar::<10>();
    let scalar_12 = LoopFilterDsp::scalar::<12>();
    let mut rng = Rng(0x2026_1119_0046_1610);

    for (bd, max_px, scalar_dsp, simd_dsps) in [
        (
            10u32,
            1023u16,
            &scalar_10,
            [
                ("ssse3", &y.ssse3_10),
                ("avx2", &y.avx2_10),
                ("avx512icl", &y.avx512_10),
            ],
        ),
        (
            12u32,
            4095u16,
            &scalar_12,
            [
                ("ssse3", &y.ssse3_12),
                ("avx2", &y.avx2_12),
                ("avx512icl", &y.avx512_12),
            ],
        ),
    ] {
        for (field, plane, dir) in FIELDS {
            let max_v: u8 = if plane == 0 { 3 } else { 2 };
            for sharp in [0i32, 3, 7] {
                let lut = build_lut(sharp);
                for &(w4, y0, count, extra_bytes) in &[
                    (2usize, 1usize, 1usize, 0usize),
                    (4, 0, 4, 2),
                    (5, 1, 3, 6),
                    (8, 0, 4, 0),
                    (16, 0, 4, 2),
                    (32, 1, 4, 0),
                ] {
                    // Tek piksel adımı (çift bayt adımı: extra_bytes = 2 veya 6) dâhil
                    let row_bytes = w4 * 8;
                    let stride = row_bytes + extra_bytes;
                    let total_rows = (y0 + count) * 4;
                    let min_dst_len = (total_rows - 1) * stride + row_bytes;
                    let mask_stride = w4 + 1;
                    let lvl_stride = w4 + 3;
                    let mask_len = (y0 + count - 1) * mask_stride + w4;
                    let lvl_len = (y0 + count - 1) * lvl_stride + w4;

                    // 7 farklı sinyal rejimi:
                    // 0: düz (flat8in + flat8out), 1: orta dalgalı (flat8in, !flat8out),
                    // 2: dar süzgeç (!hev), 3: yüksek kenar varyansı (hev),
                    // 4: 12-bit taşma sınırı (max_px - 2 .. max_px), 5: uç [0, max_px] sıçrama,
                    // 6: rastgele tam aralık + wd=4/6/8/16 + level_or yedekleme
                    for mode in 0..7u32 {
                        let mut base_dst = vec![0u8; min_dst_len];
                        for row in 0..total_rows {
                            for col in 0..(w4 * 4) {
                                let px: u16 = match mode {
                                    0 => max_px / 2 + ((row + col) as u16 & 1),
                                    1 => {
                                        let base = max_px / 2;
                                        let delta = ((row * 3 + col * 5) & 7) as u16;
                                        if (col & 3) == 0 || (row & 3) == 0 {
                                            base + delta
                                        } else {
                                            base + (delta << (bd - 8))
                                        }
                                    }
                                    2 => {
                                        let side = if dir == 0 {
                                            (col / 4) & 1
                                        } else {
                                            (row / 4) & 1
                                        };
                                        let step = 3u16 << (bd - 8);
                                        if side == 0 {
                                            max_px / 2 - step
                                        } else {
                                            max_px / 2 + step
                                        }
                                    }
                                    3 => {
                                        let edge = if dir == 0 { col & 3 } else { row & 3 };
                                        let step = 12u16 << (bd - 8);
                                        match edge {
                                            0 => max_px / 2 + step * 2,
                                            1 => max_px / 2 - step,
                                            2 => max_px / 2 + step,
                                            _ => max_px / 2 - step * 2,
                                        }
                                    }
                                    4 => max_px - ((rng.next_u32() as u16) & 3),
                                    5 => {
                                        if ((row + col) & 1) == 0 {
                                            0
                                        } else {
                                            max_px
                                        }
                                    }
                                    _ => (rng.next_u32() as u16) & max_px,
                                };
                                let off = row * stride + col * 2;
                                base_dst[off..off + 2].copy_from_slice(&px.to_le_bytes());
                            }
                        }

                        let mut mask = vec![0u8; mask_len];
                        let mut level = vec![[0u8; 4]; lvl_len];
                        let lvl_idx = (mode as usize) & 3;

                        for gy in y0..y0 + count {
                            for gx in 0..w4 {
                                let can_wd16 = if dir == 0 {
                                    gx >= 2 && gx + 1 < w4
                                } else {
                                    gy >= 2 && gy + 1 < y0 + count
                                };
                                let v = match (rng.next_u32() % 5) as u8 {
                                    0 => 0,
                                    1 => 1,
                                    2 => 2.min(max_v),
                                    3 => {
                                        if can_wd16 {
                                            max_v
                                        } else {
                                            2.min(max_v)
                                        }
                                    }
                                    _ => {
                                        if mode == 4 && can_wd16 {
                                            max_v
                                        } else {
                                            1
                                        }
                                    }
                                };
                                mask[gy * mask_stride + gx] = v;

                                let l_val = match rng.next_u32() % 4 {
                                    0 => 0u8,
                                    1 => 63u8,
                                    2 => ((rng.next_u32() % 63) + 1) as u8,
                                    _ => 36u8,
                                };
                                let li = gy * lvl_stride + gx;
                                // Bazı kenarlarda doğrudan level[li]=0 bırakıp önceki bloktan
                                // level_or yedeklemesini (li - 1 veya li - lvl_stride) tetikle
                                if (gx + gy) % 3 == 0
                                    && ((dir == 0 && gx > 0) || (dir == 1 && gy > 0))
                                {
                                    level[li][lvl_idx] = 0;
                                    let prev = if dir == 0 { li - 1 } else { li - lvl_stride };
                                    level[prev][lvl_idx] = l_val;
                                } else {
                                    level[li][lvl_idx] = l_val;
                                }
                            }
                        }

                        let mut ref_dst = base_dst.clone();
                        scalar_dsp.loop_filter_sb[plane][dir](
                            &mut ref_dst,
                            stride,
                            &mask,
                            mask_stride,
                            &level,
                            lvl_stride,
                            lvl_idx,
                            &lut,
                            w4,
                            y0,
                            count,
                        );

                        for (isa_ad, dsp) in simd_dsps {
                            let mut sb = SentinelBuffer::with_defaults(min_dst_len);
                            sb.data_mut().copy_from_slice(&base_dst);
                            dsp.loop_filter_sb[plane][dir](
                                sb.data_mut(),
                                stride,
                                &mask,
                                mask_stride,
                                &level,
                                lvl_stride,
                                lvl_idx,
                                &lut,
                                w4,
                                y0,
                                count,
                            );
                            sb.verify()
                                .expect("SentinelBuffer ön/arka koruyucu bozuldu");
                            assert_eq!(
                                sb.data(),
                                ref_dst.as_slice(),
                                "fark: bd={bd} {field} {isa_ad} sharp={sharp} w4={w4} y0={y0} count={count} extra={extra_bytes} mode={mode}"
                            );
                        }
                    }
                }
            }
        }
    }

    assert_eq!(
        lf_skaler_devir_sayisi(),
        devir_once,
        "sözleşme içi çağrılarda skaler devir olmamalı"
    );
}

#[test]
fn p46_lf_16bpc_x86_sozlesme_disi_skaler_devir() {
    let Some(y) = lf16_yuvalari("lf_16bpc_x86::p46_lf_16bpc_x86_sozlesme_disi_skaler_devir") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let lut = build_lut(0);
    let mask = [1u8; 64];
    let level = [[20u8; 4]; 64];

    for dsp in [
        &y.ssse3_10,
        &y.avx2_10,
        &y.avx512_10,
        &y.ssse3_12,
        &y.avx2_12,
        &y.avx512_12,
    ] {
        for (_field, plane, dir) in FIELDS {
            let f = dsp.loop_filter_sb[plane][dir];

            // 1) Kısa dst (mask=0 ile skaler yedek de güvenli çıkar)
            let empty_mask = [0u8; 64];
            let d0 = lf_skaler_devir_sayisi();
            let mut kisa_dst = [0u8; 16];
            f(
                &mut kisa_dst,
                64,
                &empty_mask,
                8,
                &level,
                8,
                0,
                &lut,
                4,
                0,
                2,
            );
            assert_eq!(lf_skaler_devir_sayisi(), d0 + 1);

            // 2) Kısa mask
            let d1 = lf_skaler_devir_sayisi();
            let mut dst = vec![0u8; 64 * 16];
            f(
                &mut dst,
                64,
                &empty_mask[..3],
                8,
                &level,
                8,
                0,
                &lut,
                4,
                0,
                2,
            );
            assert_eq!(lf_skaler_devir_sayisi(), d1 + 1);

            // 3) Kısa level
            let d2 = lf_skaler_devir_sayisi();
            f(
                &mut dst,
                64,
                &empty_mask,
                8,
                &level[..3],
                8,
                0,
                &lut,
                4,
                0,
                2,
            );
            assert_eq!(lf_skaler_devir_sayisi(), d2 + 1);

            // 4) Geçersiz lvl_idx >= 4
            let d3 = lf_skaler_devir_sayisi();
            f(&mut dst, 64, &empty_mask, 8, &level, 8, 4, &lut, 4, 0, 2);
            assert_eq!(lf_skaler_devir_sayisi(), d3 + 1);

            // 5) Tek bayt stride (16-bpc'de stride % 2 != 0 sözleşme dışıdır)
            let d4 = lf_skaler_devir_sayisi();
            let mut odd_dst = vec![0u8; 129 * 16];
            f(&mut odd_dst, 129, &mask, 8, &level, 8, 0, &lut, 4, 0, 2);
            assert_eq!(lf_skaler_devir_sayisi(), d4 + 1);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p46_lf_16bpc_x86_isa_siniri_objdump_denetimi() {
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
            if satir.contains("dsp::x86::lf16_ssse3::") {
                aktif_kip = Some("ssse3");
                ssse3_bulundu += 1;
            } else if satir.contains("dsp::x86::lf16::") {
                aktif_kip = Some("avx2");
                avx2_bulundu += 1;
            } else if satir.contains("dsp::x86::lf16_avx512::") {
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
                    "SSSE3 lf16_ssse3 sembolünde SSE4.1 yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSSE3 lf16_ssse3 sembolünde VEX/EVEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm") && !sag.contains("{%k"),
                    "SSSE3 lf16_ssse3 sembolünde YMM/ZMM/k yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "avx2" => {
                assert!(
                    !sag.contains("%zmm") && !sag.contains("{%k"),
                    "AVX2 lf16 sembolünde EVEX yönergesi: {aktif_sembol} -> {satir}"
                );
            }
            "avx512" => {
                assert!(
                    !op.contains("gather") && !op.contains("scatter"),
                    "AVX512ICL lf16_avx512 sembolünde gather/scatter yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
            }
            _ => {}
        }
    }

    assert!(
        ssse3_bulundu >= 8,
        "en az 8 adet dsp::x86::lf16_ssse3 sembolü bulunmalı: {ssse3_bulundu}"
    );
    assert!(
        avx2_bulundu >= 8,
        "en az 8 adet dsp::x86::lf16 sembolü bulunmalı: {avx2_bulundu}"
    );
    assert!(
        avx512_bulundu >= 8,
        "en az 8 adet dsp::x86::lf16_avx512 sembolü bulunmalı: {avx512_bulundu}"
    );
}
