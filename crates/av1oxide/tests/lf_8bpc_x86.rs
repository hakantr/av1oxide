//! P46 (F4-LF, Adım 2 / Paket A) — 8 bpc loop filter (deblocking;
//! `loop_filter_sb[2][2]`, 12 kayıt = 4 SSSE3 + 4 AVX2 + 4 AVX512ICL:
//! `D02703`, `D02705`, `D02707`, `D02709`, `D02711`, `D02713`, `D02715`,
//! `D02717`, `D02719`, `D02721`, `D02723`, `D02725`) çekirdek testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **Kurulum ve `SLOW_GATHER` paritesi (`src/x86/loopfilter.h:41-68`):**
//!    - `0x00` (`None`), `0x01` (`SSE2`): 4 yuva `"skaler"`
//!    - `0x03` (`SSSE3`), `0x07` (`SSE4.1`): 4 yuva `"ssse3"`
//!    - `0x0f` (`AVX2`), `0x2f` (`AVX2 | SLOW_GATHER`): 4 yuva `"avx2"`
//!    - `0x3f` (`AVX512ICL | SLOW_GATHER`): `loop_filter_sb[0][0]` ve
//!      `loop_filter_sb[1][0]` `"avx2"`, `loop_filter_sb[0][1]` ve
//!      `loop_filter_sb[1][1]` `"avx512icl"`
//!    - `0x1f` (`AVX512ICL`, `!SLOW_GATHER`): 4 yuvanın tamamı `"avx512icl"`.
//! 2. **Oracle eşitliği (`loopfilter-oracle`) + `SentinelBuffer` / `OverReadProbe`:**
//!    `.cache/loopfilter-oracle/loopfilter-{input,expect}.bin` içindeki bütün 8 bpc
//!    kayıtlarında SSSE3, AVX2 ve AVX512ICL gövdeleri dav1d C oracle ile **%100
//!    bit-birebir** aynı çıktıyı üretir; `wd ∈ {4, 6, 8, 16}`, `flat8in`, `flat8out`,
//!    `hev`, `level == 0` komşu geri düşüşü (`level_or`), `gy == 0` atlaması,
//!    `y0 > 0`, örtüşen `wd = 16` ardışık kenarları ve dar düzlem sınırında
//!    (`gx == 1` / `gy == 1` iken `wd == 4`) `SentinelBuffer` + `OverReadProbe`
//!    denetimleri skaler gövdeyle birebir uyuşur.
//! 3. **Sözleşme dışı güvenli skaler devir:** Bozuk/kısa `dst`, `mask`, `level`
//!    girdilerinde paniklemeden [`lf_skaler_devir_sayisi`] sayacı artar.
//! 4. **ISA sınırı (`objdump -d`):** `lf_ssse3` sembollerinde SSE4.1+/VEX/EVEX `0`,
//!    `x86::lf` (AVX2) sembollerinde EVEX `0`, `lf_avx512` sembollerinde
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
use av1oxide::dsp::lf::{LimLut, LoopFilterDsp, calc_eih, lf_skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

const LF_FIELDS: [&str; 4] = [
    "loop_filter_sb[0][0]",
    "loop_filter_sb[0][1]",
    "loop_filter_sb[1][0]",
    "loop_filter_sb[1][1]",
];

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

struct Lf8Tablolar {
    skaler: LoopFilterDsp,
    ssse3: LoopFilterDsp,
    avx2: LoopFilterDsp,
    avx512: LoopFilterDsp,
}

fn lf8_tablolari(test: &str) -> Option<Lf8Tablolar> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect().difference(CpuFlags::SLOW_GATHER);
        let d_ssse3 = DspDispatch::select(detected.intersection(CpuLevel::Ssse3.allowed()));
        let d_avx2 = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()));
        let d_avx512 = DspDispatch::select(detected);
        if d_ssse3.ssse3_token().is_some()
            && d_avx2.avx2_token().is_some()
            && d_avx512.avx512icl_token().is_some()
        {
            return Some(Lf8Tablolar {
                skaler: LoopFilterDsp::scalar::<8>(),
                ssse3: *d_ssse3.for_bit_depth(BitDepth::Eight).lf(),
                avx2: *d_avx2.for_bit_depth(BitDepth::Eight).lf(),
                avx512: *d_avx512.for_bit_depth(BitDepth::Eight).lf(),
            });
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P46] 8 bpc LF SSSE3/AVX2/AVX512ICL gövdeleri x86_64 AVX512ICL ister",
    );
    None
}

fn find_oracle_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("LOOPFILTER_ORACLE_OUT") {
        let p = PathBuf::from(d);
        if p.join("loopfilter-input.bin").is_file() && p.join("loopfilter-expect.bin").is_file() {
            return Some(p);
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let local = root.join(".cache/loopfilter-oracle");
    if local.join("loopfilter-input.bin").is_file() && local.join("loopfilter-expect.bin").is_file()
    {
        return Some(local);
    }
    let shared = PathBuf::from("/home/hakanbiris/github/av1oxide/.cache/loopfilter-oracle");
    if shared.join("loopfilter-input.bin").is_file()
        && shared.join("loopfilter-expect.bin").is_file()
    {
        return Some(shared);
    }
    None
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

#[test]
fn p46_lf_8bpc_x86_kurulum_ve_slow_gather_paritesi() {
    let Some(_t) = lf8_tablolari("lf_8bpc_x86::p46_lf_8bpc_x86_kurulum_ve_slow_gather_paritesi")
    else {
        return;
    };
    for (bits, beklenen) in [
        (0x00u32, ["skaler", "skaler", "skaler", "skaler"]),
        (0x01u32, ["skaler", "skaler", "skaler", "skaler"]),
        (0x03u32, ["ssse3", "ssse3", "ssse3", "ssse3"]),
        (0x07u32, ["ssse3", "ssse3", "ssse3", "ssse3"]),
        (0x0fu32, ["avx2", "avx2", "avx2", "avx2"]),
        (0x2fu32, ["avx2", "avx2", "avx2", "avx2"]),
        (0x3fu32, ["avx2", "avx512icl", "avx2", "avx512icl"]),
        (
            0x1fu32,
            ["avx512icl", "avx512icl", "avx512icl", "avx512icl"],
        ),
    ] {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        let d = DspDispatch::select(flags);
        for (field, exp) in LF_FIELDS.iter().zip(beklenen.iter()) {
            let st = planned
                .iter()
                .find(|s| {
                    s.key.family == "loopfilter"
                        && s.key.field == *field
                        && s.key.bit_depth == Some(8)
                })
                .expect("8 bpc loopfilter yuvası");
            assert_eq!(
                st.implementation, *exp,
                "planlanan bits={bits:#x} loopfilter/{field}@8"
            );
            assert_eq!(
                d.slot_implementation(BitDepth::Eight, "loopfilter", field),
                *exp,
                "canlı bits={bits:#x} loopfilter/{field}@8"
            );
        }
    }
}

#[test]
fn p46_lf_8bpc_x86_oracle_ve_sentinel() {
    let Some(t) = lf8_tablolari("lf_8bpc_x86::p46_lf_8bpc_x86_oracle_ve_sentinel") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir_once = lf_skaler_devir_sayisi();

    // 1) `.cache/loopfilter-oracle` (varsa bütün 8-bpc kayıtlarında C oracle ile bit-birebir eşitlik).
    if let Some(dir) = find_oracle_dir() {
        let input = std::fs::read(dir.join("loopfilter-input.bin")).expect("loopfilter-input.bin");
        let expect =
            std::fs::read(dir.join("loopfilter-expect.bin")).expect("loopfilter-expect.bin");
        assert_eq!(input.len(), expect.len());
        let lim = oracle_lut();
        let mut p = 0usize;
        let mut tested_8bpc = 0usize;
        while p + 12 <= input.len() {
            let h0u = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
            let h1u = u32::from_le_bytes([input[p + 4], input[p + 5], input[p + 6], input[p + 7]]);
            let stride = usize::from(u16::from_le_bytes([input[p + 8], input[p + 9]]));
            let offp = usize::from(u16::from_le_bytes([input[p + 10], input[p + 11]]));
            p += 12;
            let bd = (h0u & 0xff) as usize;
            let func = ((h0u >> 8) & 0xff) as usize;
            let wd = (h0u >> 16) as usize;
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
            let orig_plane = &input[p..p + plane_len];
            let want_plane = &expect[p..p + plane_len];
            p += plane_len;

            if bd != 8 {
                continue;
            }
            let plane_idx = func >> 1;
            let dir_idx = func & 1;
            let mut l_raw = [[0u8; 4]; 544];
            for (i, e) in l_raw.iter_mut().enumerate() {
                e.copy_from_slice(&lb[i * 4..i * 4 + 4]);
            }

            let mut mask = vec![0u8; 32 * 32];
            let mut lvl = vec![[0u8; 4]; 32 * 32];
            if dir_idx == 0 {
                assert_eq!(offp, 8);
                for i in 0..n {
                    let bit = 1u32 << i;
                    let v = if vmask[2] & bit != 0 {
                        3
                    } else if vmask[1] & bit != 0 {
                        2
                    } else if vmask[0] & bit != 0 {
                        1
                    } else {
                        0
                    };
                    mask[i * 32 + 2] = v;
                    lvl[i * 32 + 2] = l_raw[32 + i * 32];
                    lvl[i * 32 + 1] = l_raw[32 + i * 32 - 1];
                }
            } else {
                assert_eq!(offp, 8 * stride);
                for i in 0..n {
                    let bit = 1u32 << i;
                    let v = if vmask[2] & bit != 0 {
                        3
                    } else if vmask[1] & bit != 0 {
                        2
                    } else if vmask[0] & bit != 0 {
                        1
                    } else {
                        0
                    };
                    mask[2 * 32 + i] = v;
                    lvl[2 * 32 + i] = l_raw[32 + i];
                    lvl[32 + i] = l_raw[i];
                }
            }

            for (isa_ad, dsp) in [
                ("ssse3", &t.ssse3),
                ("avx2", &t.avx2),
                ("avx512icl", &t.avx512),
            ] {
                let mut got = orig_plane.to_vec();
                let f = dsp.loop_filter_sb[plane_idx][dir_idx];
                if dir_idx == 0 {
                    f(&mut got, stride, &mask, 32, &lvl, 32, 0, &lim, 3, 0, n);
                } else {
                    f(&mut got, stride, &mask, 32, &lvl, 32, 0, &lim, n, 2, 1);
                }
                assert_eq!(
                    got.as_slice(),
                    want_plane,
                    "loopfilter-oracle 8bpc farkı ({isa_ad} func={func} wd={wd} n={n} rec={tested_8bpc})"
                );
            }
            tested_8bpc += 1;
        }
        assert!(
            tested_8bpc >= 640,
            "loopfilter-oracle 8bpc kayıt sayısı beklenenden az: {tested_8bpc}"
        );
    }

    // 2) Kapsamlı sentetik + SentinelBuffer + OverReadProbe denetimi:
    //    - Bütün wd ∈ {4, 6, 8, 16}, flat8in, flat8out, hev, level == 0 fallback,
    //      gy == 0 atlaması, y0 > 0, kısmi maske ve ardışık örtüşen wd == 16 kenarları.
    let mut rng = Rng(0x2026_1119_0046_0801);
    for sharp in [0u8, 1, 3, 5, 7] {
        let lut = calc_eih(sharp);
        for plane in 0..2 {
            for dir in 0..2 {
                for &(w4, y0, count, stride_pad) in &[
                    (4usize, 0usize, 4usize, 0usize),
                    (8, 0, 8, 4),
                    (13, 1, 9, 7),
                    (16, 0, 16, 16),
                    (19, 2, 7, 3),
                ] {
                    let h4 = y0 + count;
                    let stride = w4 * 4 + stride_pad;
                    let dst_len = (h4 * 4 - 1) * stride + w4 * 4;
                    let mask_stride = w4 + 3;
                    let lvl_stride = w4 + 5;
                    let mask_len = h4 * mask_stride;
                    let lvl_len = h4 * lvl_stride;

                    for mode in 0..4 {
                        let mut orig = vec![0u8; dst_len];
                        let base_px = rng.aralik(40, 210);
                        for px in &mut orig {
                            let v = match mode {
                                0 => base_px + rng.aralik(-1, 1),   // flat8in + flat8out
                                1 => base_px + rng.aralik(-3, 3),   // fm + !flat
                                2 => base_px + rng.aralik(-25, 25), // hev
                                _ => rng.aralik(0, 255),            // genel
                            };
                            *px = v.clamp(0, 255) as u8;
                        }
                        let mut mask = vec![0u8; mask_len];
                        let mut lvl = vec![[0u8; 4]; lvl_len];
                        let lvl_idx = (plane * 2 + dir) & 3;
                        let max_v = if plane == 0 { 3u8 } else { 2u8 };

                        for gy in y0..h4 {
                            for gx in 0..w4 {
                                let valid_edge = if dir == 0 { gx >= 1 } else { gy >= 1 };
                                if !valid_edge {
                                    lvl[gy * lvl_stride + gx][lvl_idx] = rng.aralik(1, 63) as u8;
                                    continue;
                                }
                                let can_wd16 = if dir == 0 {
                                    gx >= 2 && gx + 1 < w4
                                } else {
                                    gy >= 2 && gy + 1 < h4
                                };
                                let max_allowed_v = if plane == 0 && !can_wd16 { 2 } else { max_v };
                                let v = if rng.aralik(0, 4) == 0 {
                                    0
                                } else {
                                    rng.aralik(1, i32::from(max_allowed_v)) as u8
                                };
                                mask[gy * mask_stride + gx] = v;
                                let l_cur = if rng.aralik(0, 3) == 0 {
                                    0
                                } else {
                                    rng.aralik(1, 63) as u8
                                };
                                lvl[gy * lvl_stride + gx][lvl_idx] = l_cur;
                            }
                        }

                        let mut ref_dst = orig.clone();
                        (t.skaler.loop_filter_sb[plane][dir])(
                            &mut ref_dst,
                            stride,
                            &mask,
                            mask_stride,
                            &lvl,
                            lvl_stride,
                            lvl_idx,
                            &lut,
                            w4,
                            y0,
                            count,
                        );

                        for (isa_ad, dsp) in [
                            ("ssse3", &t.ssse3),
                            ("avx2", &t.avx2),
                            ("avx512icl", &t.avx512),
                        ] {
                            let mut sb = SentinelBuffer::with_defaults(dst_len);
                            assert!(sb.write_data(&orig));
                            (dsp.loop_filter_sb[plane][dir])(
                                sb.data_mut(),
                                stride,
                                &mask,
                                mask_stride,
                                &lvl,
                                lvl_stride,
                                lvl_idx,
                                &lut,
                                w4,
                                y0,
                                count,
                            );
                            assert_eq!(
                                sb.verify(),
                                Ok(()),
                                "SentinelBuffer taşması ({isa_ad} plane={plane} dir={dir} w4={w4} y0={y0} count={count})"
                            );
                            assert_eq!(
                                sb.data(),
                                ref_dst.as_slice(),
                                "Sentetik skaler farkı ({isa_ad} plane={plane} dir={dir} w4={w4} y0={y0} count={count} mode={mode})"
                            );
                        }
                    }
                }
            }
        }
    }

    // 3) Dar düzlem sınırında (`gx == 1` ve `gy == 1` iken `wd == 4`, toplam 8×8 düzlem)
    //    OverReadProbe + SentinelBuffer denetimi (`p2..p6` / `q2..q6` okunmadığının kanıtı).
    let lut = calc_eih(2);
    for plane in 0..2 {
        for dir in 0..2 {
            let w4 = 2usize;
            let count = 2usize;
            let stride = 8usize;
            let dst_len = 8 * 8;
            let mut mask = [0u8; 4];
            let lvl = [[25u8; 4]; 4];
            if dir == 0 {
                mask[1] = 1; // gy=0, gx=1 -> wd=4
                mask[2 + 1] = 1; // gy=1, gx=1 -> wd=4
            } else {
                mask[2] = 1; // gy=1, gx=0 -> wd=4
                mask[2 + 1] = 1; // gy=1, gx=1 -> wd=4
            }
            let probe = OverReadProbe::new(dst_len, 32, dst_len);
            let mut ref_out = probe.payload();
            (t.skaler.loop_filter_sb[plane][dir])(
                &mut ref_out,
                stride,
                &mask,
                2,
                &lvl,
                2,
                0,
                &lut,
                w4,
                0,
                count,
            );

            for (isa_ad, dsp) in [
                ("ssse3", &t.ssse3),
                ("avx2", &t.avx2),
                ("avx512icl", &t.avx512),
            ] {
                let f = dsp.loop_filter_sb[plane][dir];
                let out = probe
                    .run(|input, out| {
                        let start = probe.data_offset();
                        out.copy_from_slice(&input[start..start + dst_len]);
                        let mut sb = SentinelBuffer::with_defaults(dst_len);
                        assert!(sb.write_data(out));
                        f(
                            sb.data_mut(),
                            stride,
                            &mask,
                            2,
                            &lvl,
                            2,
                            0,
                            &lut,
                            w4,
                            0,
                            count,
                        );
                        assert_eq!(sb.verify(), Ok(()));
                        out.copy_from_slice(sb.data());
                    })
                    .unwrap_or_else(|e| {
                        panic!("OverReadProbe ihlali ({isa_ad} plane={plane} dir={dir}): {e:?}")
                    });
                assert_eq!(
                    out, ref_out,
                    "dar sınır farkı ({isa_ad} plane={plane} dir={dir})"
                );
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
fn p46_lf_8bpc_x86_sozlesme_disi_skaler_devir() {
    let Some(t) = lf8_tablolari("lf_8bpc_x86::p46_lf_8bpc_x86_sozlesme_disi_skaler_devir") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let lut = LimLut::default();

    for dsp in [&t.ssse3, &t.avx2, &t.avx512] {
        for plane in 0..2 {
            for dir in 0..2 {
                let f = dsp.loop_filter_sb[plane][dir];
                let d0 = lf_skaler_devir_sayisi();

                // 1) Kısa dst
                let mut kisa_dst = [0u8; 32];
                let mask = [0u8; 16];
                let lvl = [[0u8; 4]; 16];
                f(&mut kisa_dst, 16, &mask, 4, &lvl, 4, 0, &lut, 4, 0, 4);

                // 2) Kısa mask
                let mut dst = [0u8; 16 * 16];
                let kisa_mask = [0u8; 8];
                f(&mut dst, 16, &kisa_mask, 4, &lvl, 4, 0, &lut, 4, 0, 4);

                // 3) Kısa level ve geçersiz lvl_idx (>= 4)
                let kisa_lvl = [[0u8; 4]; 8];
                f(&mut dst, 16, &mask, 4, &kisa_lvl, 4, 0, &lut, 4, 0, 4);
                f(&mut dst, 16, &mask, 4, &lvl, 4, 4, &lut, 4, 0, 4);

                assert_eq!(
                    lf_skaler_devir_sayisi(),
                    d0 + 4,
                    "bozuk girdilerde skaler devir sayacı 4 artmalı (plane={plane} dir={dir})"
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p46_lf_8bpc_x86_isa_siniri_objdump_denetimi() {
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
        if satir.ends_with(">:") {
            aktif_sembol = satir.to_string();
            if satir.contains("lf_ssse3") {
                aktif_kip = Some("ssse3");
                ssse3_bulundu += 1;
            } else if satir.contains("lf_avx512") {
                aktif_kip = Some("avx512");
                avx512_bulundu += 1;
            } else if satir.contains("dsp3x862lf")
                || satir.contains("lpf_h_sb_avx2")
                || satir.contains("lpf_v_sb_avx2")
                || satir.contains("filter_batch_avx2")
                || satir.contains("flush_h_batch_avx2")
                || satir.contains("flush_v_batch_avx2")
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
                    "SSSE3 LF sembolünde SSE4.1 yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSSE3 LF sembolünde VEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm"),
                    "SSSE3 LF sembolünde YMM/ZMM yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "avx2" => {
                assert!(
                    !sag.contains("%zmm") && !sag.contains("{%k"),
                    "AVX2 LF sembolünde EVEX yönergesi: {aktif_sembol} -> {satir}"
                );
            }
            "avx512" => {
                assert!(
                    !op.contains("gather") && !op.contains("scatter"),
                    "AVX512ICL LF sembolünde gather/scatter yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
            }
            _ => {}
        }
    }

    assert!(ssse3_bulundu >= 1, "lf_ssse3 sembolü bulunmalı");
    assert!(avx2_bulundu >= 1, "x86::lf (AVX2) sembolü bulunmalı");
    assert!(avx512_bulundu >= 1, "lf_avx512 sembolü bulunmalı");
}
