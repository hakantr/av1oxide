//! P45 (F4-PAL, Adım 1) — `pal/pal_idx_finish` SSSE3 (`D03040`/`D03041`), AVX2
//! (`D03036`/`D03037`) ve AVX512ICL (`D03038`/`D03039`) çekirdek testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **Kurulum ve `SLOW_GATHER` bağımsızlığı:** `pal/pal_idx_finish` bit derinliğinden
//!    bağımsız (`bit_depth == 0`) yuva olarak `ssse3`/`sse41` düzeylerinde `"ssse3"`,
//!    `avx2` düzeyinde `"avx2"`, `avx512icl`/`tespit` düzeylerinde (`SLOW_GATHER` açık ya
//!    da kapalı) `"avx512icl"`, `sse2`/`none` düzeylerinde `"skaler"` olarak kurulur.
//! 2. **Oracle eşitliği + tüm geçerli `(bw, bh, w, h)` görünmez kenar dolgusu +
//!    `SentinelBuffer`:** `tests/fixtures/ipred/ipred-small-{8,10,12}-*` ve varsa
//!    `.cache/ipred-oracle/ipred-large-{8,10,12}-*` içindeki tüm `pal_idx_finish`
//!    (`kind == 22`) kayıtlarında ve tüm geçerli `(bw, bh, w, h)` kombinasyonlarında
//!    (`w < bw`, `h < bh` dâhil) SSSE3, AVX2 ve AVX512ICL gövdeleri dav1d C oracle /
//!    skaler referansla **%100 bit-birebir** aynıdır; `SentinelBuffer` taşması `0` ve
//!    `OverReadProbe` fazladan okuması `0`'dır.
//! 3. **Sözleşme dışı güvenli skaler devir:** sözleşme içi çağrılarda
//!    [`pal_skaler_devir_sayisi`] `0` artar; kısa `dst`/`src`, 4'ün katı olmayan `w`/`h`
//!    veya geçersiz `bw`/`bh` girdilerinde paniklemeden sayacı artırır.
//! 4. **ISA sınırı (`objdump -d`):** `pal_ssse3` sembollerinde SSE4.1+/VEX/EVEX `0`,
//!    `pal::` (AVX2) sembollerinde EVEX `0`, `pal_avx512` sembollerinde gather/scatter `0`.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::pal::{PalIdxFinishFn, pal_skaler_devir_sayisi};
use av1oxide::dsp::scalar::pal::pal_idx_finish as scalar_pal_idx_finish;
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer};
use std::fs::File;
use std::io::{BufReader, Read};
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
}

struct PalYuvalar {
    ssse3: PalIdxFinishFn,
    avx2: PalIdxFinishFn,
    avx512: PalIdxFinishFn,
}

fn pal_yuvalari(test: &str) -> Option<PalYuvalar> {
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
            return Some(PalYuvalar {
                ssse3: d_ssse3.pal().pal_idx_finish,
                avx2: d_avx2.pal().pal_idx_finish,
                avx512: d_avx512.pal().pal_idx_finish,
            });
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P45] pal_idx_finish SSSE3/AVX2/AVX512ICL gövdeleri x86_64 AVX512ICL ister",
    );
    None
}

fn read_u8<R: Read>(r: &mut R) -> u8 {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).expect("u8 okuma");
    b[0]
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

fn verify_ipred_oracle_dir(dir: &Path, set_name: &str, y: &PalYuvalar) -> usize {
    let mut pal_records = 0usize;
    for bpc in [8u8, 10, 12] {
        let in_path = dir.join(format!("ipred-{set_name}-{bpc}-input.bin"));
        let exp_path = dir.join(format!("ipred-{set_name}-{bpc}-expect.bin"));
        let (Ok(in_f), Ok(exp_f)) = (File::open(&in_path), File::open(&exp_path)) else {
            continue;
        };
        let mut inp = BufReader::new(in_f);
        let mut exp = BufReader::new(exp_f);

        let mut magic = [0u8; 4];
        inp.read_exact(&mut magic).expect("IPRI magic");
        assert_eq!(&magic, b"IPRI");
        assert_eq!(read_u32(&mut inp), 1);
        assert_eq!(read_u8(&mut inp), bpc);
        let mut pad = [0u8; 3];
        inp.read_exact(&mut pad).expect("pad");
        let count = read_u32(&mut inp);

        exp.read_exact(&mut magic).expect("IPRE magic");
        assert_eq!(&magic, b"IPRE");
        assert_eq!(read_u32(&mut exp), 1);
        assert_eq!(read_u8(&mut exp), bpc);
        exp.read_exact(&mut pad).expect("exp pad");
        assert_eq!(read_u32(&mut exp), count);

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
            let payload_len =
                u32::from_le_bytes([hdr_in[36], hdr_in[37], hdr_in[38], hdr_in[39]]) as usize;

            if kind != 22 {
                skip_bytes(&mut inp, payload_len);
                let exp_len = w * h * 2;
                skip_bytes(&mut exp, exp_len);
                continue;
            }

            assert_eq!(payload_len, bw * bh);
            let mut src = vec![0u8; bw * bh];
            inp.read_exact(&mut src).expect("pal_idx_finish src");
            let dst_len = (bw / 2) * bh;
            let mut expected = vec![0u8; dst_len];
            exp.read_exact(&mut expected).expect("pal_idx_finish exp");

            for (isa_ad, f) in [
                ("ssse3", y.ssse3),
                ("avx2", y.avx2),
                ("avx512icl", y.avx512),
            ] {
                let mut dst = vec![0xA5u8; dst_len];
                f(&mut dst, &src, bw, bh, w, h);
                assert_eq!(
                    dst, expected,
                    "pal_idx_finish oracle farkı ({set_name} bpc={bpc} rec={rec_idx} {isa_ad} bw={bw} bh={bh} w={w} h={h})"
                );
            }
            pal_records += 1;
        }
    }
    pal_records
}

#[test]
fn p45_pal_x86_kurulum_uc_duzey_tam() {
    let Some(_y) = pal_yuvalari("pal_x86::p45_pal_x86_kurulum_uc_duzey_tam") else {
        return;
    };
    for (bits, beklenen) in [
        (0x00u32, "skaler"),
        (0x01u32, "skaler"),
        (0x03u32, "ssse3"),
        (0x07u32, "ssse3"),
        (0x0fu32, "avx2"),
        (0x2fu32, "avx2"),
        (0x1fu32, "avx512icl"),
        (0x3fu32, "avx512icl"),
    ] {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        let st = planned
            .iter()
            .find(|s| s.key.family == "pal" && s.key.field == "pal_idx_finish")
            .expect("pal/pal_idx_finish yuvası");
        assert_eq!(
            st.implementation, beklenen,
            "planlanan bits={bits:#x} pal/pal_idx_finish"
        );
        let d = DspDispatch::select(flags);
        assert_eq!(
            d.independent_slot_implementation("pal", "pal_idx_finish"),
            beklenen,
            "canlı bits={bits:#x} pal/pal_idx_finish"
        );
    }
}

#[test]
fn p45_pal_x86_oracle_ve_sentinel() {
    let Some(y) = pal_yuvalari("pal_x86::p45_pal_x86_oracle_ve_sentinel") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir_once = pal_skaler_devir_sayisi();

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let small_dir = root.join("tests").join("fixtures").join("ipred");
    let small_n = verify_ipred_oracle_dir(&small_dir, "small", &y);
    assert!(
        small_n >= 114,
        "ipred-small içinde en az 114 pal_idx_finish kaydı beklenir: {small_n}"
    );

    let large_dir: PathBuf = root
        .join("..")
        .join("..")
        .join(".cache")
        .join("ipred-oracle");
    if large_dir.join("ipred-large-8-input.bin").exists() {
        let large_n = verify_ipred_oracle_dir(&large_dir, "large", &y);
        assert!(
            large_n >= 19_000,
            "ipred-large içinde en az 19 000 pal_idx_finish kaydı beklenir: {large_n}"
        );
    }

    // Tüm geçerli (bw, bh, w, h) görünmez kenar dolgusu kombinasyonları + SentinelBuffer + OverReadProbe.
    let mut rng = Rng(0x2026_1118_0045_0001);
    for bw in [4usize, 8, 16, 32, 64] {
        for bh in [4usize, 8, 16, 32, 64] {
            let src_len = bw * bh;
            let dst_len = (bw / 2) * bh;
            for w in (4..=bw).step_by(4) {
                for h in (4..=bh).step_by(4) {
                    let mut src = vec![0u8; src_len];
                    for b in &mut src {
                        *b = ((rng.u64() >> 33) & 7) as u8;
                    }
                    let mut ref_dst = vec![0u8; dst_len];
                    scalar_pal_idx_finish(&mut ref_dst, &src, bw, bh, w, h);

                    for (isa_ad, f) in [
                        ("ssse3", y.ssse3),
                        ("avx2", y.avx2),
                        ("avx512icl", y.avx512),
                    ] {
                        let mut sb = SentinelBuffer::with_defaults(dst_len);
                        f(sb.data_mut(), &src, bw, bh, w, h);
                        sb.verify()
                            .expect("SentinelBuffer ön/arka koruyucu bozuldu");
                        assert_eq!(
                            sb.data(),
                            ref_dst.as_slice(),
                            "fark: {isa_ad} bw={bw} bh={bh} w={w} h={h}"
                        );

                        // OverReadProbe: src_len dışındaki baytlar çıktıyı değiştirmemeli.
                        let probe = OverReadProbe::new(src_len, 64, dst_len);
                        let mut out_a = vec![0u8; dst_len];
                        let mut out_b = vec![0u8; dst_len];
                        let res = probe.run_pair(&mut out_a, &mut out_b, |inp, out| {
                            let mut clamped =
                                inp[probe.data_offset()..probe.data_offset() + src_len].to_vec();
                            for v in &mut clamped {
                                *v &= 7;
                            }
                            f(out, &clamped, bw, bh, w, h);
                        });
                        assert!(res.is_ok(), "{isa_ad} over-read: {res:?}");
                    }
                }
            }
        }
    }
    assert_eq!(
        pal_skaler_devir_sayisi(),
        devir_once,
        "sözleşme içi çağrılarda skaler devir olmamalı"
    );
}

#[test]
fn p45_pal_x86_sozlesme_disi_skaler_devir() {
    let Some(y) = pal_yuvalari("pal_x86::p45_pal_x86_sozlesme_disi_skaler_devir") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let src = [3u8; 64];
    for f in [y.ssse3, y.avx2, y.avx512] {
        // 1) Kısa dst
        let d0 = pal_skaler_devir_sayisi();
        let mut kisa_dst = [0u8; 7];
        f(&mut kisa_dst, &src[..16], 4, 4, 4, 4);
        assert_eq!(pal_skaler_devir_sayisi(), d0 + 1);

        // 2) Kısa src
        let d1 = pal_skaler_devir_sayisi();
        let mut dst = [0u8; 32];
        f(&mut dst, &src[..15], 4, 4, 4, 4);
        assert_eq!(pal_skaler_devir_sayisi(), d1 + 1);

        // 3) 4'ün katı olmayan w / h
        let d2 = pal_skaler_devir_sayisi();
        f(&mut dst, &src, 8, 8, 6, 8);
        f(&mut dst, &src, 8, 8, 8, 6);
        assert_eq!(pal_skaler_devir_sayisi(), d2 + 2);

        // 4) Geçersiz bw / bh
        let d3 = pal_skaler_devir_sayisi();
        f(&mut dst, &src, 12, 8, 8, 8);
        f(&mut dst, &src, 8, 12, 8, 8);
        assert_eq!(pal_skaler_devir_sayisi(), d3 + 2);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p45_pal_x86_isa_siniri_objdump_denetimi() {
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
        if satir.ends_with(">:") && satir.contains("pal_idx_finish_") {
            aktif_sembol = satir.to_string();
            if satir.contains("pal_idx_finish_ssse3") {
                aktif_kip = Some("ssse3");
                ssse3_bulundu += 1;
            } else if satir.contains("pal_idx_finish_avx2") {
                aktif_kip = Some("avx2");
                avx2_bulundu += 1;
            } else if satir.contains("pal_idx_finish_avx512icl") {
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
                    "AVX512ICL pal_idx_finish sembolünde gather/scatter yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
            }
            _ => {}
        }
    }

    assert!(ssse3_bulundu >= 1, "pal_idx_finish_ssse3 sembolü bulunmalı");
    assert!(avx2_bulundu >= 1, "pal_idx_finish_avx2 sembolü bulunmalı");
    assert!(
        avx512_bulundu >= 1,
        "pal_idx_finish_avx512icl sembolü bulunmalı"
    );
}
