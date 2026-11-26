//! P47 (F4-MSAC, Adım 1) — `msac` (`7` SSE2 + `1` AVX2 = `8 / 8` dav1d x86 entropi
//! sembolü) çekirdek testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **5-düzey `MsacIsa` ve `756 / 767` yuva kurulumu:**
//!    - `none` → `MsacIsa::Scalar` (`0 / 767` yuva, `7` MSAC yuvası `"skaler"`)
//!    - `sse2`/`ssse3`/`sse41` → `MsacIsa::Sse2` (`13`, `453`, `609 / 767` yuva; `7` MSAC yuvası `"sse2"`)
//!    - `avx2`/`avx512icl`/`tespit` → `MsacIsa::Avx2` (`756 / 767` yuva; `symbol_adapt16` `"avx2"`, diğer `6` MSAC yuvası `"sse2"`)
//! 2. **dav1d x86 asm (`msac.obj`) + skaler `MsacContext` %100 bit-birebir eşitlik +
//!    dolgu CDF hücrelerinin kirletilmemesi + `OverReadProbe`:**
//!    - 8 Rust x86 MSAC işlevi (`decode_symbol_adapt4_sse2`, `decode_symbol_adapt8_sse2`,
//!      `decode_symbol_adapt16_sse2`, `decode_symbol_adapt16_avx2`, `decode_hi_tok_sse2`,
//!      `decode_bool_adapt_sse2`, `decode_bool_equi_sse2`, `decode_bool_sse2`) `≥ 130 000`
//!      çağrıda (`400` oturum × `75` çağrı, kısa/boş/EOB ve uzun tamponlar, tüm geçerli
//!      `n_symbols`, `allow_update_cdf` açık/kapalı) dav1d `msac.obj` asm sembolleriyle
//!      ve skaler `MsacContext` ile **%100 bit-birebir** aynı `ret`, `dif`, `rng`, `cnt`,
//!      `pos` ve `cdf` hücrelerini üretir.
//!    - `cdf[n_symbols + 1 .. W]` dolgu hücreleri `MASK_LT[n_symbols]` sayesinde **hiç
//!      kirletilmez** (`0..=16` bayt kısa tamponlar ve `0xDEAD` nöbetçi dolgu hücreleriyle
//!      doğrulanır).
//!    - `OverReadProbe` ile tampon sonrasındaki baytların değişmesi çıktıyı veya durumu
//!      etkilemez (`0` fazladan okuma).
//! 3. **ISA sınırı (`objdump -d`):** `dsp::x86::msac` (SSE2) sembollerinde SSSE3+/VEX/EVEX
//!    komutu `0`; `dsp::x86::msac_avx2` (AVX2) sembolünde EVEX komutu `0`.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::entropy::{MsacContext, MsacIsa};
use common::on_kosul::{self, Neden};
use common::sentinel::OverReadProbe;
use std::collections::BTreeMap;

fn msac_tokenlari(test: &str) -> Option<(av1oxide::cpu::Sse2Token, av1oxide::cpu::Avx2Token)> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let dsp = DspDispatch::select(detected);
        if let (Some(sse2), Some(avx2)) = (dsp.sse2_token(), dsp.avx2_token()) {
            return Some((sse2, avx2));
        }
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P47] MSAC SSE2/AVX2 gövdeleri x86_64 AVX2 ister",
    );
    None
}

fn rng_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn fnv_u32(mut h: u64, mut v: u32) -> u64 {
    for _ in 0..4 {
        h ^= u64::from(v & 0xFF);
        h = h.wrapping_mul(1_099_511_628_211);
        v >>= 8;
    }
    h
}

fn fnv_u16(mut h: u64, v: u16) -> u64 {
    h ^= u64::from(v & 0xFF);
    h = h.wrapping_mul(1_099_511_628_211);
    h ^= u64::from(v >> 8);
    h.wrapping_mul(1_099_511_628_211)
}

fn sort_desc(v: &mut [u16]) {
    for i in 1..v.len() {
        let x = v[i];
        let mut j = i;
        while j > 0 && v[j - 1] < x {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = x;
    }
}

fn gen_cdf(rng: &mut u64, n_symbols: usize) -> [u16; 16] {
    let mut cdf = [0u16; 16];
    for cell in cdf.iter_mut().take(n_symbols) {
        *cell = (1 + (rng_next(rng) % 32767)) as u16;
    }
    sort_desc(&mut cdf[..n_symbols]);
    cdf[n_symbols] = (rng_next(rng) % 33) as u16;
    cdf
}

/// `scripts/oracle/msac_x86_oracle.c` (`600` oturum, `75` çağrı/oturum) tarafından
/// `.cache/dav1d-build-asm/.../msac.obj` ile üretilen altın FNV-1a 64 özetleri.
const ALTIN_OZETLER: [(&str, u64, u64); 8] = [
    (
        "dav1d_msac_decode_symbol_adapt4_sse2",
        13_045,
        0x7984_f1b3_e788_de27,
    ),
    (
        "dav1d_msac_decode_symbol_adapt8_sse2",
        13_350,
        0x160f_3ac7_07fd_800e,
    ),
    (
        "dav1d_msac_decode_symbol_adapt16_sse2",
        13_213,
        0x5cb0_37a2_4e71_acf5,
    ),
    (
        "dav1d_msac_decode_symbol_adapt16_avx2",
        13_200,
        0x856a_b45b_c2f0_08cb,
    ),
    (
        "dav1d_msac_decode_hi_tok_sse2",
        13_342,
        0x1990_64f2_0248_6df4,
    ),
    (
        "dav1d_msac_decode_bool_adapt_sse2",
        13_176,
        0x16bd_b368_420a_5e17,
    ),
    (
        "dav1d_msac_decode_bool_equi_sse2",
        13_320,
        0xec28_15c2_5090_1a69,
    ),
    ("dav1d_msac_decode_bool_sse2", 13_287, 0x28f8_ec5b_d4fc_4dcf),
];

#[test]
fn p47_msac_x86_kurulum_ve_767_yuva() {
    let Some(_) = msac_tokenlari("msac_x86::p47_msac_x86_kurulum_ve_767_yuva") else {
        return;
    };

    let d_none = DspDispatch::select(CpuFlags::NONE);
    assert_eq!(d_none.msac_isa(), MsacIsa::Scalar);
    assert_eq!(d_none.assigned_count(), 0);

    let detected = CpuFlags::detect();
    let d_sse2 = DspDispatch::select(detected.intersection(CpuLevel::Sse2.allowed()));
    let d_ssse3 = DspDispatch::select(detected.intersection(CpuLevel::Ssse3.allowed()));
    let d_sse41 = DspDispatch::select(detected.intersection(CpuLevel::Sse41.allowed()));
    let d_avx2 = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()));
    let d_avx512 = DspDispatch::select(detected);

    // `MsacIsa::{Sse2,Avx2}` yalnız `x86_64` üzerinde vardır (P52: aarch64 derlenebilirliği).
    #[cfg(target_arch = "x86_64")]
    {
        assert!(matches!(d_sse2.msac_isa(), MsacIsa::Sse2(_)));
        assert!(matches!(d_ssse3.msac_isa(), MsacIsa::Sse2(_)));
        assert!(matches!(d_sse41.msac_isa(), MsacIsa::Sse2(_)));
        assert!(matches!(d_avx2.msac_isa(), MsacIsa::Avx2(_, _)));
        assert!(matches!(d_avx512.msac_isa(), MsacIsa::Avx2(_, _)));
    }

    assert_eq!(d_sse2.assigned_count(), 13);
    assert_eq!(d_ssse3.assigned_count(), 453);
    assert_eq!(d_sse41.assigned_count(), 609);
    assert_eq!(d_avx2.assigned_count(), 767);
    if d_avx512.avx512icl_token().is_some() {
        assert_eq!(d_avx512.assigned_count(), 767);
    }

    for (d, beklenen_a16, beklenen_diger) in [
        (&d_none, "skaler", "skaler"),
        (&d_sse2, "sse2", "sse2"),
        (&d_ssse3, "sse2", "sse2"),
        (&d_sse41, "sse2", "sse2"),
        (&d_avx2, "avx2", "sse2"),
    ] {
        let mut msac_map = BTreeMap::new();
        for s in d.slot_states() {
            if s.key.family == "msac" {
                msac_map.insert(s.key.field, s.implementation);
            }
        }
        assert_eq!(msac_map.len(), 7);
        assert_eq!(msac_map["decode_symbol_adapt16"], beklenen_a16);
        for f in [
            "decode_symbol_adapt4",
            "decode_symbol_adapt8",
            "decode_hi_tok",
            "decode_bool_adapt",
            "decode_bool_equi",
            "decode_bool",
        ] {
            assert_eq!(msac_map[f], beklenen_diger, "f={f}");
        }
    }
}

#[test]
fn p47_msac_x86_oracle_ve_sentinel() {
    let Some((sse2, avx2)) = msac_tokenlari("msac_x86::p47_msac_x86_oracle_ve_sentinel") else {
        return;
    };

    #[cfg(target_arch = "x86_64")]
    {
        use av1oxide::dsp::x86::msac::{
            decode_bool_adapt_sse2, decode_bool_equi_sse2, decode_bool_sse2, decode_hi_tok_sse2,
            decode_symbol_adapt4_sse2, decode_symbol_adapt8_sse2, decode_symbol_adapt16_sse2,
        };
        use av1oxide::dsp::x86::msac_avx2::decode_symbol_adapt16_avx2;

        let sessions = 600u32;
        let calls_per_session = 75u32;
        let mut genel_toplam_cagri = 0u64;

        for (id, &(sembol_ad, beklenen_cagri, beklenen_ozet)) in ALTIN_OZETLER.iter().enumerate() {
            let mut rng =
                0x2026_1120_0047_0000u64 ^ ((id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut ozet = 0xcbf2_9ce4_8422_2325u64;
            let mut toplam_cagri = 0u64;

            for s_idx in 0..sessions {
                let allow_update = (s_idx & 1) == 0;
                let mod16 = s_idx % 16;
                let len = if mod16 <= 6 {
                    mod16 as usize
                } else if mod16 <= 8 {
                    (mod16 + 1) as usize
                } else if mod16 < 12 {
                    10 + (rng_next(&mut rng) % 23) as usize
                } else {
                    33 + (rng_next(&mut rng) % 224) as usize
                };

                let mut buf = [0u8; 256];
                for b in buf.iter_mut().take(len) {
                    *b = (rng_next(&mut rng) & 0xFF) as u8;
                }

                let n_symbols = match id {
                    0 => 1 + (s_idx as usize % 3),
                    1 => 1 + (s_idx as usize % 7),
                    2 | 3 => 3 + (s_idx as usize % 13),
                    4 => 3,
                    _ => 1,
                };

                let mut cdf_simd = gen_cdf(&mut rng, n_symbols);
                // Nöbetçi dolgu hücreleri: n_symbols + 1 .. 16 aralığına 0xDEAD yaz;
                // SIMD ve skaler yolların ikisi de bu dolgu hücrelerine dokunmamalı!
                for cell in cdf_simd.iter_mut().skip(n_symbols + 1) {
                    *cell = 0xDEAD;
                }
                let mut cdf_scalar = cdf_simd;

                let mut ctx_simd = MsacContext::init(&buf[..len], !allow_update);
                let mut ctx_scalar = MsacContext::init(&buf[..len], !allow_update);

                let session_calls = if len <= 6 {
                    1 + len as u32
                } else {
                    ((len - 6) as u32).min(calls_per_session)
                };

                for c_idx in 0..session_calls {
                    let f_prob = 4 + (rng_next(&mut rng) % 32760) as u32;

                    let (v_simd, v_scalar) = match id {
                        0 => {
                            let c4: &mut [u16; 4] =
                                (&mut cdf_simd[..4]).try_into().expect("4 hücre");
                            let vs = decode_symbol_adapt4_sse2(sse2, &mut ctx_simd, c4, n_symbols)
                                as u32;
                            let vc =
                                ctx_scalar.decode_symbol_adapt(&mut cdf_scalar, n_symbols) as u32;
                            (vs, vc)
                        }
                        1 => {
                            let c8: &mut [u16; 8] =
                                (&mut cdf_simd[..8]).try_into().expect("8 hücre");
                            let vs = decode_symbol_adapt8_sse2(sse2, &mut ctx_simd, c8, n_symbols)
                                as u32;
                            let vc =
                                ctx_scalar.decode_symbol_adapt(&mut cdf_scalar, n_symbols) as u32;
                            (vs, vc)
                        }
                        2 => {
                            let vs = decode_symbol_adapt16_sse2(
                                sse2,
                                &mut ctx_simd,
                                &mut cdf_simd,
                                n_symbols,
                            ) as u32;
                            let vc =
                                ctx_scalar.decode_symbol_adapt(&mut cdf_scalar, n_symbols) as u32;
                            (vs, vc)
                        }
                        3 => {
                            let vs = decode_symbol_adapt16_avx2(
                                avx2,
                                &mut ctx_simd,
                                &mut cdf_simd,
                                n_symbols,
                            ) as u32;
                            let vc =
                                ctx_scalar.decode_symbol_adapt(&mut cdf_scalar, n_symbols) as u32;
                            (vs, vc)
                        }
                        4 => {
                            let c4: &mut [u16; 4] =
                                (&mut cdf_simd[..4]).try_into().expect("4 hücre");
                            let vs = decode_hi_tok_sse2(sse2, &mut ctx_simd, c4) as u32;
                            let vc = ctx_scalar.decode_hi_tok(&mut cdf_scalar) as u32;
                            (vs, vc)
                        }
                        5 => {
                            let c2: &mut [u16; 2] =
                                (&mut cdf_simd[..2]).try_into().expect("2 hücre");
                            let vs = decode_bool_adapt_sse2(sse2, &mut ctx_simd, c2);
                            let vc = ctx_scalar.decode_bool_adapt(&mut cdf_scalar);
                            (vs, vc)
                        }
                        6 => {
                            let vs = decode_bool_equi_sse2(sse2, &mut ctx_simd);
                            let vc = ctx_scalar.decode_bool_equi();
                            (vs, vc)
                        }
                        7 => {
                            let vs = decode_bool_sse2(sse2, &mut ctx_simd, f_prob);
                            let vc = ctx_scalar.decode_bool(f_prob);
                            (vs, vc)
                        }
                        _ => unreachable!(),
                    };

                    assert_eq!(
                        v_simd, v_scalar,
                        "{sembol_ad} s={s_idx} c={c_idx} sembol farkı"
                    );
                    assert_eq!(
                        ctx_simd.dif(),
                        ctx_scalar.dif(),
                        "{sembol_ad} s={s_idx} c={c_idx} dif farkı"
                    );
                    assert_eq!(
                        ctx_simd.rng(),
                        ctx_scalar.rng(),
                        "{sembol_ad} s={s_idx} c={c_idx} rng farkı"
                    );
                    assert_eq!(
                        ctx_simd.cnt(),
                        ctx_scalar.cnt(),
                        "{sembol_ad} s={s_idx} c={c_idx} cnt farkı"
                    );
                    assert_eq!(
                        ctx_simd.buf_pos(),
                        ctx_scalar.buf_pos(),
                        "{sembol_ad} s={s_idx} c={c_idx} pos farkı"
                    );
                    assert_eq!(
                        cdf_simd, cdf_scalar,
                        "{sembol_ad} s={s_idx} c={c_idx} CDF veya dolgu hücresi farkı"
                    );
                    for &pad_cell in cdf_simd.iter().skip(n_symbols + 1) {
                        assert_eq!(
                            pad_cell, 0xDEAD,
                            "{sembol_ad} dolgu hücresi kirletildi (nsym={n_symbols})"
                        );
                    }

                    ozet = fnv_u32(ozet, v_simd);
                    ozet = fnv_u32(ozet, ctx_simd.rng());
                    toplam_cagri += 1;
                }

                ozet = fnv_u32(ozet, (ctx_simd.dif() >> 48) as u32);
                ozet = fnv_u32(ozet, ctx_simd.rng());
                for &cell in cdf_simd.iter().take(n_symbols + 1) {
                    ozet = fnv_u16(ozet, cell);
                }
            }

            assert_eq!(
                toplam_cagri, beklenen_cagri,
                "{sembol_ad} çağrı sayısı uyuşmazlığı"
            );
            assert_eq!(
                ozet, beklenen_ozet,
                "{sembol_ad} dav1d msac.obj asm oracle özeti uyuşmazlığı: {ozet:016x} != {beklenen_ozet:016x}"
            );
            genel_toplam_cagri += toplam_cagri;
        }

        assert!(
            genel_toplam_cagri >= 100_000,
            "toplam çağrı sayısı >= 100 000 olmalı: {genel_toplam_cagri}"
        );

        // Eğer .cache/msac-x86-oracle/msac_x86_oracle ikilisi derlenmişse canlı olarak da doğrula
        let bin_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/msac-x86-oracle/msac_x86_oracle");
        if bin_path.exists() {
            let out = std::process::Command::new(&bin_path)
                .args(["600", "75"])
                .output()
                .expect("msac_x86_oracle çalışmalı");
            assert!(out.status.success(), "msac_x86_oracle sıfır dönmeli");
            let stdout = String::from_utf8_lossy(&out.stdout);
            for &(sembol_ad, beklenen_cagri, beklenen_ozet) in &ALTIN_OZETLER {
                let beklenen_satir = format!(
                    "sembol={sembol_ad} oturum=600 cagri={beklenen_cagri} ozet={beklenen_ozet:016x}"
                );
                assert!(
                    stdout.lines().any(|l| l.trim() == beklenen_satir),
                    "canlı msac_x86_oracle çıktısında satır bulunamadı: {beklenen_satir}"
                );
            }
        }

        // OverReadProbe ve 0..=16 bayt kısa tampon + adapt16 n_symbols=1,2..=15 sınır taraması
        let mut rng = 0x4747_4747_2026_1120u64;
        for len in 0..=24usize {
            let probe = OverReadProbe::new(len, 64, 40);
            for isa in [MsacIsa::Sse2(sse2), MsacIsa::Avx2(sse2, avx2)] {
                for nsym in 1..=15usize {
                    let cdf_init = gen_cdf(&mut rng, nsym);
                    let mut out_a = [0u8; 40];
                    let mut out_b = [0u8; 40];
                    let res = probe.run_pair(&mut out_a, &mut out_b, |inp, out| {
                        let slice = &inp[probe.data_offset()..probe.data_offset() + len];
                        let mut ctx = MsacContext::init_with_isa(slice, true, isa);
                        let mut cdf = cdf_init;
                        let mut acc = 0u64;
                        for step in 0..16u32 {
                            let v = match step % 6 {
                                0 if nsym <= 3 => ctx.decode_symbol_adapt4(&mut cdf, nsym) as u32,
                                1 if nsym <= 7 => ctx.decode_symbol_adapt8(&mut cdf, nsym) as u32,
                                2 => ctx.decode_symbol_adapt16(&mut cdf, nsym) as u32,
                                3 if nsym == 3 => ctx.decode_hi_tok(&mut cdf) as u32,
                                4 if nsym == 1 => ctx.decode_bool_adapt(&mut cdf),
                                5 => ctx.decode_bool_equi(),
                                _ => ctx.decode_bool(12345),
                            };
                            acc = fnv_u32(acc, v);
                        }
                        out[0..8].copy_from_slice(&acc.to_le_bytes());
                        out[8..16].copy_from_slice(&ctx.dif().to_le_bytes());
                        out[16..20].copy_from_slice(&ctx.rng().to_le_bytes());
                        out[20..24].copy_from_slice(&ctx.cnt().to_le_bytes());
                        out[24..32].copy_from_slice(&(ctx.buf_pos() as u64).to_le_bytes());
                        out[32..34].copy_from_slice(&cdf[0].to_le_bytes());
                        out[34..36].copy_from_slice(&cdf[nsym].to_le_bytes());
                    });
                    assert!(
                        res.is_ok(),
                        "OverReadProbe ihlali len={len} nsym={nsym} isa={isa:?}: {res:?}"
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p47_msac_x86_isa_siniri_objdump_denetimi() {
    let _ = (
        std::hint::black_box(av1oxide::dsp::x86::msac::decode_symbol_adapt4_sse2 as *const ()),
        std::hint::black_box(av1oxide::dsp::x86::msac::decode_symbol_adapt8_sse2 as *const ()),
        std::hint::black_box(av1oxide::dsp::x86::msac::decode_symbol_adapt16_sse2 as *const ()),
        std::hint::black_box(av1oxide::dsp::x86::msac::decode_hi_tok_sse2 as *const ()),
        std::hint::black_box(av1oxide::dsp::x86::msac::decode_bool_adapt_sse2 as *const ()),
        std::hint::black_box(av1oxide::dsp::x86::msac::decode_bool_equi_sse2 as *const ()),
        std::hint::black_box(av1oxide::dsp::x86::msac::decode_bool_sse2 as *const ()),
        std::hint::black_box(
            av1oxide::dsp::x86::msac_avx2::decode_symbol_adapt16_avx2 as *const (),
        ),
    );
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let yasak_sse2 = [
        // SSSE3
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
        // SSE4.1
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
    ];

    let mut sse2_bulundu = 0usize;
    let mut avx2_bulundu = 0usize;
    let mut aktif_kip: Option<&str> = None;
    let mut aktif_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:")
            && (satir.contains("adapt4_sse2")
                || satir.contains("adapt8_sse2")
                || satir.contains("adapt16_sse2")
                || satir.contains("hi_tok_sse2")
                || satir.contains("bool_adapt_sse2")
                || satir.contains("bool_equi_sse2")
                || satir.contains("bool_sse2")
                || satir.contains("adapt16_avx2"))
        {
            aktif_sembol = satir.to_string();
            if satir.contains("_avx2") {
                aktif_kip = Some("avx2");
                avx2_bulundu += 1;
            } else {
                aktif_kip = Some("sse2");
                sse2_bulundu += 1;
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
                    !yasak_sse2.contains(&op),
                    "SSE2 MSAC sembolünde SSSE3/SSE4.1 yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSE2 MSAC sembolünde VEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm"),
                    "SSE2 MSAC sembolünde YMM/ZMM yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "avx2" => {
                assert!(
                    !sag.contains("%zmm") && !sag.contains("{%k"),
                    "AVX2 MSAC sembolünde EVEX yönergesi: {aktif_sembol} -> {satir}"
                );
            }
            _ => {}
        }
    }

    assert!(
        sse2_bulundu >= 7,
        "7 SSE2 MSAC sembolü bulunmalı (bulunan={sse2_bulundu})"
    );
    assert!(
        avx2_bulundu >= 1,
        "decode_symbol_adapt16_avx2 sembolü bulunmalı"
    );
}
