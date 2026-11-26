//! P54 (`F5d` — `F5-MSAC`) `aarch64` NEON MSAC (`7 / 7` yuva = `8 / 8` `src/arm/64/msac.S`
//! satırı) kurulum, dav1d altın FNV-1a özetleri, skaler `MsacContext` bit-birebir eşitliği,
//! `0xDEAD` dolgu CDF nöbetçi koruması ve `OverReadProbe` testleri.

#![cfg_attr(
    not(target_arch = "aarch64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::{ARM_ISA_CANDIDATES, DspDispatch};
use av1oxide::entropy::{MsacContext, MsacIsa};
#[cfg(not(target_arch = "aarch64"))]
use common::on_kosul::{self, Neden};
use common::sentinel::OverReadProbe;
use std::collections::BTreeMap;

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

/// `dav1d` MSAC referans özetleri (`600` oturum, `75` çağrı/oturum; `msac_x86_oracle.c` ve
/// skaler `MsacContext` ile birebir özdeş 8 tohum dizisi).
const ALTIN_OZETLER: [(&str, u64, u64); 8] = [
    (
        "dav1d_msac_decode_symbol_adapt4_neon",
        13_045,
        0x7984_f1b3_e788_de27,
    ),
    (
        "dav1d_msac_decode_symbol_adapt8_neon",
        13_350,
        0x160f_3ac7_07fd_800e,
    ),
    (
        "dav1d_msac_decode_symbol_adapt16_neon_a",
        13_213,
        0x5cb0_37a2_4e71_acf5,
    ),
    (
        "dav1d_msac_decode_symbol_adapt16_neon_b",
        13_200,
        0x856a_b45b_c2f0_08cb,
    ),
    (
        "dav1d_msac_decode_hi_tok_neon",
        13_342,
        0x1990_64f2_0248_6df4,
    ),
    (
        "dav1d_msac_decode_bool_adapt_neon",
        13_176,
        0x16bd_b368_420a_5e17,
    ),
    (
        "dav1d_msac_decode_bool_equi_neon",
        13_320,
        0xec28_15c2_5090_1a69,
    ),
    ("dav1d_msac_decode_bool_neon", 13_287, 0x28f8_ec5b_d4fc_4dcf),
];

#[test]
fn f5d_msac_arm64_aday_ve_767_yuva_kurulumu() {
    let msac_adaylar: Vec<_> = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "msac")
        .collect();
    assert_eq!(msac_adaylar.len(), 7);
    for field in [
        "decode_symbol_adapt4",
        "decode_symbol_adapt8",
        "decode_symbol_adapt16",
        "decode_hi_tok",
        "decode_bool_adapt",
        "decode_bool_equi",
        "decode_bool",
    ] {
        assert!(
            msac_adaylar.iter().any(|c| c.field == field
                && c.bit_depth == 0
                && c.implementation == "neon"
                && c.requires == CpuFlags::NEON
                && c.default),
            "eksik ARM64 MSAC adayı: msac/{field}"
        );
    }
    assert_eq!(ARM_ISA_CANDIDATES.len(), 785);

    let d_none = DspDispatch::select(CpuFlags::NONE);
    assert_eq!(d_none.msac_isa(), MsacIsa::Scalar);
    assert_eq!(d_none.assigned_count(), 0);

    #[cfg(target_arch = "aarch64")]
    {
        let detected = CpuFlags::detect();
        let d_neon = DspDispatch::select(detected.intersection(CpuLevel::Neon.allowed()));
        let d_dot = DspDispatch::select(detected.intersection(CpuLevel::Dotprod.allowed()));
        let d_tespit = DspDispatch::select(detected);

        assert!(matches!(d_neon.msac_isa(), MsacIsa::Neon(_)));
        assert!(matches!(d_dot.msac_isa(), MsacIsa::Neon(_)));
        assert!(matches!(d_tespit.msac_isa(), MsacIsa::Neon(_)));

        assert_eq!(d_neon.assigned_count(), 767);
        assert_eq!(d_dot.assigned_count(), 767);
        assert_eq!(d_tespit.assigned_count(), 767);

        for (d, beklenen) in [(&d_none, "skaler"), (&d_neon, "neon"), (&d_dot, "neon")] {
            let mut msac_map = BTreeMap::new();
            for s in d.slot_states() {
                if s.key.family == "msac" {
                    msac_map.insert(s.key.field, s.implementation);
                }
            }
            assert_eq!(msac_map.len(), 7);
            for f in [
                "decode_symbol_adapt4",
                "decode_symbol_adapt8",
                "decode_symbol_adapt16",
                "decode_hi_tok",
                "decode_bool_adapt",
                "decode_bool_equi",
                "decode_bool",
            ] {
                assert_eq!(msac_map[f], beklenen, "f={f}");
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5d_msac_arm64_oracle_altin_ozet_ve_overread() {
    #[cfg(target_arch = "aarch64")]
    {
        let d_neon = DspDispatch::select(CpuFlags::detect());
        let neon_isa = d_neon.msac_isa();
        assert!(matches!(neon_isa, MsacIsa::Neon(_)));

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
                for cell in cdf_simd.iter_mut().skip(n_symbols + 1) {
                    *cell = 0xDEAD;
                }
                let mut cdf_scalar = cdf_simd;

                let mut ctx_simd = MsacContext::init_with_isa(&buf[..len], !allow_update, neon_isa);
                let mut ctx_scalar =
                    MsacContext::init_with_isa(&buf[..len], !allow_update, MsacIsa::Scalar);

                let session_calls = if len <= 6 {
                    1 + len as u32
                } else {
                    ((len - 6) as u32).min(calls_per_session)
                };

                for c_idx in 0..session_calls {
                    let f_prob = 4 + (rng_next(&mut rng) % 32760) as u32;

                    let (v_simd, v_scalar) = match id {
                        0 => (
                            ctx_simd.decode_symbol_adapt4(&mut cdf_simd, n_symbols) as u32,
                            ctx_scalar.decode_symbol_adapt(&mut cdf_scalar, n_symbols) as u32,
                        ),
                        1 => (
                            ctx_simd.decode_symbol_adapt8(&mut cdf_simd, n_symbols) as u32,
                            ctx_scalar.decode_symbol_adapt(&mut cdf_scalar, n_symbols) as u32,
                        ),
                        2 | 3 => (
                            ctx_simd.decode_symbol_adapt16(&mut cdf_simd, n_symbols) as u32,
                            ctx_scalar.decode_symbol_adapt(&mut cdf_scalar, n_symbols) as u32,
                        ),
                        4 => (
                            ctx_simd.decode_hi_tok(&mut cdf_simd) as u32,
                            ctx_scalar.decode_hi_tok(&mut cdf_scalar) as u32,
                        ),
                        5 => (
                            ctx_simd.decode_bool_adapt(&mut cdf_simd),
                            ctx_scalar.decode_bool_adapt(&mut cdf_scalar),
                        ),
                        6 => (ctx_simd.decode_bool_equi(), ctx_scalar.decode_bool_equi()),
                        7 => (ctx_simd.decode_bool(f_prob), ctx_scalar.decode_bool(f_prob)),
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
                "{sembol_ad} dav1d altın oracle özeti uyuşmazlığı: {ozet:016x} != {beklenen_ozet:016x}"
            );
            genel_toplam_cagri += toplam_cagri;
        }

        assert!(
            genel_toplam_cagri >= 100_000,
            "toplam çağrı sayısı >= 100 000 olmalı: {genel_toplam_cagri}"
        );

        // OverReadProbe ve 0..=24 bayt kısa tampon + n_symbols=1..=15 sınır taraması
        let mut rng = 0x5454_5454_2026_1123u64;
        for len in 0..=24usize {
            let probe = OverReadProbe::new(len, 64, 40);
            for nsym in 1..=15usize {
                let cdf_init = gen_cdf(&mut rng, nsym);
                let mut out_a = [0u8; 40];
                let mut out_b = [0u8; 40];
                let res = probe.run_pair(&mut out_a, &mut out_b, |inp, out| {
                    let slice = &inp[probe.data_offset()..probe.data_offset() + len];
                    let mut ctx = MsacContext::init_with_isa(slice, true, neon_isa);
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
                    "OverReadProbe ihlali len={len} nsym={nsym}: {res:?}"
                );
            }
        }
    }
}
