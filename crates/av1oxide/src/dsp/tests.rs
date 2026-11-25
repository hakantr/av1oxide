//! `DspDispatch` birim testleri (F3c yuva yapısı).

#![forbid(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use super::ISA_CANDIDATES;
use super::{DspDispatch, Selection, slot_keys};
use crate::cpu::CpuFlags;
use crate::frame::BitDepth;
use std::collections::BTreeSet;

/// F3c: zorlanmış skaler kipte **bütün** yuvalar skalerdir; ISA adayı olsa
/// bile kurulmaz ve belirteç üretilmez.
#[test]
fn skaler_kipte_butun_yuvalar_skaler() {
    for flags in [
        CpuFlags::NONE,
        CpuFlags::detect().intersection(crate::cpu::CpuLevel::None.allowed()),
    ] {
        let dsp = DspDispatch::select(flags);
        assert_eq!(dsp.assigned_count(), 0, "{flags}");
        assert!(dsp.sse2_token().is_none());
        assert!(dsp.avx2_token().is_none());
        assert!(dsp.ssse3_token().is_none());
        assert!(dsp.sse41_token().is_none());
        let states = dsp.slot_states();
        assert_eq!(states.len(), slot_keys().len());
        assert!(
            states.iter().all(|s| s.implementation == "skaler"),
            "{flags}"
        );
    }
}

/// P43/P46/P47: `sse2` düzeyinde (`0x01`) yalnız `3` `WHT_WHT` yuvası (`8, 10, 12` bpc),
/// `refmvs/splat_mv`, `looprestoration/wiener[0..2]@8` ve `7` `msac/*` yuvası (`13` yuva)
/// `sse2` olarak atanır; diğer bütün yuvalar skalerdir.
#[test]
fn sse2_duzeyinde_yalniz_wht_yuvalari_atanir() {
    let flags = CpuFlags::detect().intersection(crate::cpu::CpuLevel::Sse2.allowed());
    let dsp = DspDispatch::select(flags);
    assert_eq!(
        dsp.sse2_token().is_some(),
        CpuFlags::detect().contains(CpuFlags::SSE2)
    );
    assert!(dsp.ssse3_token().is_none());
    assert!(dsp.sse41_token().is_none());
    assert!(dsp.avx2_token().is_none());
    assert!(dsp.avx512icl_token().is_none());
    #[cfg(target_arch = "x86_64")]
    if CpuFlags::detect().contains(CpuFlags::SSE2) {
        assert_eq!(dsp.assigned_count(), 13);
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            assert_eq!(
                dsp.slot_implementation(bd, "itx", "itxfm_add[TX_4X4][WHT_WHT]"),
                "sse2"
            );
        }
        assert_eq!(
            dsp.independent_slot_implementation("refmvs", "splat_mv"),
            "sse2"
        );
        for field in ["wiener[0]", "wiener[1]"] {
            assert_eq!(
                dsp.slot_implementation(BitDepth::Eight, "looprestoration", field),
                "sse2"
            );
        }
        for (field, _, _) in super::MSAC_FUNCTIONS {
            assert_eq!(dsp.independent_slot_implementation("msac", field), "sse2");
        }
    }
}

/// Yuva anahtarları tekildir; bit derinliğine bağlı her ailenin üç bit
/// derinliğinde aynı alan kümesi vardır; pal/refmvs tek, MSAC derleme
/// zamanıdır.
#[test]
fn yuva_anahtarlari_tekil_ve_bit_derinligi_basina_tam() {
    let keys = slot_keys();
    let ids: BTreeSet<String> = keys.iter().map(super::SlotKey::id).collect();
    assert_eq!(ids.len(), keys.len(), "yinelenen yuva kimliği");
    let fields = |bd: u8| -> BTreeSet<(&str, String)> {
        keys.iter()
            .filter(|k| k.bit_depth == Some(bd))
            .map(|k| (k.family, k.field.clone()))
            .collect()
    };
    assert_eq!(fields(8), fields(10));
    assert_eq!(fields(8), fields(12));
    let count = |family: &str, bd: Option<u8>| {
        keys.iter()
            .filter(|k| k.family == family && k.bit_depth == bd)
            .count()
    };
    // dav1d bağlam alanları (bit derinliği başına).
    assert_eq!(
        count("mc", Some(8)),
        52,
        "mc[10]+mc_scaled[10]+mct[10]+mct_scaled[10]+12"
    );
    assert_eq!(
        count("ipred", Some(8)),
        22,
        "intra_pred[14]+cfl_ac[3]+cfl_pred[4]+pal_pred"
    );
    assert_eq!(count("loopfilter", Some(8)), 4);
    assert_eq!(count("cdef", Some(8)), 4);
    assert_eq!(count("looprestoration", Some(8)), 5);
    assert_eq!(count("filmgrain", Some(8)), 8);
    assert_eq!(count("resize", Some(8)), 1);
    assert_eq!(count("pal", None), 1);
    assert_eq!(count("refmvs", None), 3);
    assert_eq!(count("msac", None), 7);
    // itx: dav1d `dav1d_itx_dsp_init` hücreleri (`src/itx_tmpl.c:240-302`).
    let itx = count("itx", Some(8));
    assert_eq!(itx, 8 * 16 + 12 + 5 * 2 + 5 + 1, "itx hücre sayısı");
    for k in &keys {
        assert_eq!(
            k.selection == Selection::CompileTime,
            k.family == "msac",
            "{}",
            k.id()
        );
    }
}

/// Her bit derinliği tablosu kendi monomorflarını taşır (DOG-05).
///
/// P30: işlev adresi karşılaştırılmaz (optimize derlemede aynı işlevin adresi
/// kod birimine göre değişebilir). Tablo kaydı (`monomorph_bpc`) her ailenin
/// `X::scalar::<BPC>()` ile kurulduğu `BPC`'dir ve bit derinliğiyle uyumu
/// derleme zamanında denetlenir; itx ve CDEF yuvaları ayrıca **davranışla**
/// denetlenir: yuva çıktısı o bit derinliğinin skaler gövdesiyle aynı, başka
/// bit derinliğinin gövdesinden farklıdır.
#[test]
fn bit_derinligi_tablolari_kendi_monomorflarini_tasir() {
    use crate::dsp::cdef::{CdefBitDepth, CdefBlockSize, CdefEdgeFlags, CdefFilterParams};
    use crate::dsp::itx::{ItxFn, TxSize, TxType};
    use crate::dsp::scalar;
    let dsp = DspDispatch::select(CpuFlags::NONE);
    let t8 = dsp.for_bit_depth(BitDepth::Eight);
    let t10 = dsp.for_bit_depth(BitDepth::Ten);
    let t12 = dsp.for_bit_depth(BitDepth::Twelve);
    assert_eq!(t8.bit_depth(), BitDepth::Eight);
    assert_eq!(t10.bit_depth(), BitDepth::Ten);
    assert_eq!(t12.bit_depth(), BitDepth::Twelve);
    assert_eq!(
        [t8.monomorph_bpc(), t10.monomorph_bpc(), t12.monomorph_bpc()],
        [8, 10, 12]
    );

    // itx (DC yolu, 4×4 DCT_DCT): piksel 200 (8 bit) / 1000 (16 bpc) + DC 2000
    // (+63): 10 bit 1023'te kırpar, 12 bit kırpmaz.
    let itx = |f: ItxFn, bd: BitDepth| -> Vec<u8> {
        let bpp = if bd == BitDepth::Eight { 1 } else { 2 };
        let mut dst = vec![0u8; 16 * bpp];
        for px in dst.chunks_exact_mut(bpp) {
            if bpp == 1 {
                px[0] = 200;
            } else {
                px.copy_from_slice(&1000u16.to_le_bytes());
            }
        }
        let mut coeff = [0i32; 32 * 32];
        coeff[0] = 2000;
        f(
            &mut dst,
            4 * bpp,
            &mut coeff,
            0,
            TxSize::Tx4x4,
            TxType::DctDct,
            bd,
        );
        dst
    };
    for (t, bd) in [
        (t8, BitDepth::Eight),
        (t10, BitDepth::Ten),
        (t12, BitDepth::Twelve),
    ] {
        let beklenen = match bd {
            BitDepth::Eight => itx(scalar::itx::inverse_transform_add::<8>, bd),
            BitDepth::Ten => itx(scalar::itx::inverse_transform_add::<10>, bd),
            BitDepth::Twelve => itx(scalar::itx::inverse_transform_add::<12>, bd),
        };
        assert_eq!(
            itx(
                t.itx().itxfm_add[TxSize::Tx4x4 as usize][TxType::DctDct as usize],
                bd
            ),
            beklenen,
            "{bd:?}"
        );
    }
    assert_ne!(
        itx(scalar::itx::inverse_transform_add::<10>, BitDepth::Ten),
        itx(scalar::itx::inverse_transform_add::<12>, BitDepth::Twelve),
        "ayırt edici girdi: 10 ve 12 bit gövdeleri farklı çıktı vermeli"
    );

    // CDEF `dir` (10 bit) ve `fb` (12 bit): deterministik 16 bpc blok.
    let mut img = [0u8; 8 * 16];
    for (i, px) in img.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let v = ((i * 2_654_435_761) >> 7) as u16 & 0x3ff;
        px.copy_from_slice(&v.to_le_bytes());
    }
    let dir10 = (t10.cdef().dir)(&img, 16, CdefBitDepth::B10);
    assert_eq!(
        dir10,
        scalar::cdef::find_dir::<10>(&img, 16, CdefBitDepth::B10)
    );
    assert_ne!(
        scalar::cdef::find_dir::<10>(&img, 16, CdefBitDepth::B10),
        scalar::cdef::find_dir::<12>(&img, 16, CdefBitDepth::B12),
        "ayırt edici girdi: 10 ve 12 bit dir farklı olmalı"
    );
    // Pürüzsüz zemin + küçük desen: `constrain` sıfırdan farklı katkı verir.
    let fb = |f: crate::dsp::cdef::CdefFilterFn, bd: CdefBitDepth| -> Vec<u8> {
        let stride = 2 * 12;
        let mut dst = vec![0u8; 8 * stride];
        for (i, px) in dst.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let (x, y) = (i % 12, i / 12);
            let v = 2000 + ((x * 3 + y * 5) % 11) as u16 - 5;
            px.copy_from_slice(&v.to_le_bytes());
        }
        let params = CdefFilterParams {
            bit_depth: bd,
            size: CdefBlockSize::W8H8,
            pri_strength: 5 << bd.min_8(),
            sec_strength: 2 << bd.min_8(),
            dir: 3,
            damping: 4 + bd.min_8() as u8,
            edges: CdefEdgeFlags::NONE,
        };
        f(&mut dst, stride, &[], &[], &[], params);
        dst
    };
    for i in 0..3 {
        assert_eq!(
            fb(t12.cdef().fb[i], CdefBitDepth::B12),
            fb(scalar::cdef::filter::<12>, CdefBitDepth::B12),
            "fb[{i}] 12 bit"
        );
    }
    assert_ne!(
        fb(scalar::cdef::filter::<12>, CdefBitDepth::B12),
        fb(scalar::cdef::filter::<10>, CdefBitDepth::B10),
        "ayırt edici girdi: 10 ve 12 bit fb farklı olmalı"
    );
}

/// Her ISA adayının bir yuvası vardır ve aday 8 bit tablosundadır.
// P38 ara karar 3: x86 aday listesine (`ISA_CANDIDATES`) dayanır; x86_64 dışında
// liste boştur (test anlamsızlaşır ya da sayı tutmaz).
#[cfg(target_arch = "x86_64")]
#[test]
fn isa_adaylarinin_yuvasi_var() {
    let keys = slot_keys();
    for c in ISA_CANDIDATES {
        assert!(
            keys.iter().any(|k| k.family == c.family
                && k.field == c.field
                && (k.bit_depth == Some(c.bit_depth)
                    || (k.bit_depth.is_none() && c.bit_depth == 0))),
            "{}/{}: yuvası yok",
            c.family,
            c.field
        );
        assert!(!c.decision.is_empty());
    }
}

/// P35: düzey azalan kurulum ([`super::kurulum_sirasi`]) eski düzenin
/// (düzey artan yürüyüş; kurulan aday aynı aile, alan ve bit derinliğindeki
/// adayların bitini siler) kurduğu bit kümesini verir — tespit, düzeyler ve
/// her adayın bayrakları ile tümleyeni.
// P38 ara karar 3: x86 aday listesine (`ISA_CANDIDATES`) dayanır; x86_64 dışında
// liste boştur (test anlamsızlaşır ya da sayı tutmaz).
#[cfg(target_arch = "x86_64")]
#[test]
fn kurulum_eski_duzey_artan_duzenle_ayni() {
    let detected = CpuFlags::detect();
    let mut kumeler = vec![CpuFlags::NONE, CpuFlags::ALL_KNOWN, detected];
    for level in crate::cpu::CpuLevel::ALL {
        kumeler.push(detected.intersection(level.allowed()));
    }
    for c in ISA_CANDIDATES {
        kumeler.push(c.requires);
        kumeler.push(c.requires.complement());
    }
    for flags in kumeler {
        let dsp = DspDispatch::select(flags);
        let mut eski = [0u64; super::KURULU_KELIME];
        for rank in 1..=5 {
            for (index, c) in ISA_CANDIDATES.iter().enumerate() {
                if !c.default
                    || super::isa_rank(c.implementation) != rank
                    || dsp.candidate_entry(c).is_none()
                {
                    continue;
                }
                for (other, o) in ISA_CANDIDATES.iter().enumerate() {
                    if o.family == c.family && o.field == c.field && o.bit_depth == c.bit_depth {
                        eski[other / 64] &= !(1u64 << (other % 64));
                    }
                }
                eski[index / 64] |= 1u64 << (index % 64);
            }
        }
        assert_eq!(dsp.installed, eski, "{flags:?}");
    }
}

/// ADR-0003 §5.2: ISA adayları `select` ile tutarlıdır — P30: bir yuvanın
/// her düzeydeki adayı için, yalnız adayın bayraklarıyla kurulumda yuvayı
/// (varsayılan ve tespit edilmişse) o aday tutar; bayrakları kapalıyken
/// tutmaz. Adayı olmayan yuva hiçbir bayrakla atanmaz.
#[test]
fn yuva_adaylari_select_ile_tutarli() {
    let detected = CpuFlags::detect();
    let all = DspDispatch::select(CpuFlags::ALL_KNOWN).slot_states();
    for (i, state) in DspDispatch::select(CpuFlags::NONE)
        .slot_states()
        .iter()
        .enumerate()
    {
        if state.candidates.is_empty() {
            assert!(
                !all[i].is_assigned(),
                "{}: adaysız yuva atandı",
                state.key.id()
            );
            continue;
        }
        for c in &state.candidates {
            let on = &DspDispatch::select(c.requires).slot_states()[i];
            let off = &DspDispatch::select(c.requires.complement()).slot_states()[i];
            if c.default && detected.contains(c.requires) {
                assert_eq!(on.implementation, c.implementation, "{}", state.key.id());
            } else {
                assert_ne!(on.implementation, c.implementation, "{}", state.key.id());
            }
            assert_ne!(off.implementation, c.implementation, "{}", state.key.id());
        }
    }
}

/// F3a: maskeyle kapatılan bir ISA'nın yuvası **hiçbir koşulda** atanmaz.
#[test]
fn maskeyle_kapatilan_isa_yuvasi_atanmaz() {
    use crate::cpu::CpuLevel;
    for subset in 0..64u32 {
        let detected = CpuFlags::from_dav1d_x86_bits(subset);
        for level in CpuLevel::ALL {
            let effective = detected.difference(level.cpu_mask());
            let dsp = DspDispatch::select(effective);
            for state in dsp.slot_states() {
                if state.is_assigned() {
                    let c = state.candidate.expect("atanmış yuvanın adayı olmalı");
                    assert!(
                        level.allowed().contains(c.requires),
                        "{}: {} düzeyinde {} atandı",
                        state.key.id(),
                        level.name(),
                        c.implementation
                    );
                }
            }
            if level == CpuLevel::None {
                assert_eq!(dsp.assigned_count(), 0, "none düzeyi: {detected}");
            }
        }
    }
}

/// F3b/F3c/P30: `select` belirteçleri yalnız tespit + maske izin verdiğinde
/// üretir; her düzeyde yuvaların seçim kaydı (`slot_implementation`)
/// `planned_slot_states` ile aynıdır. Yuva kimliği işlev adresiyle değil,
/// kurulumun seçim kaydıyla denetlenir (P30, release profili).
#[test]
fn belirtecli_kurulum_tespit_ve_maskeye_baglidir() {
    use crate::cpu::CpuLevel;
    let detected = CpuFlags::detect();
    for level in CpuLevel::ALL {
        let flags = detected.intersection(level.allowed());
        let dsp = DspDispatch::select(flags);
        let izin = |f: CpuFlags| detected.contains(f) && flags.contains(f);
        assert_eq!(
            dsp.avx2_token().is_some(),
            izin(CpuFlags::AVX2),
            "{}",
            level.name()
        );
        assert_eq!(
            dsp.ssse3_token().is_some(),
            izin(CpuFlags::SSSE3),
            "{}",
            level.name()
        );
        assert_eq!(
            dsp.sse41_token().is_some(),
            izin(CpuFlags::SSE4_1) && izin(CpuFlags::SSSE3),
            "{}",
            level.name()
        );
        assert_eq!(
            dsp.avx512icl_token().is_some(),
            flags.contains(CpuFlags::AVX512ICL),
            "{}",
            level.name()
        );
        let planned = DspDispatch::planned_slot_states(flags);
        assert_eq!(dsp.slot_states(), planned, "{}", level.name());
        for state in &planned {
            let Some(bd) = state.key.bit_depth else {
                if state.key.selection == Selection::Runtime {
                    assert_eq!(
                        dsp.independent_slot_implementation(state.key.family, &state.key.field),
                        state.implementation,
                        "{} {}",
                        level.name(),
                        state.key.id()
                    );
                }
                continue;
            };
            let bd = match bd {
                8 => BitDepth::Eight,
                10 => BitDepth::Ten,
                _ => BitDepth::Twelve,
            };
            assert_eq!(
                dsp.slot_implementation(bd, state.key.family, &state.key.field),
                state.implementation,
                "{} {}",
                level.name(),
                state.key.id()
            );
        }
        // Yuvayı tutan aday en yüksek düzeydeki uygun varsayılan adaydır.
        for state in dsp.slot_states() {
            let en_yuksek = state
                .candidates
                .iter()
                .filter(|c| c.default && c.kurulabilir(flags))
                .map(|c| super::isa_rank(c.implementation))
                .max()
                .unwrap_or(0);
            assert_eq!(
                super::isa_rank(state.implementation),
                en_yuksek,
                "{} {}",
                level.name(),
                state.key.id()
            );
        }
        if dsp.sse2_token().is_none() && dsp.avx2_token().is_none() && dsp.ssse3_token().is_none() {
            assert_eq!(dsp.assigned_count(), 0, "{}", level.name());
        }
        // P34: MC 10/12 bit AVX2 belirteci varsa AVX2'dir; P35: yoksa SSSE3
        // belirteci varsa SSSE3, o da yoksa skaler; P36: AVX512ICL belirteci
        // varsa AVX512ICL.
        let mc16 = if dsp.avx512icl_token().is_some() {
            "avx512icl"
        } else if dsp.avx2_token().is_some() {
            "avx2"
        } else if dsp.ssse3_token().is_some() {
            "ssse3"
        } else {
            "skaler"
        };
        for bd in [BitDepth::Ten, BitDepth::Twelve] {
            assert_eq!(
                dsp.slot_implementation(bd, "mc", "mc[FILTER_2D_8TAP_REGULAR]"),
                mc16,
                "{}",
                level.name()
            );
        }
    }
    // Sahte küme: tespitte olmayan bayrak belirteç üretmez.
    let fake = DspDispatch::select(CpuFlags::X86_DAV1D);
    assert_eq!(
        fake.avx2_token().is_some(),
        detected.contains(CpuFlags::AVX2)
    );
    // P36: `SLOW_GATHER` yetenek değil kısıttır; verilen küme içerdiğinde
    // tespit edilmemiş olsa da uyulur (yalnız aday kaldırır).
    // P52 (Apple M4 canlı koşusu): `x86_64` dışında tespit kümesi (`NEON |
    // DOTPROD`) `X86_DAV1D` ile kesişmez; beklenen plan verilen kümenin
    // tespitle kesişimidir (`x86_64` üzerinde `detected ⊆ X86_DAV1D`, aynı sonuç).
    assert_eq!(
        fake.slot_states(),
        DspDispatch::planned_slot_states(
            CpuFlags::X86_DAV1D.intersection(detected) | CpuFlags::SLOW_GATHER
        )
    );
}

/// Zorlanmış skaler kipte AVX2 gövdeleri çağrılamaz: üretim kaynağında x86
/// çekirdeklerine tek giriş yolu `DspDispatch::select`'tir (`install_avx2`;
/// yorumlar ve test modülleri dışında başka başvuru yok).
#[test]
fn x86_govdelerine_tek_giris_select() {
    use std::fs;
    use std::path::{Path, PathBuf};
    fn files(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("dizin okunmalı") {
            let path = entry.expect("girdi").path();
            if path.is_dir() {
                files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    fn code_lines(text: &str) -> Vec<&str> {
        let body = text.split("#[cfg(test)]\nmod tests").next().unwrap_or(text);
        body.lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .filter(|l| l.contains("x86::cdef::") || l.contains("x86::mc::"))
            .collect()
    }
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut all = Vec::new();
    files(&src, &mut all);
    let mut seen = 0;
    for file in all {
        let rel = file
            .strip_prefix(&src)
            .expect("src altında")
            .to_string_lossy()
            .replace('\\', "/");
        if rel.starts_with("dsp/x86/") || rel == "dsp/tests.rs" {
            continue;
        }
        let text = fs::read_to_string(&file).expect("kaynak okunmalı");
        let lines = code_lines(&text);
        if rel == "dsp/mod.rs" {
            // P30: tek giriş `DspDispatch::candidate_entry`'dir (kurulum ve
            // ölçüm onu çağırır); x86 başvurusu yalnız o işlevin gövdesinde.
            let govde = text
                .split("pub fn candidate_entry(")
                .nth(1)
                .and_then(|r| r.split("\n    }\n").next())
                .expect("candidate_entry gövdesi");
            let disari = text.replace(govde, "");
            seen = govde.matches("x86::").count();
            assert!(
                code_lines(&disari).is_empty() && !disari.contains("use x86::"),
                "candidate_entry dışı x86 başvurusu"
            );
        } else {
            assert!(
                lines.is_empty(),
                "{rel}: x86 çekirdeğine doğrudan başvuru: {lines:?}"
            );
        }
    }
    assert!(
        seen >= 2,
        "candidate_entry x86 başvurusu bulunamadı ({seen})"
    );
}

/// Test kurucusu üretim tablosundan ayrıdır: yalnız `#[cfg(test)]`. P30:
/// değişiklik davranışla gösterilir — 8 bit tablosunun `dir` yuvasına 10 bit
/// gövdesi yazılınca yuva 10 bit girdisinde 10 bit sonucunu verir.
#[test]
fn test_kurucusu_tabloyu_degistirir() {
    use crate::dsp::cdef::CdefBitDepth;
    let dsp = DspDispatch::scalar_for_test(|t| {
        let ten = t[1].cdef().dir;
        DspDispatch::test_cdef_mut(&mut t[0]).dir = ten;
    });
    let mut img = [0u8; 8 * 16];
    for (i, px) in img.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        px.copy_from_slice(&(((i * 37) % 1024) as u16).to_le_bytes());
    }
    let a = (dsp.for_bit_depth(BitDepth::Eight).cdef().dir)(&img, 16, CdefBitDepth::B10);
    let b = (dsp.for_bit_depth(BitDepth::Ten).cdef().dir)(&img, 16, CdefBitDepth::B10);
    assert_eq!(a, b);
    let skaler = DspDispatch::select(CpuFlags::NONE);
    let c = (skaler.for_bit_depth(BitDepth::Eight).cdef().dir)(&img, 16, CdefBitDepth::B8);
    assert_ne!((a.dir, a.var), (c.dir, c.var), "ayırt edici girdi");
}

/// F3c: çözme hattı (`decode/`, `api/`) bütün DSP çağrılarını yuvalardan
/// yapar — test modülleri dışında skaler gövdeye (`dsp::scalar::`) ya da
/// skaler tabloya (`::scalar::<`, `scalar_for(`, `::SCALAR`) doğrudan
/// başvuru yoktur. Tablolar yalnız `DspDispatch` üzerinden alınır.
#[test]
fn cozme_hatti_yuvalardan_cagirir() {
    use std::fs;
    use std::path::Path;
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut hits = Vec::new();
    let mut files = 0;
    for dir in ["decode", "api"] {
        for entry in fs::read_dir(src.join(dir)).expect("dizin okunmalı") {
            let path = entry.expect("girdi").path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            files += 1;
            let text = fs::read_to_string(&path).expect("kaynak okunmalı");
            let body = text.split("#[cfg(test)]").next().unwrap_or(&text);
            for (n, line) in body.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                if ["dsp::scalar::", "::scalar::<", "scalar_for(", "::SCALAR"]
                    .iter()
                    .any(|p| code.contains(p))
                {
                    hits.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                }
            }
        }
    }
    assert!(files >= 10, "decode/ ve api/ taranmalı ({files} dosya)");
    assert!(
        hits.is_empty(),
        "yuvayı atlayan doğrudan çağrılar: {hits:#?}"
    );
}

/// P36: `SLOW_GATHER` bayrağında AVX512ICL `warp8x8`, `warp8x8t` ve `resize`
/// adaylarının girişi verilmez ve kurulmaz (dav1d `mc_init_x86`); öbür
/// AVX512ICL MC adayları etkilenmez. Bayrak kayıt tablosundan
/// ([`crate::cpu::quirks`]) gelir: tespit kümesine Zen 4 (aile 0x19) ve Zen 5
/// (aile 0x1A) kayıtları uygulanarak sınanır.
/// P39: dav1d `film_grain_dsp_init_x86` kuralı — AVX2 `generate_grain_*`
/// her zaman, `fgy_32x32xn`/`fguv_32x32xn[*]` yalnız `!SLOW_GATHER`'da kurulur.
/// Quirk uygulanmış bayrak kümeleri: AMD aile 0x17 ve 0x19 (`SLOW_GATHER`),
/// 0x1A (yok).
#[cfg(target_arch = "x86_64")]
#[test]
fn p39_yavas_toplamada_fg_avx2_fgy_fguv_kurulmaz() {
    use crate::cpu::quirks;
    let detected = CpuFlags::detect().difference(CpuFlags::SLOW_GATHER);
    let normal_f = quirks::apply(b"AuthenticAMD", 0x1a, 0x44, detected);
    assert!(!normal_f.contains(CpuFlags::SLOW_GATHER));
    let normal = DspDispatch::select(normal_f);
    let fg: Vec<_> = ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "filmgrain" && c.implementation == "avx2" && c.bit_depth == 8)
        .collect();
    assert_eq!(fg.len(), 8, "8 bpc AVX2 film grain adayı");
    for aile in [0x17u32, 0x19] {
        let yavas_f = quirks::apply(b"AuthenticAMD", aile, 0x31, detected);
        assert_eq!(
            yavas_f.contains(CpuFlags::SLOW_GATHER),
            detected.contains(CpuFlags::AVX2),
            "aile {aile:#x}"
        );
        let yavas = DspDispatch::select(yavas_f);
        let mut toplama = 0;
        for c in &fg {
            let gather = c.field == "fgy_32x32xn" || c.field.starts_with("fguv_32x32xn[");
            toplama += usize::from(gather);
            assert_eq!(c.yavas_toplamada_kurulmaz(), gather, "{}", c.field);
            assert_eq!(
                c.kurulabilir(CpuFlags::AVX2 | CpuFlags::SLOW_GATHER),
                !gather,
                "{}",
                c.field
            );
            assert!(c.kurulabilir(CpuFlags::AVX2), "{}", c.field);
            let avx2 = yavas.avx2_token().is_some();
            assert_eq!(
                yavas.candidate_entry(c).is_some(),
                avx2 && !(gather && yavas_f.contains(CpuFlags::SLOW_GATHER)),
                "{}",
                c.field
            );
            assert_eq!(
                normal.candidate_entry(c).is_some(),
                normal.avx2_token().is_some(),
                "{}",
                c.field
            );
            if gather {
                assert_ne!(
                    yavas.slot_implementation(BitDepth::Eight, "filmgrain", c.field),
                    "avx2",
                    "{}",
                    c.field
                );
                // P41: SLOW_GATHER etkinken AVX512ICL varsa `avx512icl`,
                // yalnız AVX2/SSSE3 varsa `ssse3` kurulur (`filmgrain.h:48-82`).
                if yavas.avx512icl_token().is_some() {
                    assert_eq!(
                        yavas.slot_implementation(BitDepth::Eight, "filmgrain", c.field),
                        "avx512icl",
                        "{}",
                        c.field
                    );
                } else if yavas.ssse3_token().is_some() {
                    assert_eq!(
                        yavas.slot_implementation(BitDepth::Eight, "filmgrain", c.field),
                        "ssse3",
                        "{}",
                        c.field
                    );
                }
            } else if yavas.avx2_token().is_some() {
                assert_eq!(
                    yavas.slot_implementation(BitDepth::Eight, "filmgrain", c.field),
                    "avx2",
                    "{}",
                    c.field
                );
            }
        }
        assert_eq!(toplama, 4);
    }
}

/// P41: 8 bpc SSSE3 film grain adayları (8 kayıt) `ssse3` ve `sse41`
/// düzeylerinde 8 yuvanın tamamını tutar; `AVX2 | SLOW_GATHER` kümesinde
/// `generate_grain_*` (4 yuva) `avx2`, `fgy`/`fguv` (4 yuva) `ssse3` olur.
#[cfg(target_arch = "x86_64")]
#[test]
fn p41_fg_ssse3_kurulum_ve_yavas_toplama_paritesi() {
    let fg_ssse3: Vec<_> = ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "filmgrain" && c.implementation == "ssse3" && c.bit_depth == 8)
        .collect();
    assert_eq!(fg_ssse3.len(), 8, "8 bpc SSSE3 film grain adayı");
    let detected = CpuFlags::detect();
    for duzey in [crate::cpu::CpuLevel::Ssse3, crate::cpu::CpuLevel::Sse41] {
        let d = DspDispatch::select(detected.intersection(duzey.allowed()));
        if d.ssse3_token().is_none() {
            continue;
        }
        for c in &fg_ssse3 {
            assert!(d.candidate_entry(c).is_some(), "{}", c.field);
            assert_eq!(
                d.slot_implementation(BitDepth::Eight, "filmgrain", c.field),
                "ssse3",
                "{} {}",
                duzey.name(),
                c.field
            );
        }
    }
    let yavas_avx2 = detected
        .intersection(crate::cpu::CpuLevel::Avx2.allowed())
        .union(CpuFlags::SLOW_GATHER);
    let d = DspDispatch::select(yavas_avx2);
    if d.avx2_token().is_some() && d.ssse3_token().is_some() {
        for c in &fg_ssse3 {
            let gather = c.field == "fgy_32x32xn" || c.field.starts_with("fguv_32x32xn[");
            let beklenen = if gather { "ssse3" } else { "avx2" };
            assert_eq!(
                d.slot_implementation(BitDepth::Eight, "filmgrain", c.field),
                beklenen,
                "{}",
                c.field
            );
        }
    }
}

/// P41: 8 bpc AVX512ICL film grain adayları (4 kayıt: `fgy_32x32xn` ve
/// `fguv_32x32xn[420/422/444]`) `vpermi2b`/`vpermt2b` tabanlı olduğu için
/// `SLOW_GATHER` bayrağı açıkken de (`0x3f`) kapalıyken de (`0x1f`) kurulur
/// (`dav1d` `src/x86/filmgrain.h:74-82`); `generate_grain_*` (4 yuva) ise
/// AVX2'de kalır. `0x2f` (`AVX2 | SLOW_GATHER`) kümesinde `fgy`/`fguv`
/// `ssse3`, `0x0f` (`AVX2`) kümesinde ise 8 yuvanın tamamı `avx2` olur.
#[cfg(target_arch = "x86_64")]
#[test]
fn p41_fg_avx512_kurulum_ve_yavas_toplama_paritesi() {
    let fg_avx512: Vec<_> = ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "filmgrain" && c.implementation == "avx512icl" && c.bit_depth == 8)
        .collect();
    assert_eq!(fg_avx512.len(), 4, "8 bpc AVX512ICL film grain adayı");
    for c in &fg_avx512 {
        assert!(!c.yavas_toplamada_kurulmaz(), "{}", c.field);
        assert!(
            c.kurulabilir(CpuFlags::AVX512ICL | CpuFlags::SLOW_GATHER),
            "{}",
            c.field
        );
    }
    let fg_alanlari: Vec<_> = ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "filmgrain" && c.implementation == "ssse3" && c.bit_depth == 8)
        .map(|c| c.field)
        .collect();
    assert_eq!(fg_alanlari.len(), 8);

    // 4 dav1d x86 bayrak kümesi: 0x3f (AVX512ICL|SLOW_GATHER), 0x1f (AVX512ICL),
    // 0x2f (AVX2|SLOW_GATHER), 0x0f (AVX2).
    for (bits, beklenen_gen, beklenen_uyg) in [
        (0x3fu32, "avx2", "avx512icl"),
        (0x1fu32, "avx2", "avx512icl"),
        (0x2fu32, "avx2", "ssse3"),
        (0x0fu32, "avx2", "avx2"),
    ] {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        for alan in &fg_alanlari {
            let uyg = *alan == "fgy_32x32xn" || alan.starts_with("fguv_32x32xn[");
            let beklenen = if uyg { beklenen_uyg } else { beklenen_gen };
            let st = planned
                .iter()
                .find(|s| {
                    s.key.family == "filmgrain"
                        && s.key.field == *alan
                        && s.key.bit_depth == Some(8)
                })
                .expect("filmgrain 8 bpc yuvası");
            assert_eq!(st.implementation, beklenen, "bits={bits:#x} {alan}");
        }
        let d = DspDispatch::select(flags);
        let icl_ok = !flags.contains(CpuFlags::AVX512ICL) || d.avx512icl_token().is_some();
        let avx2_ok = !flags.contains(CpuFlags::AVX2) || d.avx2_token().is_some();
        let ssse3_ok = !flags.contains(CpuFlags::SSSE3) || d.ssse3_token().is_some();
        if icl_ok && avx2_ok && ssse3_ok {
            for alan in &fg_alanlari {
                let uyg = *alan == "fgy_32x32xn" || alan.starts_with("fguv_32x32xn[");
                let beklenen = if uyg { beklenen_uyg } else { beklenen_gen };
                assert_eq!(
                    d.slot_implementation(BitDepth::Eight, "filmgrain", alan),
                    beklenen,
                    "canlı bits={bits:#x} {alan}"
                );
            }
        }
    }
}

// P38 ara karar 3: x86 aday listesine (`ISA_CANDIDATES`) dayanır; x86_64 dışında
// liste boştur (test anlamsızlaşır ya da sayı tutmaz).
#[cfg(target_arch = "x86_64")]
#[test]
fn p36_yavas_toplamada_avx512_warp_resize_kurulmaz() {
    use crate::cpu::quirks;
    let detected = CpuFlags::detect().difference(CpuFlags::SLOW_GATHER);
    let zen4 = quirks::apply(b"AuthenticAMD", 0x19, 0x61, detected);
    let zen5 = quirks::apply(b"AuthenticAMD", 0x1a, 0x44, detected);
    assert_eq!(
        zen4.contains(CpuFlags::SLOW_GATHER),
        detected.contains(CpuFlags::AVX2)
    );
    assert!(!zen5.contains(CpuFlags::SLOW_GATHER));
    let normal = DspDispatch::select(zen5);
    let yavas = DspDispatch::select(zen4);
    let mut engellenen = 0;
    for c in ISA_CANDIDATES
        .iter()
        .filter(|c| c.implementation == "avx512icl")
    {
        if c.yavas_toplamada_kurulmaz() {
            engellenen += 1;
            assert!(
                yavas.candidate_entry(c).is_none(),
                "{} {}",
                c.field,
                c.bit_depth
            );
            assert!(!c.kurulabilir(CpuFlags::AVX512ICL | CpuFlags::SLOW_GATHER));
        } else if c.family != "cdef" {
            assert_eq!(
                yavas.candidate_entry(c).is_some(),
                normal.candidate_entry(c).is_some(),
                "{} {}",
                c.field,
                c.bit_depth
            );
        }
        assert_eq!(
            normal.candidate_entry(c).is_some(),
            normal.avx512icl_token().is_some(),
            "{} {}",
            c.field,
            c.bit_depth
        );
    }
    // 8, 10, 12 bit × (warp8x8, warp8x8t, resize) = 9
    // + 10, 12 bit × (fgy, fguv[0..2]) = 8
    // + 8, 10, 12 bit × (loop_filter_sb[0][0], loop_filter_sb[1][0]) = 6 -> 23.
    assert_eq!(engellenen, 23);
    for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
        for (aile, alan) in [
            ("mc", "warp8x8"),
            ("mc", "warp8x8t"),
            ("resize", "resize"),
            ("loopfilter", "loop_filter_sb[0][0]"),
            ("loopfilter", "loop_filter_sb[1][0]"),
        ] {
            assert_ne!(yavas.slot_implementation(bd, aile, alan), "avx512icl");
        }
    }
    for bd in [BitDepth::Ten, BitDepth::Twelve] {
        for alan in [
            "fgy_32x32xn",
            "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
            "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
            "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        ] {
            assert_ne!(
                yavas.slot_implementation(bd, "filmgrain", alan),
                "avx512icl"
            );
        }
    }
}

/// P42 Adım 1: 16 bpc (10/12 bit) SSSE3 film grain adayları (16 kayıt: 8 × 2 bit
/// derinliği) `ssse3` ve `sse41` düzeylerinde 16 yuvanın tamamını tutar.
#[cfg(target_arch = "x86_64")]
#[test]
fn p42_fg16_ssse3_kurulum_paritesi() {
    let fg16_ssse3: Vec<_> = ISA_CANDIDATES
        .iter()
        .filter(|c| {
            c.family == "filmgrain"
                && c.implementation == "ssse3"
                && (c.bit_depth == 10 || c.bit_depth == 12)
        })
        .collect();
    assert_eq!(fg16_ssse3.len(), 16, "10/12 bpc SSSE3 film grain adayı");
    let detected = CpuFlags::detect();
    for duzey in [crate::cpu::CpuLevel::Ssse3, crate::cpu::CpuLevel::Sse41] {
        let d = DspDispatch::select(detected.intersection(duzey.allowed()));
        if d.ssse3_token().is_none() {
            continue;
        }
        for c in &fg16_ssse3 {
            assert!(
                d.candidate_entry(c).is_some(),
                "{} {}",
                c.field,
                c.bit_depth
            );
            let bd = if c.bit_depth == 10 {
                BitDepth::Ten
            } else {
                BitDepth::Twelve
            };
            assert_eq!(
                d.slot_implementation(bd, "filmgrain", c.field),
                "ssse3",
                "{} {} @{}",
                duzey.name(),
                c.field,
                c.bit_depth
            );
        }
    }
}

/// P42 Adım 2: 16 bpc (10/12 bit) AVX2 film grain adayları (16 kayıt: 8 × 2 bit
/// derinliği) ve `SLOW_GATHER` paritesi (`dav1d` `src/x86/filmgrain.h:67-80`):
/// - `0x0f` (`avx2`, `!SLOW_GATHER`) ve (Adım 3'e kadar) `0x1f` (`avx512icl`, `!SLOW_GATHER`)
///   altında 16 yuvanın tamamı `avx2` olur.
/// - `0x2f` (`AVX2 | SLOW_GATHER`) ve Zen 4 `0x3f` (`AVX512ICL | SLOW_GATHER`) altında
///   8 adet `generate_grain_*` yuvası (`B10` + `B12`) `avx2` olurken 8 adet `fgy`/`fguv`
///   yuvası (`B10` + `B12`) `ssse3`'te kalır.
#[cfg(target_arch = "x86_64")]
#[test]
fn p42_fg16_avx2_kurulum_ve_yavas_toplama_paritesi() {
    let fg16_avx2: Vec<_> = ISA_CANDIDATES
        .iter()
        .filter(|c| {
            c.family == "filmgrain"
                && c.implementation == "avx2"
                && (c.bit_depth == 10 || c.bit_depth == 12)
        })
        .collect();
    assert_eq!(fg16_avx2.len(), 16, "10/12 bpc AVX2 film grain adayı");
    let mut toplama = 0usize;
    for c in &fg16_avx2 {
        let gather = c.field == "fgy_32x32xn" || c.field.starts_with("fguv_32x32xn[");
        toplama += usize::from(gather);
        assert_eq!(
            c.yavas_toplamada_kurulmaz(),
            gather,
            "{} @{}",
            c.field,
            c.bit_depth
        );
        assert_eq!(
            c.kurulabilir(CpuFlags::AVX2 | CpuFlags::SLOW_GATHER),
            !gather,
            "{} @{}",
            c.field,
            c.bit_depth
        );
        assert!(
            c.kurulabilir(CpuFlags::AVX2),
            "{} @{}",
            c.field,
            c.bit_depth
        );
    }
    assert_eq!(toplama, 8);

    for (bits, beklenen_gen, beklenen_uyg) in [
        (0x3fu32, "avx2", "ssse3"),
        (0x1fu32, "avx2", "avx512icl"),
        (0x2fu32, "avx2", "ssse3"),
        (0x0fu32, "avx2", "avx2"),
    ] {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        for c in &fg16_avx2 {
            let uyg = c.field == "fgy_32x32xn" || c.field.starts_with("fguv_32x32xn[");
            let beklenen = if uyg { beklenen_uyg } else { beklenen_gen };
            let st = planned
                .iter()
                .find(|s| {
                    s.key.family == "filmgrain"
                        && s.key.field == c.field
                        && s.key.bit_depth == Some(c.bit_depth)
                })
                .expect("filmgrain 16 bpc yuvası");
            assert_eq!(
                st.implementation, beklenen,
                "planlanan bits={bits:#x} {} @{}",
                c.field, c.bit_depth
            );
        }
        let d = DspDispatch::select(flags);
        let icl_ok = !flags.contains(CpuFlags::AVX512ICL) || d.avx512icl_token().is_some();
        let avx2_ok = !flags.contains(CpuFlags::AVX2) || d.avx2_token().is_some();
        let ssse3_ok = !flags.contains(CpuFlags::SSSE3) || d.ssse3_token().is_some();
        if icl_ok && avx2_ok && ssse3_ok {
            for c in &fg16_avx2 {
                let uyg = c.field == "fgy_32x32xn" || c.field.starts_with("fguv_32x32xn[");
                let beklenen = if uyg { beklenen_uyg } else { beklenen_gen };
                let bd = if c.bit_depth == 10 {
                    BitDepth::Ten
                } else {
                    BitDepth::Twelve
                };
                assert_eq!(
                    d.slot_implementation(bd, "filmgrain", c.field),
                    beklenen,
                    "canlı bits={bits:#x} {} @{}",
                    c.field,
                    c.bit_depth
                );
            }
        }
    }
}

/// P42 Adım 3: 16 bpc (10/12 bit) AVX512ICL film grain adayları (8 kayıt: 4 × 2 bit
/// derinliği) ve 8/10/12 bpc tam `SLOW_GATHER` kurulum matrisi (`dav1d`
/// `src/x86/filmgrain.h:48-80`):
/// - `0x3f` (Zen 4 `AVX512ICL | SLOW_GATHER`): 8 bpc `fgy`/`fguv` `avx512icl`'de kalırken
///   10/12 bpc `fgy`/`fguv` hem `avx512icl` hem `avx2`'de engellenir ve `ssse3`'te kalır;
///   bütün `generate_grain_*` yuvaları `avx2`'dedir.
/// - `0x1f` (`AVX512ICL`, `!SLOW_GATHER`): 8, 10 ve 12 bpc `fgy`/`fguv` (12 yuva) `avx512icl`,
///   `generate_grain_*` (12 yuva) `avx2` olur.
/// - `0x2f` (`AVX2 | SLOW_GATHER`): 8, 10 ve 12 bpc `fgy`/`fguv` `ssse3`, `generate_grain_*` `avx2`.
/// - `0x0f` (`AVX2`, `!SLOW_GATHER`): 24 film grain yuvasının tamamı `avx2`.
#[cfg(target_arch = "x86_64")]
#[test]
fn p42_fg16_avx512_kurulum_ve_yavas_toplama_paritesi() {
    let fg16_avx512: Vec<_> = ISA_CANDIDATES
        .iter()
        .filter(|c| {
            c.family == "filmgrain"
                && c.implementation == "avx512icl"
                && (c.bit_depth == 10 || c.bit_depth == 12)
        })
        .collect();
    assert_eq!(fg16_avx512.len(), 8, "10/12 bpc AVX512ICL film grain adayı");
    for c in &fg16_avx512 {
        assert!(
            c.yavas_toplamada_kurulmaz(),
            "{} @{} SLOW_GATHER'da kurulmamalı",
            c.field,
            c.bit_depth
        );
        assert!(
            !c.kurulabilir(CpuFlags::AVX512ICL | CpuFlags::SLOW_GATHER),
            "{} @{}",
            c.field,
            c.bit_depth
        );
        assert!(
            c.kurulabilir(CpuFlags::AVX512ICL),
            "{} @{}",
            c.field,
            c.bit_depth
        );
    }

    let fg8_avx512: Vec<_> = ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "filmgrain" && c.implementation == "avx512icl" && c.bit_depth == 8)
        .collect();
    assert_eq!(fg8_avx512.len(), 4, "8 bpc AVX512ICL film grain adayı");
    for c in &fg8_avx512 {
        assert!(
            !c.yavas_toplamada_kurulmaz(),
            "8 bpc {} vpermi2b/vpermt2b tabanlıdır; SLOW_GATHER'da kurulmalı",
            c.field
        );
        assert!(c.kurulabilir(CpuFlags::AVX512ICL | CpuFlags::SLOW_GATHER));
    }

    for (bits, bek_gen, bek_uyg8, bek_uyg16) in [
        (0x3fu32, "avx2", "avx512icl", "ssse3"),
        (0x1fu32, "avx2", "avx512icl", "avx512icl"),
        (0x2fu32, "avx2", "ssse3", "ssse3"),
        (0x0fu32, "avx2", "avx2", "avx2"),
    ] {
        let flags = CpuFlags::from_dav1d_x86_bits(bits);
        let planned = DspDispatch::planned_slot_states(flags);
        let fg_yuvalari: Vec<_> = planned
            .iter()
            .filter(|s| s.key.family == "filmgrain")
            .collect();
        assert_eq!(fg_yuvalari.len(), 24, "3 bit derinliği × 8 yuva = 24");
        for s in fg_yuvalari {
            let uyg = s.key.field == "fgy_32x32xn" || s.key.field.starts_with("fguv_32x32xn[");
            let beklenen = match (uyg, s.key.bit_depth) {
                (false, _) => bek_gen,
                (true, Some(8)) => bek_uyg8,
                (true, Some(10 | 12)) => bek_uyg16,
                _ => unreachable!("beklenmeyen filmgrain yuvası"),
            };
            assert_eq!(
                s.implementation, beklenen,
                "planlanan bits={bits:#x} {} @{:?}",
                s.key.field, s.key.bit_depth
            );
        }
    }
}
