//! P30 (F4-CDEF c) — bütün CDEF ISA gövdeleri ↔ skaler ve dav1d oracle,
//! **bit-exact**.
//!
//! Gövdeler adaylardan alınır ([`common::cdef_fixture::isa_govdeleri`]:
//! `DspDispatch::candidate_entry`); varsayılan atanmayan (ölçümle yuvaya
//! girmeyen) gövde de sınanır. Düzeyler: SSSE3, SSE4.1 (`sse4`), AVX2,
//! AVX512ICL; her düzey yalnız dav1d'de kaydı olan işlevleri taşır (16 bpc
//! SSE4.1 yalnız `dir`, AVX512ICL yalnız `fb`).
//!
//! - oracle: küçük fixture (8, 10, 12 bit; dav1d C, `asm=OFF`);
//! - rastgele: her (işlev, bit derinliği, blok boyutu) hücresinde 10 000 blok;
//! - uç girdiler: `dir` sabit / dama / şerit / sekiz yönde çizgi desenleri (8,
//!   10, 12 bit, iki satır adımı); `fb` 16 kenar birleşimi × boş kenar dilimi
//!   (8 bpc) ve güç / sönüm / piksel uçları (10, 12 bit).
//!
//! Sentinel (dilim dışına 0 öğe) testleri `f3b_sentinel.rs`'tedir.

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::cdef::{CdefBitDepth, CdefDirFn, CdefEdgeFlags, CdefFilterFn};
use common::cdef_fixture::{
    IsaGovde, Rng, block_size, isa_govdeleri, params, random_case, run_dir, run_filter, scalar_dir,
    scalar_filter,
};
use common::on_kosul::{self, Neden};

/// Hücre başına rastgele blok sayısı.
const HUCRE_BLOK: usize = 10_000;

/// `bd` bit derinliğinin gövdeleri; eksik düzey ya da x86 dışı hedef ön koşul
/// ilkesiyle bildirilir.
fn govdeler(test: &str, bd: u8) -> Vec<IsaGovde> {
    let (var, yok) = isa_govdeleri(bd);
    if var.is_empty() && yok.is_empty() {
        on_kosul::eksik(test, Neden::X86Degil, "CDEF ISA adayı yok (x86_64 dışı)");
    }
    for ad in yok {
        on_kosul::eksik(
            test,
            Neden::isa(),
            &format!("{ad} düzeyi açık değil ({bd} bit)"),
        );
    }
    var
}

fn cdef_bd(bd: u8) -> CdefBitDepth {
    match bd {
        8 => CdefBitDepth::B8,
        10 => CdefBitDepth::B10,
        _ => CdefBitDepth::B12,
    }
}

/// Oracle fixture'ı (dav1d C): her ISA gövdesi kaydın beklenen zarfını üretir.
#[test]
fn p30_isa_govdeleri_oracle_fixtureinda_bit_ayni() {
    const T: &str = "cdef_isa::p30_isa_govdeleri_oracle_fixtureinda_bit_ayni";
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cdef");
    let input = std::fs::read(dir.join("cdef-small-input.bin")).expect("küçük girdi");
    let expect = std::fs::read(dir.join("cdef-small-expect.bin")).expect("küçük beklenen");
    let cases = common::cdef_fixture::read_cases(&input, &expect);
    let mut sayac = std::collections::BTreeMap::new();
    for bd in [8u8, 10, 12] {
        for g in govdeler(T, bd) {
            for case in cases.iter().filter(|c| c.bit_depth == bd) {
                let got = match (case.kind, g.dir, g.fb) {
                    (0, Some(f), _) => run_dir(case, f),
                    (1, _, Some(f)) => run_filter(case, f),
                    _ => continue,
                };
                assert_eq!(
                    got, case.expect,
                    "{} {bd} bit: oracle kaydı farklı (tür {}, boyut {}, kenar {:#06b})",
                    g.ad, case.kind, case.block_size, case.edges
                );
                *sayac.entry(g.ad).or_insert(0usize) += 1;
            }
        }
    }
    println!(
        "[P30 oracle] {} kayıt; gövde başına kayıt: {sayac:?}",
        cases.len()
    );
}

/// Rastgele bloklar: her ISA gövdesi skalerle bit-aynı (her hücre 10 000 blok).
#[test]
fn p30_isa_govdeleri_skaler_ile_bit_ayni() {
    const T: &str = "cdef_isa::p30_isa_govdeleri_skaler_ile_bit_ayni";
    let mut rng = Rng::new(0x9A30_CDEF_0000_0001);
    let mut toplam = 0usize;
    for bd in [8u8, 10, 12] {
        for g in govdeler(T, bd) {
            if let Some(f) = g.dir {
                for i in 0..HUCRE_BLOK {
                    let case = random_case(&mut rng, 0, bd, 0, (i % 4) as u8);
                    assert_eq!(
                        run_dir(&case, f),
                        run_dir(&case, scalar_dir(&case)),
                        "{} {bd} bit dir, blok {i}",
                        g.ad
                    );
                }
                toplam += HUCRE_BLOK;
            }
            if let Some(f) = g.fb {
                for size in 0u8..3 {
                    for i in 0..HUCRE_BLOK {
                        let case = random_case(&mut rng, 1, bd, size, (i % 4) as u8);
                        assert_eq!(
                            run_filter(&case, f),
                            run_filter(&case, scalar_filter(&case)),
                            "{} {bd} bit fb boyut {size}, blok {i} (kenar {:#06b}, pri {}, sec {}, dir {}, damping {})",
                            g.ad,
                            case.edges,
                            case.pri,
                            case.sec,
                            case.dir,
                            case.damping
                        );
                    }
                    toplam += HUCRE_BLOK;
                }
            }
        }
    }
    println!("[P30 rastgele] {toplam} blok ISA == skaler");
}

/// `dir` uç desenleri (8 × 8 piksel, `mx` = bit derinliğinin en büyüğü).
fn dir_desenleri(mx: u16) -> Vec<[u16; 64]> {
    let mut d: Vec<[u16; 64]> = Vec::new();
    for v in [0u16, 1, mx / 2, mx / 2 + 1, mx - 1, mx] {
        d.push([v; 64]);
    }
    for (a, b) in [(0u16, mx), (mx, 0), (mx / 16, mx - mx / 16)] {
        d.push(core::array::from_fn(|i| {
            if (i / 8 + i % 8) % 2 == 0 { a } else { b }
        }));
        d.push(core::array::from_fn(
            |i| if (i / 8) % 2 == 0 { a } else { b },
        ));
        d.push(core::array::from_fn(
            |i| if (i % 8) % 2 == 0 { a } else { b },
        ));
    }
    for (p, q) in [
        (1i32, 0i32),
        (0, 1),
        (1, 1),
        (1, -1),
        (2, 1),
        (1, 2),
        (2, -1),
        (1, -2),
    ] {
        for faz in 0..8 {
            d.push(core::array::from_fn(|i| {
                let (y, x) = ((i / 8) as i32, (i % 8) as i32);
                if (y * p + x * q).rem_euclid(8) == faz {
                    mx
                } else {
                    0
                }
            }));
        }
    }
    d
}

/// `dir` uç girdileri: 8, 10, 12 bit, iki satır adımı; (`dir`, `var`) skalerle aynı.
#[test]
fn p30_isa_dir_uc_girdiler_skaler_ile_ayni() {
    const T: &str = "cdef_isa::p30_isa_dir_uc_girdiler_skaler_ile_ayni";
    let mut n = 0usize;
    for bd in [8u8, 10, 12] {
        let cbd = cdef_bd(bd);
        let bpp = cbd.bytes();
        let skaler: CdefDirFn = av1oxide::dsp::cdef::CdefDsp::scalar_for(cbd).dir;
        for g in govdeler(T, bd) {
            let Some(f) = g.dir else {
                continue;
            };
            for d in dir_desenleri(((1u32 << bd) - 1) as u16) {
                for stride in [8 * bpp, 16 * bpp] {
                    let mut img = vec![0x5Au8; 7 * stride + 8 * bpp];
                    for y in 0..8 {
                        for x in 0..8 {
                            let o = y * stride + bpp * x;
                            if bpp == 1 {
                                img[o] = d[y * 8 + x] as u8;
                            } else {
                                img[o..o + 2].copy_from_slice(&d[y * 8 + x].to_ne_bytes());
                            }
                        }
                    }
                    let a = skaler(&img, stride, cbd);
                    let b = f(&img, stride, cbd);
                    assert_eq!(
                        (a.dir, a.var),
                        (b.dir, b.var),
                        "{} {bd} bit dir uç {d:?} stride={stride}",
                        g.ad
                    );
                    n += 1;
                }
            }
        }
    }
    println!("[P30 dir uç] {n} uç blok ISA == skaler");
}

/// `fb` kenar yolu (8 bpc): 3 boyut × 16 kenar birleşimi × 400 rastgele blok;
/// eksik kenar dilimi hem tam uzunluklu hem **boş** (çözme hattı biçimi).
#[test]
fn p30_isa_filter_kenar_yolu_skaler_ile_ayni() {
    const T: &str = "cdef_isa::p30_isa_filter_kenar_yolu_skaler_ile_ayni";
    let mut n = 0usize;
    for g in govdeler(T, 8) {
        let Some(f) = g.fb else {
            continue;
        };
        let mut rng = Rng::new(0x9A30_CDEF_0000_0002);
        for size_code in 0u8..3 {
            for e in 0u8..16 {
                for i in 0..400usize {
                    let mut case = random_case(&mut rng, 1, 8, size_code, (i % 4) as u8);
                    case.edges = e;
                    let a = run_filter(&case, scalar_filter(&case));
                    assert_eq!(
                        run_filter(&case, f),
                        a,
                        "{} fb {size_code} kenar {e:04b} blok {i}",
                        g.ad
                    );
                    let (w, h) = (block_size(size_code).w(), block_size(size_code).h());
                    let ef = CdefEdgeFlags::from_bits(e).expect("kenar");
                    let (dst0, left, top, bottom) = common::cdef_fixture::slices(&case);
                    let bos: &[u8] = &[];
                    let l = if ef.contains(CdefEdgeFlags::HAVE_LEFT) {
                        left
                    } else {
                        bos
                    };
                    let t = if ef.contains(CdefEdgeFlags::HAVE_TOP) {
                        top
                    } else {
                        bos
                    };
                    let bt = if ef.contains(CdefEdgeFlags::HAVE_BOTTOM) {
                        bottom
                    } else {
                        bos
                    };
                    let mut d = dst0.to_vec();
                    f(&mut d, case.stride, l, t, bt, params(&case));
                    for y in 0..h {
                        let r = y * case.stride..y * case.stride + w;
                        assert_eq!(
                            d[r],
                            a[y * w..y * w + w],
                            "{} boş dilim {size_code} {e:04b} {i}",
                            g.ad
                        );
                    }
                    n += 1;
                }
            }
        }
    }
    println!("[P30 fb kenar] {n} blok × 2 dilim biçimi ISA == skaler");
}

/// `fb` uç girdileri (10, 12 bit): 16 kenar birleşimi × güç uçları × sönüm
/// uçları × piksel desenleri (rastgele, {0, en büyük}, yürüyüş, en büyük).
#[test]
fn p30_isa_filter16_uc_girdiler_skaler_ile_ayni() {
    const T: &str = "cdef_isa::p30_isa_filter16_uc_girdiler_skaler_ile_ayni";
    let mut n = 0usize;
    for (bd, s) in [(10u8, 2u8), (12, 4)] {
        for g in govdeler(T, bd) {
            let Some(f): Option<CdefFilterFn> = g.fb else {
                continue;
            };
            let mut rng = Rng::new(0x9A30_CDEF_0000_0003);
            for size_code in 0u8..3 {
                for e in 0u8..16 {
                    for (pri, sec) in [(15u8, 4u8), (15, 0), (0, 4), (1, 1), (14, 2)] {
                        for damping in [3u8, 6] {
                            for desen in 0u8..4 {
                                let mut case = random_case(&mut rng, 1, bd, size_code, desen);
                                case.edges = e;
                                case.pri = pri << s;
                                case.sec = sec << s;
                                case.damping = damping + s;
                                case.dir = (rng.u64() % 8) as u8;
                                assert_eq!(
                                    run_filter(&case, f),
                                    run_filter(&case, scalar_filter(&case)),
                                    "{} fb {bd} bit boyut {size_code} kenar {e:04b} pri {pri} sec {sec} damp {damping} desen {desen}",
                                    g.ad
                                );
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    println!("[P30 fb16 uç] {n} blok ISA == skaler");
}

/// Zorlanmış düzeylerde (`ssse3`, `sse41`, `avx2`, `avx512icl`) her CDEF yuvası
/// o düzeyin kurulumundan koşar: seçim kaydı planlanan uygulamayla aynı
/// (düzeyin izin verdiği en yüksek varsayılan aday) ve yuvanın çıktısı
/// skalerle bit-aynı. Düzey bu makinede yoksa ön koşul ilkesiyle atlanır.
#[test]
fn p30_zorlanmis_duzeylerde_yuva_govdesi_skalerle_ayni() {
    use av1oxide::cpu::{CpuFlags, CpuLevel};
    use av1oxide::dsp::DspDispatch;
    const T: &str = "cdef_isa::p30_zorlanmis_duzeylerde_yuva_govdesi_skalerle_ayni";
    let detected = CpuFlags::detect();
    let mut rng = Rng::new(0x9A30_CDEF_0000_0004);
    let mut n = 0usize;
    for level in [
        CpuLevel::Ssse3,
        CpuLevel::Sse41,
        CpuLevel::Avx2,
        CpuLevel::Avx512Icl,
    ] {
        if !detected.contains(level.allowed().difference(CpuFlags::SLOW_GATHER)) {
            on_kosul::eksik(
                T,
                Neden::isa(),
                &format!("{} düzeyi bu makinede yok", level.name()),
            );
            continue;
        }
        let flags = detected.intersection(level.allowed());
        let dsp = DspDispatch::select(flags);
        let planli = DspDispatch::planned_slot_states(flags);
        for (bd, fbd) in [
            (8u8, av1oxide::frame::BitDepth::Eight),
            (10, av1oxide::frame::BitDepth::Ten),
            (12, av1oxide::frame::BitDepth::Twelve),
        ] {
            let t = *dsp.for_bit_depth(fbd).cdef();
            for (i, field) in ["dir", "fb[0]", "fb[1]", "fb[2]"].into_iter().enumerate() {
                let p = planli
                    .iter()
                    .find(|s| {
                        s.key.family == "cdef"
                            && s.key.field == field
                            && s.key.bit_depth == Some(bd)
                    })
                    .expect("planlı yuva");
                assert_eq!(
                    dsp.slot_implementation(fbd, "cdef", field),
                    p.implementation,
                    "{} {bd} bit {field}",
                    level.name()
                );
                for k in 0..256usize {
                    if i == 0 {
                        let case = random_case(&mut rng, 0, bd, 0, (k % 4) as u8);
                        assert_eq!(
                            run_dir(&case, t.dir),
                            run_dir(&case, scalar_dir(&case)),
                            "{} {bd} dir",
                            level.name()
                        );
                    } else {
                        let case = random_case(&mut rng, 1, bd, (i - 1) as u8, (k % 4) as u8);
                        assert_eq!(
                            run_filter(&case, t.fb[i - 1]),
                            run_filter(&case, scalar_filter(&case)),
                            "{} {bd} {field}",
                            level.name()
                        );
                    }
                    n += 1;
                }
            }
        }
    }
    println!("[P30 zorlanmış düzeyler] {n} blok yuvadan == skaler");
}
