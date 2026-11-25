//! P36 (F4-MC e) — 8 bpc AVX512ICL karışım gövdeleri: `blend`, `blend_v`, `blend_h`.
//!
//! - **Eşitlik:** her boyut (`blend` 4…32 kare; `blend_v` `w` 2…32 × `h`
//!   2…128; `blend_h` `w` 2…128 × `h` 2…32) × hedef satır adımı × piksel
//!   deseni (rastgele, 0/255, dama) × maske deseni (`blend`: rastgele 0…64,
//!   hepsi 0, hepsi 64, dama) için AVX512ICL çıktısı skaler gövdeyle bit-exact.
//! - **Sentinel:** `tmp` ve maske dilimleri tam `w·h` öğede biter; hedefte
//!   satır aralarına ve sona nöbetçi konur (`w` ötesine ve yazılan satırların
//!   ötesine yazma 0).
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; sözleşme dışı
//!   girdi skalere devreder ve sayılır.
//!
//! AVX512ICL yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::{BlendDirFn, BlendFn, McDsp, skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

const NOBETCI: u8 = 0xA5;

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
    fn bayt(&mut self) -> u8 {
        (self.u64() >> 32) as u8
    }
}

struct Yuvalar {
    blend: BlendFn,
    blend_v: BlendDirFn,
    blend_h: BlendDirFn,
}

fn avx512_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx512icl_token()
    {
        use av1oxide::dsp::x86::mc_karisim_avx512 as k;
        return Some(Yuvalar {
            blend: k::blend_fn(t),
            blend_v: k::blend_v_fn(t),
            blend_h: k::blend_h_fn(t),
        });
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P36] 8 bpc AVX512ICL karışım gövdeleri AVX512ICL ister",
    );
    None
}

/// Piksel deseni (0 rastgele, 1 hepsi 0 / 255, 2 dama).
fn pikseller(rng: &mut Rng, desen: u8, n: usize, ikinci: bool) -> Vec<u8> {
    (0..n)
        .map(|i| match desen {
            0 => rng.bayt(),
            1 => {
                if ikinci {
                    255
                } else {
                    0
                }
            }
            _ => {
                if (i + usize::from(ikinci)).is_multiple_of(2) {
                    0
                } else {
                    255
                }
            }
        })
        .collect()
}

fn maske(rng: &mut Rng, desen: u8, n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| match desen {
            0 => (rng.u64() % 65) as u8,
            1 => 0,
            2 => 64,
            _ => {
                if i.is_multiple_of(2) {
                    0
                } else {
                    64
                }
            }
        })
        .collect()
}

/// Hedef: `h` satır, adım `ds`; satır aralarında ve sonda nöbetçi, satır içi
/// desen. Döner: (dilim, satır içi ilk baytlar).
fn hedef(rng: &mut Rng, desen: u8, w: usize, h: usize, ds: usize) -> Vec<u8> {
    let mut d = vec![NOBETCI; (h - 1) * ds + w + 16];
    let p = pikseller(rng, desen, w * h, false);
    for y in 0..h {
        d[y * ds..y * ds + w].copy_from_slice(&p[y * w..(y + 1) * w]);
    }
    d
}

fn nobetci_saglam(d: &[u8], w: usize, h: usize, ds: usize) -> bool {
    (0..h).all(|y| {
        let son = if y + 1 < h { (y + 1) * ds } else { d.len() };
        d[y * ds + w..son].iter().all(|&b| b == NOBETCI)
    })
}

#[test]
fn p36_karisim_avx512_skalerle_bit_exact_ve_sentinel() {
    let Some(y) =
        avx512_yuvalari("mc_karisim_avx512::p36_karisim_avx512_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x0bad_5eed_1234_5678);
    let mut cagri = 0usize;
    let bd = BitDepth::Eight;
    // (işlev, w aralığı, h aralığı): 0 blend, 1 blend_v, 2 blend_h.
    let araliklar = [((4, 32), (4, 32)), ((2, 32), (2, 128)), ((2, 128), (2, 32))];
    for (islev, (wa, ha)) in araliklar.into_iter().enumerate() {
        let mut w = wa.0;
        while w <= wa.1 {
            let mut h = ha.0;
            while h <= ha.1 {
                let n = w * h;
                for &ds in &[w, w + 5, w + 32] {
                    for desen in 0..3u8 {
                        let tmp = pikseller(&mut rng, desen, n, true);
                        let d0 = hedef(&mut rng, desen, w, h, ds);
                        let maskeler: Vec<Vec<u8>> = if islev == 0 {
                            (0..4).map(|md| maske(&mut rng, md, n)).collect()
                        } else {
                            vec![Vec::new()]
                        };
                        for m in &maskeler {
                            let (mut a, mut b) = (d0.clone(), d0.clone());
                            match islev {
                                0 => {
                                    (y.blend)(&mut a, ds, &tmp, w, h, m, bd, 255);
                                    (sk.blend)(&mut b, ds, &tmp, w, h, m, bd, 255);
                                }
                                1 => {
                                    (y.blend_v)(&mut a, ds, &tmp, w, h, bd, 255);
                                    (sk.blend_v)(&mut b, ds, &tmp, w, h, bd, 255);
                                }
                                _ => {
                                    (y.blend_h)(&mut a, ds, &tmp, w, h, bd, 255);
                                    (sk.blend_h)(&mut b, ds, &tmp, w, h, bd, 255);
                                }
                            }
                            assert_eq!(a, b, "işlev {islev} {w}x{h} ds={ds} desen={desen}");
                            assert!(
                                nobetci_saglam(&a, w, h, ds),
                                "işlev {islev} {w}x{h} nöbetçi"
                            );
                            cagri += 1;
                        }
                    }
                }
                h *= 2;
            }
            w *= 2;
        }
    }
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrılar skalere devretmemeli"
    );
    println!("[P36] 8 bpc AVX512ICL karışım: {cagri} çağrı skalerle bit-exact (3 yuva)");
}

#[test]
fn p36_karisim_avx512_sozlesme_disi_skalere_devreder() {
    let Some(y) =
        avx512_yuvalari("mc_karisim_avx512::p36_karisim_avx512_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let mut rng = Rng(11);
    let bd = BitDepth::Eight;
    let d0 = skaler_devir_sayisi();
    // `blend` w = 2: sözleşme dışı (4…32).
    let (w, h, ds) = (2usize, 8usize, 2usize);
    let tmp = pikseller(&mut rng, 0, w * h, true);
    let m = maske(&mut rng, 0, w * h);
    let base = hedef(&mut rng, 0, w, h, ds);
    let (mut a, mut b) = (base.clone(), base);
    (y.blend)(&mut a, ds, &tmp, w, h, &m, bd, 255);
    (sk.blend)(&mut b, ds, &tmp, w, h, &m, bd, 255);
    assert_eq!(a, b);
    // `blend_h` h = 1: sözleşme dışı (2…32), skalere devreder (yazılan satır 0).
    let (w, h, ds) = (8usize, 1usize, 8usize);
    let tmp = pikseller(&mut rng, 0, w * h, true);
    let base = hedef(&mut rng, 0, w, h, ds);
    let (mut a, mut b) = (base.clone(), base);
    (y.blend_h)(&mut a, ds, &tmp, w, h, bd, 255);
    (sk.blend_h)(&mut b, ds, &tmp, w, h, bd, 255);
    assert_eq!(a, b);
    assert_eq!(
        skaler_devir_sayisi(),
        d0 + 2,
        "iki sözleşme dışı çağrı sayılmalı"
    );
}
