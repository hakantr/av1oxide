//! P34 (F4-MC c) — 16 bpc (10/12 bit) AVX2 bileşik gövdeleri: `avg`, `w_avg`,
//! `mask`, `w_mask[0..2]`.
//!
//! - **Eşitlik:** 10 ve 12 bitte her boyut (`w`, `h` 4…128; `w_mask` 8…128) ×
//!   hedef satır adımı (bayt) × girdi deseni (rastgele, en küçük, en büyük,
//!   en küçük/en büyük dama — 16 bpc `prep` aralığı, dav1d `mc_tmpl.c:41-48`;
//!   ek olarak bütün `i16` aralığı: gövde değer kümesi varsaymaz) × ağırlık
//!   (`w_avg` 0…16) / maske (rastgele 0…64, hepsi 0, hepsi 64, 0/64 dama,
//!   rastgele 0…255) / alt tür × `sign` için AVX2 çıktısı skaler gövdeyle
//!   bit-exact.
//! - **Sentinel:** `tmp` ve maske dilimleri tam `w·h` öğede biter (fazlası
//!   okunursa dilim denetimi panikler); hedefte satır aralarına (`2w` ötesi) ve
//!   sona, `w_mask` maskesinde bölge sonrasına nöbetçi konur.
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; sözleşme dışı
//!   girdi skalere devreder ve sayılır.
//!
//! AVX2 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::mc::{AvgFn, MaskFn, McDsp, WAvgFn, WMaskFn, skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

/// 16 bpc `prep` çıktı aralığı (dav1d `mc_tmpl.c:41-48`): 10 bit, 12 bit.
const PREP16_ARALIK: [(i16, i16); 2] = [(-28780, 28764), (-28794, 28791)];
const NOBETCI: u8 = 0xA5;

/// Devir sayacı süreç geneldir: sayaca bakan testler sırayla koşar.
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
    fn aralik(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.u64() >> 33) as i32 % (hi - lo + 1)
    }
}

struct Yuvalar {
    avg: AvgFn,
    w_avg: WAvgFn,
    mask: MaskFn,
    w_mask: [WMaskFn; 3],
}

fn avx2_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        use av1oxide::dsp::x86::mc16_bilesik as b;
        return Some(Yuvalar {
            avg: b::avg_fn(t),
            w_avg: b::w_avg_fn(t),
            mask: b::mask_fn(t),
            w_mask: [b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)],
        });
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P34] 16 bpc AVX2 bileşik gövdeleri AVX2 ister",
    );
    None
}

/// `n` öğelik `tmp` (desen: 0 rastgele, 1 en küçük, 2 en büyük, 3 dama —
/// `prep` aralığında; 4 rastgele, 5 dama — bütün `i16` aralığında).
fn tmp(rng: &mut Rng, aralik: (i16, i16), desen: u8, n: usize, kaydir: usize) -> Vec<i16> {
    let (lo, hi) = if desen >= 4 {
        (i16::MIN, i16::MAX)
    } else {
        aralik
    };
    (0..n)
        .map(|i| match desen {
            0 | 4 => rng.aralik(i32::from(lo), i32::from(hi)) as i16,
            1 => lo,
            2 => hi,
            _ => {
                if (i + kaydir).is_multiple_of(2) {
                    lo
                } else {
                    hi
                }
            }
        })
        .collect()
}

/// Bileşik maske (desen: 0 rastgele 0…64, 1 hepsi 0, 2 hepsi 64, 3 dama,
/// 4 rastgele 0…255 — skaler tanımı geçersiz maskede de karşılaştırılır).
fn maske(rng: &mut Rng, desen: u8, n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| match desen {
            0 => rng.aralik(0, 64) as u8,
            1 => 0,
            2 => 64,
            3 => {
                if i.is_multiple_of(2) {
                    0
                } else {
                    64
                }
            }
            _ => rng.aralik(0, 255) as u8,
        })
        .collect()
}

/// Hedef: `h` satır, adım `ds` bayt; satır arası (`2w` ötesi) ve son 32 bayt nöbetçi.
fn hedef(w: usize, h: usize, ds: usize) -> Vec<u8> {
    vec![NOBETCI; (h - 1) * ds + 2 * w + 32]
}

/// Nöbetçi denetimi: satır aralarında ve sonda değişiklik yok.
fn nobetci_saglam(d: &[u8], w: usize, h: usize, ds: usize) -> bool {
    (0..h).all(|y| {
        let son = if y + 1 < h { (y + 1) * ds } else { d.len() };
        d[y * ds + 2 * w..son].iter().all(|&b| b == NOBETCI)
    })
}

const BOYUTLAR: [usize; 6] = [4, 8, 16, 32, 64, 128];

#[test]
fn p34_bilesik16_avx2_skalerle_bit_exact_ve_sentinel() {
    let Some(y) =
        avx2_yuvalari("mc16_bilesik_avx2::p34_bilesik16_avx2_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c34);
    let mut cagri = 0usize;
    for (bi, (bd, max)) in [(BitDepth::Ten, 1023u16), (BitDepth::Twelve, 4095)]
        .into_iter()
        .enumerate()
    {
        let sk = McDsp::scalar_for(bd);
        let aralik = PREP16_ARALIK[bi];
        for &w in &BOYUTLAR {
            for &h in &BOYUTLAR {
                let n = w * h;
                let buyuk = n > 4096;
                let adimlar: &[usize] = if buyuk {
                    &[2 * w + 10]
                } else {
                    &[2 * w, 2 * w + 10, 2 * w + 64]
                };
                let desenler: &[u8] = if buyuk {
                    &[0, 3, 4]
                } else {
                    &[0, 1, 2, 3, 4, 5]
                };
                let agirliklar: Vec<i32> = if buyuk {
                    vec![0, 1, 7, 8, 15, 16]
                } else {
                    (0..=16).collect()
                };
                for &ds in adimlar {
                    for &desen in desenler {
                        let t1 = tmp(&mut rng, aralik, desen, n, 0);
                        let t2 = tmp(&mut rng, aralik, desen, n, 1);
                        let ctx = format!("{bd:?} {w}x{h} ds={ds} desen={desen}");
                        // avg
                        let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
                        (y.avg)(&mut a, ds, &t1, &t2, w, h, bd, max);
                        (sk.avg)(&mut b, ds, &t1, &t2, w, h, bd, max);
                        assert_eq!(a, b, "avg {ctx}");
                        assert!(nobetci_saglam(&a, w, h, ds), "avg nöbetçi {ctx}");
                        cagri += 1;
                        // w_avg
                        for &wt in &agirliklar {
                            let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
                            (y.w_avg)(&mut a, ds, &t1, &t2, w, h, wt, bd, max);
                            (sk.w_avg)(&mut b, ds, &t1, &t2, w, h, wt, bd, max);
                            assert_eq!(a, b, "w_avg {ctx} w={wt}");
                            assert!(nobetci_saglam(&a, w, h, ds), "w_avg nöbetçi {ctx}");
                            cagri += 1;
                        }
                        // mask
                        for md in 0..5u8 {
                            let m = maske(&mut rng, md, n);
                            let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
                            (y.mask)(&mut a, ds, &t1, &t2, w, h, &m, bd, max);
                            (sk.mask)(&mut b, ds, &t1, &t2, w, h, &m, bd, max);
                            assert_eq!(a, b, "mask {ctx} m={md}");
                            assert!(nobetci_saglam(&a, w, h, ds), "mask nöbetçi {ctx}");
                            cagri += 1;
                        }
                        // w_mask (w, h ≥ 8)
                        if w < 8 || h < 8 {
                            continue;
                        }
                        for alt in 0..3usize {
                            let (sh, sv) = (alt != 0, alt == 2);
                            let mn = (w >> usize::from(sh)) * (h >> usize::from(sv));
                            for sign in 0..2 {
                                let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
                                let (mut ma, mut mb) = (vec![0u8; mn + 8], vec![0u8; mn + 8]);
                                ma[mn..].fill(NOBETCI);
                                mb[mn..].fill(NOBETCI);
                                (y.w_mask[alt])(
                                    &mut a, ds, &t1, &t2, w, h, &mut ma, sign, sh, sv, bd, max,
                                );
                                (sk.w_mask[alt])(
                                    &mut b, ds, &t1, &t2, w, h, &mut mb, sign, sh, sv, bd, max,
                                );
                                let ad = format!("w_mask[{alt}] {ctx} sign={sign}");
                                assert_eq!(a, b, "{ad} piksel");
                                assert_eq!(ma, mb, "{ad} maske");
                                assert!(nobetci_saglam(&a, w, h, ds), "{ad} nöbetçi");
                                assert!(
                                    ma[mn..].iter().all(|&v| v == NOBETCI),
                                    "{ad} maske nöbetçi"
                                );
                                cagri += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrılar skalere devretmemeli"
    );
    println!("[P34] 16 bpc AVX2 bileşik: {cagri} çağrı skalerle bit-exact (6 yuva × 10/12 bit)");
}

#[test]
fn p34_bilesik16_sozlesme_disi_skalere_devreder() {
    let Some(y) = avx2_yuvalari("mc16_bilesik_avx2::p34_bilesik16_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (bd, max) = (BitDepth::Ten, 1023u16);
    let sk = McDsp::scalar_for(bd);
    let mut rng = Rng(7);
    let (w, h, ds) = (16usize, 8usize, 32usize);
    let t1 = tmp(&mut rng, PREP16_ARALIK[0], 0, w * h, 0);
    let t2 = tmp(&mut rng, PREP16_ARALIK[0], 0, w * h, 1);
    let d0 = skaler_devir_sayisi();
    // Ağırlık 0…16 dışında: skalere devreder, sonuç skalerle aynı.
    let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
    (y.w_avg)(&mut a, ds, &t1, &t2, w, h, 20, bd, max);
    (sk.w_avg)(&mut b, ds, &t1, &t2, w, h, 20, bd, max);
    assert_eq!(a, b);
    // `w_mask[0]`'a 4:2:0 bayrakları: alt tür uyuşmaz, skalere devreder.
    let mut ma = vec![0u8; (w / 2) * (h / 2)];
    let mut mb = ma.clone();
    let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
    (y.w_mask[0])(&mut a, ds, &t1, &t2, w, h, &mut ma, 0, true, true, bd, max);
    (sk.w_mask[2])(&mut b, ds, &t1, &t2, w, h, &mut mb, 0, true, true, bd, max);
    assert_eq!((a, ma), (b, mb));
    // Hedef satır adımı `2w`'den küçük (`ds = w` bayt): sözleşme dışı; skaler
    // gövde satırları sırayla yazar (örtüşen satırlar), sonuç skalerle aynı.
    let (mut a, mut b) = (hedef(w, h, w), hedef(w, h, w));
    (y.avg)(&mut a, w, &t1, &t2, w, h, bd, max);
    (sk.avg)(&mut b, w, &t1, &t2, w, h, bd, max);
    assert_eq!(a, b);
    assert_eq!(
        skaler_devir_sayisi(),
        d0 + 3,
        "üç sözleşme dışı çağrı sayılmalı"
    );
}
