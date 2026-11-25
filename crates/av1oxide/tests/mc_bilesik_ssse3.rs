//! P35 (F4-MC d) — 8 bpc SSSE3 bileşik gövdeleri: `avg`, `w_avg`, `mask`,
//! `w_mask[0..2]`.
//!
//! - **Eşitlik:** her boyut (`w`, `h` 4…128; `w_mask` 8…128) × hedef satır
//!   adımı × girdi deseni (rastgele, en küçük, en büyük, en küçük/en büyük
//!   dama; değerler 8 bpc `prep` aralığında, [`PREP_ARALIK`]) × ağırlık
//!   (`w_avg` 0…16) / maske (rastgele 0…64, hepsi 0, hepsi 64, 0/64 dama) /
//!   alt tür × `sign` için SSSE3 çıktısı skaler gövdeyle bit-exact.
//! - **Sentinel:** `tmp` ve maske dilimleri tam `w·h` öğede biter (fazlası
//!   okunursa dilim denetimi panikler); hedefte satır aralarına ve sona,
//!   `w_mask` maskesinde bölge sonrasına nöbetçi konur.
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; sözleşme dışı
//!   girdi skalere devreder ve sayılır.
//!
//! SSSE3 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::{AvgFn, MaskFn, McDsp, WAvgFn, WMaskFn, skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

/// 8 bpc `prep` / `warp8x8t` çıktı aralığı (`dsp::x86::mc_bilesik::PREP_ARALIK`
/// ile aynı; aarch64'te modül derlenmediği için burada da yazılı).
const PREP_ARALIK: (i16, i16) = (-5132, 9212);
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

fn ssse3_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).ssse3_token()
    {
        use av1oxide::dsp::x86::mc_bilesik_ssse3 as b;
        assert_eq!(av1oxide::dsp::x86::mc_bilesik::PREP_ARALIK, PREP_ARALIK);
        return Some(Yuvalar {
            avg: b::avg_fn(t),
            w_avg: b::w_avg_fn(t),
            mask: b::mask_fn(t),
            w_mask: [b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)],
        });
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P35] 8 bpc SSSE3 bileşik gövdeleri SSSE3 ister",
    );
    None
}

/// `n` öğelik `tmp` (desen: 0 rastgele, 1 en küçük, 2 en büyük, 3 dama).
fn tmp(rng: &mut Rng, desen: u8, n: usize, kaydir: usize) -> Vec<i16> {
    let (lo, hi) = (i32::from(PREP_ARALIK.0), i32::from(PREP_ARALIK.1));
    (0..n)
        .map(|i| match desen {
            0 => rng.aralik(lo, hi) as i16,
            1 => PREP_ARALIK.0,
            2 => PREP_ARALIK.1,
            _ => {
                if (i + kaydir).is_multiple_of(2) {
                    PREP_ARALIK.0
                } else {
                    PREP_ARALIK.1
                }
            }
        })
        .collect()
}

/// Bileşik maske (desen: 0 rastgele 0…64, 1 hepsi 0, 2 hepsi 64, 3 dama).
fn maske(rng: &mut Rng, desen: u8, n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| match desen {
            0 => rng.aralik(0, 64) as u8,
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

/// Hedef: `h` satır, adım `ds`; satır arası ve son `16` bayt nöbetçi.
fn hedef(w: usize, h: usize, ds: usize) -> Vec<u8> {
    let _ = w;
    vec![NOBETCI; (h - 1) * ds + w + 16]
}

/// Nöbetçi denetimi: satır aralarında ve sonda değişiklik yok.
fn nobetci_saglam(d: &[u8], w: usize, h: usize, ds: usize) -> bool {
    (0..h).all(|y| {
        let son = if y + 1 < h { (y + 1) * ds } else { d.len() };
        d[y * ds + w..son].iter().all(|&b| b == NOBETCI)
    })
}

const BOYUTLAR: [usize; 6] = [4, 8, 16, 32, 64, 128];

#[test]
fn p35_bilesik_ssse3_skalerle_bit_exact_ve_sentinel() {
    let Some(y) =
        ssse3_yuvalari("mc_bilesik_ssse3::p35_bilesik_ssse3_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut cagri = 0usize;
    for &w in &BOYUTLAR {
        for &h in &BOYUTLAR {
            let n = w * h;
            let buyuk = n > 4096;
            let adimlar: &[usize] = if buyuk { &[w + 5] } else { &[w, w + 5, w + 32] };
            let desenler: &[u8] = if buyuk { &[0, 3] } else { &[0, 1, 2, 3] };
            let agirliklar: Vec<i32> = if buyuk {
                vec![0, 1, 7, 8, 15, 16]
            } else {
                (0..=16).collect()
            };
            for &ds in adimlar {
                for &desen in desenler {
                    let t1 = tmp(&mut rng, desen, n, 0);
                    let t2 = tmp(&mut rng, desen, n, 1);
                    // avg
                    let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
                    (y.avg)(&mut a, ds, &t1, &t2, w, h, BitDepth::Eight, 255);
                    (sk.avg)(&mut b, ds, &t1, &t2, w, h, BitDepth::Eight, 255);
                    assert_eq!(a, b, "avg {w}x{h} ds={ds} desen={desen}");
                    assert!(nobetci_saglam(&a, w, h, ds), "avg nöbetçi {w}x{h}");
                    cagri += 1;
                    // w_avg
                    for &wt in &agirliklar {
                        let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
                        (y.w_avg)(&mut a, ds, &t1, &t2, w, h, wt, BitDepth::Eight, 255);
                        (sk.w_avg)(&mut b, ds, &t1, &t2, w, h, wt, BitDepth::Eight, 255);
                        assert_eq!(a, b, "w_avg {w}x{h} ds={ds} desen={desen} w={wt}");
                        assert!(nobetci_saglam(&a, w, h, ds), "w_avg nöbetçi");
                        cagri += 1;
                    }
                    // mask
                    for md in 0..4u8 {
                        let m = maske(&mut rng, md, n);
                        let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
                        (y.mask)(&mut a, ds, &t1, &t2, w, h, &m, BitDepth::Eight, 255);
                        (sk.mask)(&mut b, ds, &t1, &t2, w, h, &m, BitDepth::Eight, 255);
                        assert_eq!(a, b, "mask {w}x{h} ds={ds} desen={desen} m={md}");
                        assert!(nobetci_saglam(&a, w, h, ds), "mask nöbetçi");
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
                                &mut a,
                                ds,
                                &t1,
                                &t2,
                                w,
                                h,
                                &mut ma,
                                sign,
                                sh,
                                sv,
                                BitDepth::Eight,
                                255,
                            );
                            (sk.w_mask[alt])(
                                &mut b,
                                ds,
                                &t1,
                                &t2,
                                w,
                                h,
                                &mut mb,
                                sign,
                                sh,
                                sv,
                                BitDepth::Eight,
                                255,
                            );
                            let ad =
                                format!("w_mask[{alt}] {w}x{h} ds={ds} desen={desen} sign={sign}");
                            assert_eq!(a, b, "{ad} piksel");
                            assert_eq!(ma, mb, "{ad} maske");
                            assert!(nobetci_saglam(&a, w, h, ds), "{ad} nöbetçi");
                            assert!(ma[mn..].iter().all(|&v| v == NOBETCI), "{ad} maske nöbetçi");
                            cagri += 1;
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
    println!("[P35] 8 bpc SSSE3 bileşik: {cagri} çağrı skalerle bit-exact (6 yuva)");
}

#[test]
fn p35_bilesik_ssse3_sozlesme_disi_skalere_devreder() {
    let Some(y) =
        ssse3_yuvalari("mc_bilesik_ssse3::p35_bilesik_ssse3_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let mut rng = Rng(7);
    let (w, h, ds) = (16usize, 8usize, 16usize);
    let t1 = tmp(&mut rng, 0, w * h, 0);
    let t2 = tmp(&mut rng, 0, w * h, 1);
    let d0 = skaler_devir_sayisi();
    // Ağırlık 0…16 dışında: skalere devreder, sonuç skalerle aynı.
    let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
    (y.w_avg)(&mut a, ds, &t1, &t2, w, h, 20, BitDepth::Eight, 255);
    (sk.w_avg)(&mut b, ds, &t1, &t2, w, h, 20, BitDepth::Eight, 255);
    assert_eq!(a, b);
    // `w_mask[0]`'a 4:2:0 bayrakları: alt tür uyuşmaz, skalere devreder.
    let mut ma = vec![0u8; (w / 2) * (h / 2)];
    let mut mb = ma.clone();
    let (mut a, mut b) = (hedef(w, h, ds), hedef(w, h, ds));
    (y.w_mask[0])(
        &mut a,
        ds,
        &t1,
        &t2,
        w,
        h,
        &mut ma,
        0,
        true,
        true,
        BitDepth::Eight,
        255,
    );
    (sk.w_mask[2])(
        &mut b,
        ds,
        &t1,
        &t2,
        w,
        h,
        &mut mb,
        0,
        true,
        true,
        BitDepth::Eight,
        255,
    );
    assert_eq!((a, ma), (b, mb));
    // `w = 2`: sözleşme dışı (`avg` 4…128), skaler gövde yine doğru.
    let (w2, h2) = (2usize, 4usize);
    let (mut a, mut b) = (hedef(w2, h2, 2), hedef(w2, h2, 2));
    (y.avg)(&mut a, 2, &t1[..8], &t2[..8], w2, h2, BitDepth::Eight, 255);
    (sk.avg)(&mut b, 2, &t1[..8], &t2[..8], w2, h2, BitDepth::Eight, 255);
    assert_eq!(a, b);
    assert_eq!(
        skaler_devir_sayisi(),
        d0 + 3,
        "üç sözleşme dışı çağrı sayılmalı"
    );
}
