//! P34 (F4-MC c) — 16 bpc (10/12 bit) AVX2 `put`/`prep` gövdeleri
//! (`mc[0..9]`, `mct[0..9]`).
//!
//! - **Eşitlik:** 10 ve 12 bitte her yuva, her genişlik sınıfı (2…128) ×
//!   yükseklik × alt piksel konumu (dar bloklarda `mx`, `my` 0..15'in tamamı) ×
//!   satır adımı (bayt) × girdi deseni (rastgele, hepsi 0, hepsi
//!   `bitdepth_max`, 0/`bitdepth_max` dama) için AVX2 çıktısı skaler gövdeyle
//!   bit-exact.
//! - **Sentinel:** kaynak dilimi okuma bölgesinin sonu + `MC_KUYRUK` baytta
//!   biter (fazlası okunursa dilim denetimi panikler); `put` hedefinde satır
//!   aralarına ve sona nöbetçi bayt (`2w` ötesine yazma 0); `prep` `tmp`'nin
//!   `w·h` sonrasına nöbetçi öğe; satır boşlukları ve kuyruk iki desenle
//!   doldurulunca çıktı değişmez.
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; kuyruksuz dilim
//!   ve tek yükseklik skalere devreder ve sayılır.
//!
//! AVX2 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::{Filter2d, MC_KUYRUK, McDsp, McPrepFn, McPutFn, skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

const NOBETCI: u8 = 0xA5;
const NOBETCI16: i16 = 0x5A5A;

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
}

type Yuvalar = ([McPutFn; 10], [McPrepFn; 10]);

/// AVX2 16 bpc yuvaları ya da ön koşul eksikliği.
fn avx2_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        use av1oxide::dsp::x86::mc16::{prep16_fn, put16_fn};
        return Some((
            Filter2d::ALL.map(|f| put16_fn(t, f)),
            Filter2d::ALL.map(|f| prep16_fn(t, f)),
        ));
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P34] 16 bpc AVX2 MC gövdeleri AVX2 ister",
    );
    None
}

/// Kaynak (bayt): dolgu hücresinden `(h+6)·ss + (w+7)·2 + MC_KUYRUK` bayt;
/// pikseller `u16` küçük sonlu; satır boşlukları ve kuyruk `dolgu` baytıyla.
fn kaynak(rng: &mut Rng, desen: u8, w: usize, h: usize, ss: usize, dolgu: u8, max: u16) -> Vec<u8> {
    let n = (h + 6) * ss + (w + 7) * 2 + MC_KUYRUK;
    let mut v = vec![dolgu; n];
    for r in 0..h + 7 {
        for c in 0..w + 7 {
            let p: u16 = match desen {
                0 => (rng.u64() >> 40) as u16 % (max + 1),
                1 => 0,
                2 => max,
                _ => {
                    if (r + c) % 2 == 0 {
                        0
                    } else {
                        max
                    }
                }
            };
            let o = r * ss + 2 * c;
            v[o..o + 2].copy_from_slice(&p.to_le_bytes());
        }
    }
    v
}

/// Boyut × alt piksel kümesi: dar bloklarda bütün `mx`, `my`; geniş
/// bloklarda kenar ve ara konumlar.
fn konumlar(w: usize, h: usize) -> Vec<(u8, u8)> {
    if w * h <= 128 {
        (0..16u8)
            .flat_map(|mx| (0..16u8).map(move |my| (mx, my)))
            .collect()
    } else {
        let k = [0u8, 1, 4, 8, 11, 15];
        k.iter()
            .flat_map(|&mx| k.iter().map(move |&my| (mx, my)))
            .collect()
    }
}

fn boyutlar() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for w in [2usize, 4, 8, 16, 32, 64, 128] {
        for h in [2usize, 4, 6, 8, 12, 16, 32, 64, 128] {
            let uygun = match w {
                2 => h <= 8,
                4 => h <= 16,
                8 => h <= 32,
                16 => (4..=64).contains(&h),
                32 => (8..=64).contains(&h),
                64 => (16..=128).contains(&h),
                _ => h >= 64,
            };
            if uygun {
                v.push((w, h));
            }
        }
    }
    v
}

/// Bit derinliğinin skaler tablosu ve `bitdepth_max`'ı.
fn skaler(bd: BitDepth) -> (McDsp, u16) {
    match bd {
        BitDepth::Ten => (McDsp::scalar::<10>(), 1023),
        _ => (McDsp::scalar::<12>(), 4095),
    }
}

#[test]
fn p34_mc16_avx2_skalerle_bit_exact_ve_sentinel() {
    let Some((put, prep)) =
        avx2_yuvalari("mc16_avx2::p34_mc16_avx2_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x0F4_3C34);
    let mut n = 0usize;
    for bd in [BitDepth::Ten, BitDepth::Twelve] {
        let (sk, max) = skaler(bd);
        for (w, h) in boyutlar() {
            for (mi, (mx, my)) in konumlar(w, h).into_iter().enumerate() {
                let desen = (mi % 7).min(3) as u8;
                let ss = (w + 7) * 2 + [0usize, 10, 64][mi % 3];
                let ds = 2 * w + [0usize, 6][mi % 2];
                let dolgu = [0u8, 255][mi % 2];
                let src = kaynak(&mut rng, desen, w, h, ss, dolgu, max);
                for f in Filter2d::ALL {
                    let i = f.index();
                    let ctx = format!(
                        "{bd:?} {f:?} {w}x{h} mx={mx} my={my} ss={ss} ds={ds} desen={desen}"
                    );
                    let dlen = (h - 1) * ds + 2 * w + 32;
                    let mut beklenen = vec![NOBETCI; dlen];
                    (sk.mc[i])(&mut beklenen, ds, &src, ss, w, h, mx, my, bd, max);
                    let mut d = vec![NOBETCI; dlen];
                    (put[i])(&mut d, ds, &src, ss, w, h, mx, my, bd, max);
                    assert_eq!(
                        d, beklenen,
                        "put {ctx}: skaler farkı ya da 2w ötesine yazma"
                    );
                    for y in 0..h {
                        let son = if y + 1 < h { (y + 1) * ds } else { dlen };
                        assert!(
                            d[y * ds + 2 * w..son].iter().all(|&v| v == NOBETCI),
                            "put {ctx}: nöbetçi"
                        );
                    }
                    let mut tb = vec![NOBETCI16; w * h + 16];
                    (sk.mct[i])(&mut tb, &src, ss, w, h, mx, my, bd, max);
                    let mut t = vec![NOBETCI16; w * h + 16];
                    (prep[i])(&mut t, &src, ss, w, h, mx, my, bd, max);
                    assert_eq!(t, tb, "prep {ctx}: skaler farkı ya da w·h ötesine yazma");
                    assert!(
                        t[w * h..].iter().all(|&v| v == NOBETCI16),
                        "prep {ctx}: nöbetçi"
                    );
                    n += 2;
                }
            }
        }
    }
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrılar skalere devretmemeli"
    );
    println!("[P34] 16 bpc AVX2 put/prep: {n} çağrı skalerle bit-exact (20 yuva × 10/12 bit)");
}

/// Kaynak dilimi tam okuma bölgesi + `MC_KUYRUK` ile biter; satır boşlukları
/// ve kuyruk iki farklı desenle doldurulunca çıktı değişmez.
#[test]
fn p34_mc16_okuma_bolgesi_disi_ciktiya_karismaz() {
    let Some((put, prep)) =
        avx2_yuvalari("mc16_avx2::p34_mc16_okuma_bolgesi_disi_ciktiya_karismaz")
    else {
        return;
    };
    let mut n = 0usize;
    for bd in [BitDepth::Ten, BitDepth::Twelve] {
        let max = skaler(bd).1;
        for (w, h) in boyutlar() {
            for (mx, my) in [(0u8, 0u8), (5, 0), (0, 11), (7, 9), (15, 15)] {
                let ss = (w + 7) * 2;
                for f in Filter2d::ALL {
                    let i = f.index();
                    let mut cikti: Option<(Vec<u8>, Vec<i16>)> = None;
                    for dolgu in [0u8, 255] {
                        let src =
                            kaynak(&mut Rng(0x5EED + (w * h) as u64), 0, w, h, ss, dolgu, max);
                        assert_eq!(src.len(), (h + 6) * ss + (w + 7) * 2 + MC_KUYRUK);
                        let mut d = vec![0u8; (h - 1) * 2 * w + 2 * w];
                        (put[i])(&mut d, 2 * w, &src, ss, w, h, mx, my, bd, max);
                        let mut t = vec![0i16; w * h];
                        (prep[i])(&mut t, &src, ss, w, h, mx, my, bd, max);
                        match &cikti {
                            None => cikti = Some((d, t)),
                            Some((d0, t0)) => {
                                let ctx = format!("{bd:?} {f:?} {w}x{h} mx={mx} my={my}");
                                assert_eq!(&d, d0, "put {ctx}: dolgu çıktıya karıştı");
                                assert_eq!(&t, t0, "prep {ctx}: dolgu çıktıya karıştı");
                            }
                        }
                        n += 2;
                    }
                }
            }
        }
    }
    println!("[P34] 16 bpc okuma bölgesi + K sınırı: {n} çağrı");
}

/// Kuyruksuz dilim ve tek yükseklik sözleşme dışıdır: giriş skaler
/// gövdeye devreder, sonucu skalerle aynıdır ve devir sayılır.
#[test]
fn p34_mc16_sozlesme_disi_skalere_devreder() {
    let Some((put, prep)) = avx2_yuvalari("mc16_avx2::p34_mc16_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (sk, max) = skaler(BitDepth::Ten);
    let (w, h, ss) = (8usize, 8usize, 30usize);
    let mut src = kaynak(&mut Rng(7), 0, w, h, ss, 0, max);
    src.truncate((h + 6) * ss + (w + 7) * 2);
    let once = skaler_devir_sayisi();
    let mut d = vec![0u8; 2 * w * h];
    (put[0])(&mut d, 2 * w, &src, ss, w, h, 3, 5, BitDepth::Ten, max);
    let mut b = vec![0u8; 2 * w * h];
    (sk.mc[0])(&mut b, 2 * w, &src, ss, w, h, 3, 5, BitDepth::Ten, max);
    assert_eq!(d, b);
    let mut t = vec![0i16; w * h];
    (prep[0])(&mut t, &src, ss, w, h, 3, 5, BitDepth::Ten, max);
    let mut tb = vec![0i16; w * h];
    (sk.mct[0])(&mut tb, &src, ss, w, h, 3, 5, BitDepth::Ten, max);
    assert_eq!(t, tb);
    assert_eq!(skaler_devir_sayisi(), once + 2, "kuyruksuz dilim sayılmalı");
    // Tek yükseklik (`h` 1…127 tek) sözleşme dışı: tam bölgeli dilimde de devreder.
    let h = 5usize;
    let src5 = kaynak(&mut Rng(9), 0, w, h, ss, 0, max);
    let once = skaler_devir_sayisi();
    let mut d = vec![0u8; 2 * w * h];
    let mut b = vec![0u8; 2 * w * h];
    (put[0])(&mut d, 2 * w, &src5, ss, w, h, 3, 5, BitDepth::Ten, max);
    (sk.mc[0])(&mut b, 2 * w, &src5, ss, w, h, 3, 5, BitDepth::Ten, max);
    assert_eq!(d, b);
    assert_eq!(skaler_devir_sayisi(), once + 1, "tek yükseklik sayılmalı");
}
