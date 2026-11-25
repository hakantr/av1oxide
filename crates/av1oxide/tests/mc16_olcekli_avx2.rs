//! P34 (F4-MC c) — 16 bpc (10/12 bit) AVX2 ölçekli gövdeler:
//! `mc_scaled[0..9]`, `mct_scaled[0..9]`.
//!
//! - **Eşitlik:** 10 ve 12 bitte 20 yuva × `w` 2…128 × `h` 2…128 ve 1, 3, 6,
//!   12, 24 × parametre seti (`dy` 1024, 2048, genel; `dx` 1024, 2048, 512…2048
//!   arası, uç 1; `mx`, `my` 0, 1023 ve rastgele) × kaynak satır adımı (bayt) ×
//!   piksel deseni (rastgele, hepsi `bitdepth_max`, 0/`bitdepth_max` dama) için
//!   AVX2 çıktısı skaler gövdeyle bit-exact.
//! - **Sentinel:** kaynak dilimi okuma bölgesinin son pikselinde biter (8-tap:
//!   dolgu satırı `son_y + 7`, sütunu `son_x + 7`; bilinear `+ 4`); `put`
//!   hedefinde satır aralarına (`2w` ötesi) ve sona, `prep` `tmp`'nin `w·h`
//!   sonrasına nöbetçi.
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; sözleşme dışı
//!   girdi (`dy > 2048`, bölge sonundan kısa dilim) skalere devreder ve sayılır.
//!
//! AVX2 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::mc::{McDsp, McPrepScaledFn, McPutScaledFn, skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

const NOBETCI: u8 = 0xA5;
const NOBETCI16: i16 = 0x5A5A;

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
        lo + ((self.u64() >> 33) % (hi - lo + 1) as u64) as i32
    }
}

type Yuvalar = ([McPutScaledFn; 10], [McPrepScaledFn; 10]);

fn avx2_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        use av1oxide::dsp::{mc::Filter2d, x86::mc16_olcekli as o};
        return Some((
            Filter2d::ALL.map(|f| o::put_scaled_fn(t, f)),
            Filter2d::ALL.map(|f| o::prep_scaled_fn(t, f)),
        ));
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P34] 16 bpc AVX2 ölçekli gövdeler AVX2 ister",
    );
    None
}

/// Okuma bölgesinin sonundan sonraki bayt ofseti (`ss` bayt).
fn bolge_sonu(bilin: bool, ss: usize, w: usize, h: usize, p: (i32, i32, i32, i32)) -> usize {
    let (mx, my, dx, dy) = p;
    let son_y = ((my + dy * (h as i32 - 1)) >> 10) as usize;
    let son_x = ((mx + dx * (w as i32 - 1)) >> 10) as usize;
    let ek = if bilin { 4 } else { 7 };
    (son_y + ek) * ss + 2 * (son_x + ek) + 2
}

/// `n` bayt (çift) kaynak: `u16` pikseller (desen 0 rastgele, 1 hepsi `max`,
/// 2 dama; satır adımı `ss` bayt).
fn kaynak(rng: &mut Rng, desen: usize, n: usize, ss: usize, max: u16) -> Vec<u8> {
    (0..n / 2)
        .flat_map(|i| {
            let (r, c) = ((2 * i) / ss, ((2 * i) % ss) / 2);
            let v: u16 = match desen {
                0 => rng.aralik(0, i32::from(max)) as u16,
                1 => max,
                _ => {
                    if (r + c).is_multiple_of(2) {
                        0
                    } else {
                        max
                    }
                }
            };
            v.to_ne_bytes()
        })
        .collect()
}

#[test]
fn p34_olcekli16_avx2_skalerle_bit_exact_ve_sentinel() {
    let Some((put, prep)) =
        avx2_yuvalari("mc16_olcekli_avx2::p34_olcekli16_avx2_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x0005_ca1e_d0d0_ca34);
    let boy = [2usize, 4, 8, 16, 32, 64, 128];
    // Çözme hattının 2'nin kuvveti olmayan yükseklikleri (OBMC üst komşusu) ve uçlar.
    let ek_h = [1usize, 3, 6, 12, 24];
    let mut cagri = 0usize;
    for (bd, max) in [(BitDepth::Ten, 1023u16), (BitDepth::Twelve, 4095)] {
        let sk = McDsp::scalar_for(bd);
        for f in 0..10usize {
            for &w in &boy {
                for &h in boy.iter().chain(&ek_h) {
                    let buyuk = w * h > 2048;
                    let setler = if buyuk { 2 } else { 6 };
                    for ps in 0..setler {
                        let dy = match ps % 3 {
                            0 => 1024,
                            1 => 2048,
                            _ => rng.aralik(512, 2048),
                        };
                        let dx = match ps {
                            0 => 1024,
                            1 => 2048,
                            5 => 1,
                            _ => rng.aralik(512, 2048),
                        };
                        let mx = match ps % 4 {
                            0 => 0,
                            1 => 1023,
                            _ => rng.aralik(0, 1023),
                        };
                        let my = match (ps / 2) % 3 {
                            0 => 0,
                            1 => 1023,
                            _ => rng.aralik(0, 1023),
                        };
                        let bilin = f == 9;
                        let son_x = ((mx + dx * (w as i32 - 1)) >> 10) as usize;
                        let ss = 2 * (son_x + 8 + (ps % 3) * 5);
                        let n = bolge_sonu(bilin, ss, w, h, (mx, my, dx, dy));
                        let src = kaynak(&mut rng, ps % 3, n, ss, max);
                        // put
                        let ds = 2 * w + 6;
                        let dn = (h - 1) * ds + 2 * w + 32;
                        let (mut a, mut b) = (vec![NOBETCI; dn], vec![NOBETCI; dn]);
                        put[f](&mut a, ds, &src, ss, w, h, mx, my, dx, dy, bd, max);
                        sk.mc_scaled[f](&mut b, ds, &src, ss, w, h, mx, my, dx, dy, bd, max);
                        let ad =
                            format!("{bd:?} f={f} {w}x{h} mx={mx} my={my} dx={dx} dy={dy} ss={ss}");
                        assert_eq!(a, b, "mc_scaled {ad}");
                        for y in 0..h {
                            let son = if y + 1 < h { (y + 1) * ds } else { dn };
                            assert!(
                                a[y * ds + 2 * w..son].iter().all(|&v| v == NOBETCI),
                                "put nöbetçi {ad}"
                            );
                        }
                        // prep
                        let (mut a, mut b) =
                            (vec![NOBETCI16; w * h + 8], vec![NOBETCI16; w * h + 8]);
                        prep[f](&mut a, &src, ss, w, h, mx, my, dx, dy, bd, max);
                        sk.mct_scaled[f](&mut b, &src, ss, w, h, mx, my, dx, dy, bd, max);
                        assert_eq!(a, b, "mct_scaled {ad}");
                        assert!(
                            a[w * h..].iter().all(|&v| v == NOBETCI16),
                            "prep nöbetçi {ad}"
                        );
                        cagri += 2;
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
    println!("[P34] 16 bpc AVX2 ölçekli: {cagri} çağrı skalerle bit-exact (20 yuva × 10/12 bit)");
}

#[test]
fn p34_olcekli16_sozlesme_disi_skalere_devreder() {
    let Some((put, prep)) =
        avx2_yuvalari("mc16_olcekli_avx2::p34_olcekli16_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (bd, max) = (BitDepth::Ten, 1023u16);
    let sk = McDsp::scalar_for(bd);
    let (w, h, dy) = (8usize, 8usize, 3000);
    let ss = 64usize;
    let n = bolge_sonu(false, ss, w, h, (0, 0, 1024, dy));
    let src: Vec<u8> = (0..n / 2)
        .flat_map(|i| ((i * 13) as u16 % (max + 1)).to_ne_bytes())
        .collect();
    let d0 = skaler_devir_sayisi();
    let (mut a, mut b) = (vec![0u8; 16 * h], vec![0u8; 16 * h]);
    put[0](&mut a, 16, &src, ss, w, h, 0, 0, 1024, dy, bd, max);
    sk.mc_scaled[0](&mut b, 16, &src, ss, w, h, 0, 0, 1024, dy, bd, max);
    assert_eq!(a, b);
    // Bölge sonundan bir piksel (2 bayt) kısa dilim sözleşme dışı: devreder.
    // `mx = 0`, `dx = 1024`'te sütun alt pikseli 0 (birim): skaler gövde son
    // sütunu okumaz, sonucu aynı.
    let n = bolge_sonu(false, ss, w, h, (0, 0, 1024, 1024));
    let src: Vec<u8> = (0..n / 2)
        .flat_map(|i| ((i * 7) as u16 % (max + 1)).to_ne_bytes())
        .collect();
    let mut t = vec![0i16; w * h];
    let mut tb = vec![0i16; w * h];
    prep[0](&mut t, &src[..n - 2], ss, w, h, 0, 0, 1024, 1024, bd, max);
    sk.mct_scaled[0](&mut tb, &src[..n - 2], ss, w, h, 0, 0, 1024, 1024, bd, max);
    assert_eq!(t, tb);
    assert_eq!(
        skaler_devir_sayisi(),
        d0 + 2,
        "dy > 2048 ve kısa dilim sözleşme dışı, sayılmalı"
    );
    // Tam bölgeli dilim sözleşmede: devir yok.
    prep[0](&mut t, &src, ss, w, h, 0, 0, 1024, 1024, bd, max);
    assert_eq!(skaler_devir_sayisi(), d0 + 2, "tam bölge sözleşmede");
}
