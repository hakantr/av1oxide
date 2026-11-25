//! P33 (F4-MC b) — 8 bpc AVX2 ölçekli gövdeler: `mc_scaled[0..9]`,
//! `mct_scaled[0..9]`.
//!
//! - **Eşitlik:** 20 yuva × `w` 2…128 × `h` 2…128 ve 1, 3, 6, 12, 24 (OBMC üst
//!   komşusu `3·2^k` yükseklikle çağrılır) × parametre seti (`dy` 1024, 2048,
//!   genel; `dx` 1024, 2048, 512…2048 arası, uç 1; `mx`, `my` 0, 1023 ve
//!   rastgele) × kaynak satır adımı × piksel deseni için AVX2 çıktısı skaler
//!   gövdeyle bit-exact.
//! - **Sentinel:** kaynak dilimi okuma bölgesinin son baytında biter (8-tap:
//!   dolgu satırı `son_y + 7`, sütunu `son_x + 7`; bilinear `+ 4`); `put`
//!   hedefinde satır aralarına ve sona, `prep` `tmp`'nin `w·h` sonrasına nöbetçi.
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; sözleşme dışı
//!   girdi (`dy > 2048`) skalere devreder ve sayılır.
//!
//! AVX2 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
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
        use av1oxide::dsp::{mc::Filter2d, x86::mc_olcekli as o};
        return Some((
            Filter2d::ALL.map(|f| o::put_scaled_fn(t, f)),
            Filter2d::ALL.map(|f| o::prep_scaled_fn(t, f)),
        ));
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P33] 8 bpc AVX2 ölçekli gövdeler AVX2 ister",
    );
    None
}

/// Okuma bölgesinin sonundan sonraki ofset.
fn bolge_sonu(bilin: bool, ss: usize, w: usize, h: usize, p: (i32, i32, i32, i32)) -> usize {
    let (mx, my, dx, dy) = p;
    let son_y = ((my + dy * (h as i32 - 1)) >> 10) as usize;
    let son_x = ((mx + dx * (w as i32 - 1)) >> 10) as usize;
    let ek = if bilin { 4 } else { 7 };
    (son_y + ek) * ss + son_x + ek + 1
}

#[test]
fn p33_olcekli_avx2_skalerle_bit_exact_ve_sentinel() {
    let Some((put, prep)) =
        avx2_yuvalari("mc_olcekli_avx2::p33_olcekli_avx2_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x0005_ca1e_d0d0_cafe);
    let bd = BitDepth::Eight;
    let boy = [2usize, 4, 8, 16, 32, 64, 128];
    // Çözme hattının 2'nin kuvveti olmayan yükseklikleri (OBMC üst komşusu) ve uçlar.
    let ek_h = [1usize, 3, 6, 12, 24];
    let mut cagri = 0usize;
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
                    let ss = son_x + 8 + (ps % 3) * 5;
                    let n = bolge_sonu(bilin, ss, w, h, (mx, my, dx, dy));
                    let desen = ps % 3;
                    let src: Vec<u8> = (0..n)
                        .map(|i| match desen {
                            0 => (rng.u64() >> 40) as u8,
                            1 => 255,
                            _ => {
                                if (i / ss + i % ss).is_multiple_of(2) {
                                    0
                                } else {
                                    255
                                }
                            }
                        })
                        .collect();
                    // put
                    let ds = w + 3;
                    let dn = (h - 1) * ds + w + 16;
                    let (mut a, mut b) = (vec![NOBETCI; dn], vec![NOBETCI; dn]);
                    put[f](&mut a, ds, &src, ss, w, h, mx, my, dx, dy, bd, 255);
                    sk.mc_scaled[f](&mut b, ds, &src, ss, w, h, mx, my, dx, dy, bd, 255);
                    let ad = format!("f={f} {w}x{h} mx={mx} my={my} dx={dx} dy={dy} ss={ss}");
                    assert_eq!(a, b, "mc_scaled {ad}");
                    for y in 0..h {
                        let son = if y + 1 < h { (y + 1) * ds } else { dn };
                        assert!(
                            a[y * ds + w..son].iter().all(|&v| v == NOBETCI),
                            "put nöbetçi {ad}"
                        );
                    }
                    // prep
                    let (mut a, mut b) = (vec![NOBETCI16; w * h + 8], vec![NOBETCI16; w * h + 8]);
                    prep[f](&mut a, &src, ss, w, h, mx, my, dx, dy, bd, 255);
                    sk.mct_scaled[f](&mut b, &src, ss, w, h, mx, my, dx, dy, bd, 255);
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
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrılar skalere devretmemeli"
    );
    println!("[P33] 8 bpc AVX2 ölçekli: {cagri} çağrı skalerle bit-exact (20 yuva)");
}

#[test]
fn p33_olcekli_sozlesme_disi_skalere_devreder() {
    let Some((put, _)) =
        avx2_yuvalari("mc_olcekli_avx2::p33_olcekli_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let bd = BitDepth::Eight;
    let (w, h, dy) = (8usize, 8usize, 3000);
    let ss = 32usize;
    let n = bolge_sonu(false, ss, w, h, (0, 0, 1024, dy));
    let src: Vec<u8> = (0..n).map(|i| (i * 13) as u8).collect();
    let d0 = skaler_devir_sayisi();
    let (mut a, mut b) = (vec![0u8; 7 * w + w], vec![0u8; 7 * w + w]);
    put[0](&mut a, w, &src, ss, w, h, 0, 0, 1024, dy, bd, 255);
    sk.mc_scaled[0](&mut b, w, &src, ss, w, h, 0, 0, 1024, dy, bd, 255);
    assert_eq!(a, b);
    assert_eq!(
        skaler_devir_sayisi(),
        d0 + 1,
        "dy > 2048 sözleşme dışı, sayılmalı"
    );
}
