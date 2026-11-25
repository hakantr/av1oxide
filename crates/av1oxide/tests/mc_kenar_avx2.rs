//! P33 (F4-MC b) — 8 bpc AVX2 `emu_edge` gövdesi.
//!
//! - **Eşitlik:** `bw`, `bh` 1…135 (2'nin kuvvetleri ve `w + 7` biçimli tek
//!   değerler) × konum (içeride, kısmi ve tamamen sol/sağ/üst/alt taşma,
//!   köşeler) × görüntü boyu (`bw`'den büyük ve küçük) × satır adımı için AVX2
//!   çıktısı skaler gövdeyle bit-exact.
//! - **Sentinel:** kaynak dilimi kırpılmış okuma penceresinin son baytında
//!   biter (pencere dışı okuma panik olurdu); hedefte satır aralarına ve sona
//!   nöbetçi konur (`bw` ötesine yazma 0).
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; sözleşme dışı
//!   girdi (hedef satır adımı `< bw`) skalere devreder ve sayılır.
//!
//! AVX2 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::mc::{EmuEdgeFn, McDsp, skaler_devir_sayisi};
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
}

fn avx2_yuvasi(test: &str) -> Option<EmuEdgeFn> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        return Some(av1oxide::dsp::x86::mc_kenar::emu_edge_fn(t));
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P33] 8 bpc AVX2 emu_edge gövdesi AVX2 ister",
    );
    None
}

/// Konum: 0 içeride, 1 kısmi sol/üst, 2 tamamen sol/üst, 3 kısmi sağ/alt,
/// 4 tamamen sağ/alt.
fn konum(k: usize, b: usize, i: usize) -> i32 {
    let (b, i) = (b as i32, i as i32);
    match k {
        0 => (i - b).max(0) / 2,
        1 => -(b / 2),
        2 => -b - 3,
        3 => i - b / 2,
        _ => i + 3,
    }
}

/// Kırpılmış okuma penceresinin son baytından sonraki ofset.
fn pencere_sonu(bw: usize, bh: usize, iw: usize, ih: usize, x: i32, y: i32, ss: usize) -> usize {
    let [bw, bh, iw, ih] = [bw, bh, iw, ih].map(|v| v as i64);
    let (x, y) = (i64::from(x), i64::from(y));
    let sol = (-x).clamp(0, bw - 1);
    let sag = (x + bw - iw).clamp(0, bw - 1);
    let ust = (-y).clamp(0, bh - 1);
    let alt = (y + bh - ih).clamp(0, bh - 1);
    let (cx, cy) = (x.clamp(0, iw - 1), y.clamp(0, ih - 1));
    ((cy + bh - ust - alt - 1) * ss as i64 + cx + bw - sol - sag) as usize
}

#[test]
fn p33_emu_edge_avx2_skalerle_bit_exact_ve_sentinel() {
    let Some(f) = avx2_yuvasi("mc_kenar_avx2::p33_emu_edge_avx2_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x00e3_ed9e_0005_eed1);
    let bd = BitDepth::Eight;
    let boylar = [1usize, 2, 4, 7, 8, 11, 15, 16, 23, 32, 39, 64, 71, 128, 135];
    let mut cagri = 0usize;
    for (bi, &bw) in boylar.iter().enumerate() {
        let bh = boylar[(bi + 5) % boylar.len()];
        for &(iw, ih) in &[(160usize, 160usize), (40, 160), (160, 40), (24, 24)] {
            let ss = iw + 3 + bi % 5;
            let goruntu: Vec<u8> = (0..ss * ih).map(|_| (rng.u64() >> 40) as u8).collect();
            for xk in 0..5 {
                for yk in 0..5 {
                    let (x, y) = (konum(xk, bw, iw), konum(yk, bh, ih));
                    // Kaynak dilimi pencerenin son baytında biter.
                    let src = &goruntu[..pencere_sonu(bw, bh, iw, ih, x, y, ss)];
                    for &ds in &[bw, bw + 9] {
                        let n = (bh - 1) * ds + bw + 16;
                        let (mut a, mut b) = (vec![NOBETCI; n], vec![NOBETCI; n]);
                        f(bw, bh, iw, ih, x, y, &mut a, ds, src, ss, bd, 255);
                        (sk.emu_edge)(bw, bh, iw, ih, x, y, &mut b, ds, src, ss, bd, 255);
                        assert_eq!(
                            a, b,
                            "emu_edge {bw}x{bh} görüntü {iw}x{ih} konum ({x},{y}) ds={ds}"
                        );
                        for r in 0..bh {
                            let son = if r + 1 < bh { (r + 1) * ds } else { n };
                            assert!(a[r * ds + bw..son].iter().all(|&v| v == NOBETCI), "nöbetçi");
                        }
                        cagri += 1;
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
    println!("[P33] 8 bpc AVX2 emu_edge: {cagri} çağrı skalerle bit-exact");
}

#[test]
fn p33_emu_edge_sozlesme_disi_skalere_devreder() {
    let Some(f) = avx2_yuvasi("mc_kenar_avx2::p33_emu_edge_sozlesme_disi_skalere_devreder") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let bd = BitDepth::Eight;
    let (bw, bh, iw, ih) = (8usize, 8usize, 32usize, 32usize);
    let goruntu: Vec<u8> = (0..32 * 32).map(|i| (i * 7) as u8).collect();
    let d0 = skaler_devir_sayisi();
    // Hedef satır adımı `bw`'den küçük: sözleşme dışı (skaler gövde aynı sonucu
    // kendi düzeniyle üretir; satırlar örtüşür, karşılaştırma bayt bayt).
    let ds = 4usize;
    let n = (bh - 1) * ds + bw;
    let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
    f(bw, bh, iw, ih, 3, 3, &mut a, ds, &goruntu, 32, bd, 255);
    (sk.emu_edge)(bw, bh, iw, ih, 3, 3, &mut b, ds, &goruntu, 32, bd, 255);
    assert_eq!(a, b);
    assert_eq!(
        skaler_devir_sayisi(),
        d0 + 1,
        "sözleşme dışı çağrı sayılmalı"
    );
}
