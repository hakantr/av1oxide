//! P36 (F4-MC e) — 16 bpc (10/12 bit) AVX512ICL superres büyütme (`resize`).
//!
//! - **Eşitlik:** 10 ve 12 bitte superres oranları (`dx` = `(src_w << 14 +
//!   dst_w/2) / dst_w` biçiminde, 9/8…16/8 büyütme) ve uç adımlar × `dst_w`
//!   (1…260; 8'in katı olan ve olmayan) × `mx0` (0, 0x3fff, rastgele) × satır
//!   sayısı × satır adımı (bayt) × piksel deseni (rastgele, 0,
//!   `bitdepth_max`, dama) için AVX512ICL çıktısı skaler gövdeyle bit-exact; kenar
//!   kırpması iki uçta da sınanır.
//! - **Sentinel:** son kaynak satırı dilimi tam `2·src_w` baytta biter;
//!   hedefte satır aralarına ve sona nöbetçi konur (`2·dst_w` ötesine yazma 0).
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; sözleşme dışı
//!   girdi (kaynak satır adımı `< 2·src_w`) skalere devreder ve sayılır.
//!
//! AVX512ICL yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::mc::skaler_devir_sayisi;
use av1oxide::dsp::resize::{ResizeBitDepth, ResizeFn, scalar_for};
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
    fn aralik(&mut self, lo: i32, hi: i32) -> i32 {
        lo + ((self.u64() >> 33) % (hi - lo + 1) as u64) as i32
    }
}

fn avx512_yuvasi(test: &str) -> Option<ResizeFn> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx512icl_token()
    {
        return Some(av1oxide::dsp::x86::buyutme16_avx512::resize_fn(t));
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P36] 16 bpc AVX512ICL büyütme gövdesi AVX512ICL ister",
    );
    None
}

#[test]
fn p36_buyutme16_avx512_skalerle_bit_exact_ve_sentinel() {
    let Some(f) =
        avx512_yuvasi("buyutme16_avx512::p36_buyutme16_avx512_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x00b0_7ee5_0123_4634);
    let mut cagri = 0usize;
    for bd in [ResizeBitDepth::B10, ResizeBitDepth::B12] {
        let sk = scalar_for(bd);
        let max = bd.bitdepth_max() as u16;
        for &dst_w in &[1usize, 2, 7, 8, 9, 15, 16, 33, 64, 100, 127, 128, 255, 260] {
            for d in 9..=16usize {
                // superres: src_w = dst_w · 8 / d (en az 1), dx = (src_w << 14 + dst_w/2) / dst_w.
                let src_w = (dst_w * 8 / d).max(1);
                let dx = (((src_w as i64) << 14) + (dst_w as i64 / 2)) / dst_w as i64;
                for kip in 0..3 {
                    let mx0 = match kip {
                        0 => 0,
                        1 => 0x3fff,
                        _ => rng.aralik(0, 0x3fff),
                    };
                    let h = 1 + (d + kip) % 5;
                    let ss = 2 * (src_w + kip * 7);
                    let ds = 2 * (dst_w + (d % 3) * 5);
                    let desen = (d + kip) % 4;
                    let np = ((h - 1) * ss) / 2 + src_w;
                    let mut src: Vec<u8> = (0..np)
                        .flat_map(|i| {
                            let (r, c) = ((2 * i) / ss, ((2 * i) % ss) / 2);
                            let v: u16 = match desen {
                                0 => rng.aralik(0, i32::from(max)) as u16,
                                1 => 0,
                                2 => max,
                                _ => {
                                    if (r + c).is_multiple_of(2) {
                                        0
                                    } else {
                                        max
                                    }
                                }
                            };
                            v.to_le_bytes()
                        })
                        .collect();
                    // Satır boşluklarına farklı değer (kenar ötesi okuma sonucu bozar).
                    for y in 0..h.saturating_sub(1) {
                        src[y * ss + 2 * src_w..(y + 1) * ss].fill(0x3c);
                    }
                    let n = (h - 1) * ds + 2 * dst_w + 32;
                    let (mut a, mut b) = (vec![NOBETCI; n], vec![NOBETCI; n]);
                    f(&mut a, ds, &src, ss, dst_w, h, src_w, dx as i32, mx0, bd);
                    sk(&mut b, ds, &src, ss, dst_w, h, src_w, dx as i32, mx0, bd);
                    let ctx = format!(
                        "{bd:?} resize dst_w={dst_w} src_w={src_w} dx={dx} mx0={mx0} h={h}"
                    );
                    assert_eq!(a, b, "{ctx}");
                    for y in 0..h {
                        let son = if y + 1 < h { (y + 1) * ds } else { n };
                        assert!(
                            a[y * ds + 2 * dst_w..son].iter().all(|&v| v == NOBETCI),
                            "nöbetçi {ctx}"
                        );
                    }
                    cagri += 1;
                }
            }
        }
        // Uç adımlar: birim (1 << 14), büyük küçültme (2 << 14), genel.
        for &(dst_w, src_w, dx) in &[
            (64usize, 64usize, 1i32 << 14),
            (40, 80, 2 << 14),
            (17, 5, 4800),
        ] {
            let src: Vec<u8> = (0..src_w)
                .flat_map(|_| (rng.aralik(0, i32::from(max)) as u16).to_le_bytes())
                .collect();
            let (mut a, mut b) = (vec![0u8; 2 * dst_w], vec![0u8; 2 * dst_w]);
            f(
                &mut a,
                2 * dst_w,
                &src,
                2 * src_w,
                dst_w,
                1,
                src_w,
                dx,
                777,
                bd,
            );
            sk(
                &mut b,
                2 * dst_w,
                &src,
                2 * src_w,
                dst_w,
                1,
                src_w,
                dx,
                777,
                bd,
            );
            assert_eq!(a, b, "{bd:?} resize uç adım dx={dx}");
            cagri += 1;
        }
    }
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrılar skalere devretmemeli"
    );
    println!("[P36] 16 bpc AVX512ICL büyütme: {cagri} çağrı skalerle bit-exact (10/12 bit)");
}

#[test]
fn p36_buyutme16_avx512_sozlesme_disi_skalere_devreder() {
    let Some(f) =
        avx512_yuvasi("buyutme16_avx512::p36_buyutme16_avx512_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let bd = ResizeBitDepth::B10;
    let sk = scalar_for(bd);
    let src: Vec<u8> = (0..32u16).flat_map(|v| (v * 31).to_le_bytes()).collect();
    let d0 = skaler_devir_sayisi();
    // Kaynak satır adımı genişlikten küçük (satırlar örtüşür): sözleşme dışı.
    let (mut a, mut b) = (vec![0u8; 2 * 32], vec![0u8; 2 * 32]);
    f(&mut a, 32, &src, 16, 16, 2, 16, 8192, 0, bd);
    sk(&mut b, 32, &src, 16, 16, 2, 16, 8192, 0, bd);
    assert_eq!(a, b);
    assert_eq!(
        skaler_devir_sayisi(),
        d0 + 1,
        "sözleşme dışı çağrı sayılmalı"
    );
}
