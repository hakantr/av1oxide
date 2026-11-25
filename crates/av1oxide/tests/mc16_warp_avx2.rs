//! P34 (F4-MC c) — 16 bpc (10/12 bit) AVX2 warp gövdeleri: `warp8x8`,
//! `warp8x8t`.
//!
//! - **Eşitlik:** 10 ve 12 bitte rastgele `abcd` ∈ [−4096, 4096] ve filtre
//!   indeksini köşelerde [0, 192]'de tutan `mx`/`my` (alt uç, üst uç, iç
//!   nokta) × kaynak satır adımı {30, 40, 94} bayt × piksel deseni (rastgele,
//!   0, `bitdepth_max`, 0/`bitdepth_max` dama) için AVX2 çıktısı skaler
//!   gövdeyle bit-exact.
//! - **Sentinel:** kaynak dilimi okuma bölgesi (`15 × 15` piksel) +
//!   `MC_KUYRUK`'ta biter; hedefte satır aralarına ve sona nöbetçi konur
//!   (8 piksel ötesine yazma 0).
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; kuyruksuz dilim
//!   ve indeksi aralık dışına çıkaran parametre skalere devreder ve sayılır.
//!
//! AVX2 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::{MC_KUYRUK, McDsp, Warp8x8Fn, Warp8x8tFn, skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

const NOBETCI: u8 = 0xA5;
const NOBETCI16: i16 = 0x5A5A;
/// Filtre indeksi `[0, 192]` ⇔ `t` ∈ `[T_ALT, T_UST]`.
const T_ALT: i32 = -66048;
const T_UST: i32 = 131_583;

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

fn avx2_yuvalari(test: &str) -> Option<(Warp8x8Fn, Warp8x8tFn)> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        use av1oxide::dsp::x86::mc16_warp as wp;
        return Some((wp::warp8x8_fn(t), wp::warp8x8t_fn(t)));
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P34] 16 bpc AVX2 warp gövdeleri AVX2 ister",
    );
    None
}

/// `t0 + r·ks + x·kx` köşelerde aralıkta kalacak biçimde `t0` (kip 0 alt uç,
/// 1 üst uç, 2 iç nokta).
fn baslangic(rng: &mut Rng, ks: i32, kx: i32, son_satir: i32, kip: u32) -> i32 {
    let omin = son_satir * ks.min(0) + 7 * kx.min(0);
    let omax = son_satir * ks.max(0) + 7 * kx.max(0);
    let (lo, hi) = (T_ALT - omin, T_UST - omax);
    match kip {
        0 => lo,
        1 => hi,
        _ => rng.aralik(lo, hi),
    }
}

/// Kaynak: `14·ss + 30 + MC_KUYRUK` bayt, `u16` pikseller (satır boşlukları da
/// desenle dolu).
fn kaynak(rng: &mut Rng, desen: u8, ss: usize, max: u16) -> Vec<u8> {
    let n = 14 * ss + 30 + MC_KUYRUK;
    let mut v = vec![0u8; n];
    for i in 0..n / 2 {
        let (r, c) = ((2 * i) / ss, ((2 * i) % ss) / 2);
        let p: u16 = match desen {
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
        v[2 * i..2 * i + 2].copy_from_slice(&p.to_ne_bytes());
    }
    v
}

#[test]
fn p34_warp16_avx2_skalerle_bit_exact_ve_sentinel() {
    let Some((w8, wt)) =
        avx2_yuvalari("mc16_warp_avx2::p34_warp16_avx2_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x005e_ed0f_3a3a_1634);
    let mut cagri = 0usize;
    for (bd, max) in [(BitDepth::Ten, 1023u16), (BitDepth::Twelve, 4095)] {
        let sk = McDsp::scalar_for(bd);
        for ps in 0..90u32 {
            let abcd: [i16; 4] = core::array::from_fn(|_| rng.aralik(-4096, 4096) as i16);
            let mx = baslangic(&mut rng, i32::from(abcd[1]), i32::from(abcd[0]), 14, ps % 3);
            let my = baslangic(
                &mut rng,
                i32::from(abcd[3]),
                i32::from(abcd[2]),
                7,
                (ps / 3) % 3,
            );
            // (kaynak satır adımı bayt, hedef satır adımı: warp8x8 bayt, warp8x8t öğe)
            for &(ss, ds, ts) in &[(30usize, 16usize, 8usize), (40, 26, 13), (94, 80, 40)] {
                for desen in 0..4u8 {
                    let src = kaynak(&mut rng, desen, ss, max);
                    let ctx = format!("{bd:?} abcd={abcd:?} mx={mx} my={my} ss={ss} desen={desen}");
                    // warp8x8: hedef 8 satır × 16 bayt, satır arası ve son nöbetçi.
                    let n = 7 * ds + 16 + 32;
                    let (mut a, mut b) = (vec![NOBETCI; n], vec![NOBETCI; n]);
                    w8(&mut a, ds, &src, ss, &abcd, mx, my, bd, max);
                    (sk.warp8x8)(&mut b, ds, &src, ss, &abcd, mx, my, bd, max);
                    assert_eq!(a, b, "warp8x8 {ctx}");
                    for y in 0..8 {
                        let son = if y < 7 { (y + 1) * ds } else { n };
                        assert!(
                            a[y * ds + 16..son].iter().all(|&v| v == NOBETCI),
                            "warp8x8 nöbetçi {ctx}"
                        );
                    }
                    // warp8x8t
                    let n = 7 * ts + 8 + 16;
                    let (mut a, mut b) = (vec![NOBETCI16; n], vec![NOBETCI16; n]);
                    wt(&mut a, ts, &src, ss, &abcd, mx, my, bd, max);
                    (sk.warp8x8t)(&mut b, ts, &src, ss, &abcd, mx, my, bd, max);
                    assert_eq!(a, b, "warp8x8t {ctx}");
                    for y in 0..8 {
                        let son = if y < 7 { (y + 1) * ts } else { n };
                        assert!(
                            a[y * ts + 8..son].iter().all(|&v| v == NOBETCI16),
                            "warp8x8t nöbetçi {ctx}"
                        );
                    }
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
    println!("[P34] 16 bpc AVX2 warp: {cagri} çağrı skalerle bit-exact (2 yuva × 10/12 bit)");
}

#[test]
fn p34_warp16_sozlesme_disi_skalere_devreder() {
    let Some((w8, wt)) = avx2_yuvalari("mc16_warp_avx2::p34_warp16_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (bd, max) = (BitDepth::Twelve, 4095u16);
    let sk = McDsp::scalar_for(bd);
    let mut rng = Rng(3);
    let (ss, ds) = (30usize, 16usize);
    let src = kaynak(&mut rng, 0, ss, max);
    let d0 = skaler_devir_sayisi();
    // Kuyruksuz dilim (bölge tam, `MC_KUYRUK` yok): skalere devreder.
    let kisa = &src[..14 * ss + 30];
    let abcd = [0i16; 4];
    let (mut a, mut b) = (vec![0u8; 7 * ds + 16], vec![0u8; 7 * ds + 16]);
    w8(&mut a, ds, kisa, ss, &abcd, 0, 0, bd, max);
    (sk.warp8x8)(&mut b, ds, kisa, ss, &abcd, 0, 0, bd, max);
    assert_eq!(a, b);
    // `warp8x8t` hedef satır adımı 7 öğe (< 8): sözleşme dışı; skaler gövde
    // örtüşen satırları sırayla yazar.
    let (mut a, mut b) = (vec![0i16; 7 * 7 + 8], vec![0i16; 7 * 7 + 8]);
    wt(&mut a, 7, &src, ss, &abcd, 0, 0, bd, max);
    (sk.warp8x8t)(&mut b, 7, &src, ss, &abcd, 0, 0, bd, max);
    assert_eq!(a, b);
    assert_eq!(
        skaler_devir_sayisi(),
        d0 + 2,
        "iki sözleşme dışı çağrı sayılmalı"
    );
}
