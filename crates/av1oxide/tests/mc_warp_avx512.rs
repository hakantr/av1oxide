//! P36 (F4-MC e) — 8 bpc AVX512ICL warp gövdeleri: `warp8x8`, `warp8x8t`.
//!
//! - **Eşitlik:** rastgele `abcd` ∈ [−4096, 4096] ve filtre indeksini köşelerde
//!   [0, 192]'de tutan `mx`/`my` (alt uç, üst uç, iç nokta) × kaynak satır
//!   adımı {15, 20, 47} × piksel deseni (rastgele, 0, 255, 0/255 dama) için
//!   AVX512ICL çıktısı skaler gövdeyle bit-exact.
//! - **Sentinel:** kaynak dilimi okuma bölgesi (`15 × 15`) + `MC_KUYRUK`'ta
//!   biter; hedefte satır aralarına ve sona nöbetçi konur (`8` ötesine yazma 0).
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; indeksi
//!   aralık dışına çıkaran parametre ve kısa dilim skalere devreder ve sayılır.
//!
//! AVX512ICL yoksa testler ön koşul ilkesiyle atlanır.

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

fn avx512_yuvalari(test: &str) -> Option<(Warp8x8Fn, Warp8x8tFn)> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx512icl_token()
    {
        use av1oxide::dsp::x86::mc_warp_avx512 as wp;
        return Some((wp::warp8x8_fn(t), wp::warp8x8t_fn(t)));
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P36] 8 bpc AVX512ICL warp gövdeleri AVX512ICL ister",
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

fn kaynak(rng: &mut Rng, desen: u8, ss: usize) -> Vec<u8> {
    let n = 14 * ss + 15 + MC_KUYRUK;
    (0..n)
        .map(|i| match desen {
            0 => (rng.u64() >> 40) as u8,
            1 => 0,
            2 => 255,
            _ => {
                if (i / ss + i % ss).is_multiple_of(2) {
                    0
                } else {
                    255
                }
            }
        })
        .collect()
}

#[test]
fn p36_warp_avx512_skalerle_bit_exact_ve_sentinel() {
    let Some((w8, wt)) =
        avx512_yuvalari("mc_warp_avx512::p36_warp_avx512_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x005e_ed0f_3a3a_1234);
    let bd = BitDepth::Eight;
    let mut cagri = 0usize;
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
        for &(ss, ds) in &[(15usize, 8usize), (20, 13), (47, 40)] {
            for desen in 0..4u8 {
                let src = kaynak(&mut rng, desen, ss);
                // warp8x8: hedef 8 satır, satır arası ve son nöbetçi.
                let n = 7 * ds + 8 + 16;
                let (mut a, mut b) = (vec![NOBETCI; n], vec![NOBETCI; n]);
                w8(&mut a, ds, &src, ss, &abcd, mx, my, bd, 255);
                (sk.warp8x8)(&mut b, ds, &src, ss, &abcd, mx, my, bd, 255);
                assert_eq!(a, b, "warp8x8 abcd={abcd:?} mx={mx} my={my} ss={ss}");
                for y in 0..8 {
                    let son = if y < 7 { (y + 1) * ds } else { n };
                    assert!(
                        a[y * ds + 8..son].iter().all(|&v| v == NOBETCI),
                        "warp8x8 nöbetçi"
                    );
                }
                // warp8x8t
                let n = 7 * ds + 8 + 16;
                let (mut a, mut b) = (vec![NOBETCI16; n], vec![NOBETCI16; n]);
                wt(&mut a, ds, &src, ss, &abcd, mx, my, bd, 255);
                (sk.warp8x8t)(&mut b, ds, &src, ss, &abcd, mx, my, bd, 255);
                assert_eq!(a, b, "warp8x8t abcd={abcd:?} mx={mx} my={my} ss={ss}");
                for y in 0..8 {
                    let son = if y < 7 { (y + 1) * ds } else { n };
                    assert!(
                        a[y * ds + 8..son].iter().all(|&v| v == NOBETCI16),
                        "warp8x8t nöbetçi"
                    );
                }
                cagri += 2;
            }
        }
    }
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrılar skalere devretmemeli"
    );
    println!("[P36] 8 bpc AVX512ICL warp: {cagri} çağrı skalerle bit-exact (2 yuva)");
}

#[test]
fn p36_warp_avx512_sozlesme_disi_skalere_devreder() {
    let Some((w8, _)) =
        avx512_yuvalari("mc_warp_avx512::p36_warp_avx512_sozlesme_disi_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = McDsp::scalar_for(BitDepth::Eight);
    let mut rng = Rng(3);
    let bd = BitDepth::Eight;
    let (ss, ds) = (15usize, 8usize);
    let src = kaynak(&mut rng, 0, ss);
    let d0 = skaler_devir_sayisi();
    // Kuyruksuz dilim (bölge tam, `MC_KUYRUK` yok): skalere devreder.
    let kisa = &src[..14 * ss + 15];
    let abcd = [0i16; 4];
    let (mut a, mut b) = (vec![0u8; 7 * ds + 8], vec![0u8; 7 * ds + 8]);
    w8(&mut a, ds, kisa, ss, &abcd, 0, 0, bd, 255);
    (sk.warp8x8)(&mut b, ds, kisa, ss, &abcd, 0, 0, bd, 255);
    assert_eq!(a, b);
    assert_eq!(skaler_devir_sayisi(), d0 + 1, "kuyruksuz dilim sayılmalı");
}
