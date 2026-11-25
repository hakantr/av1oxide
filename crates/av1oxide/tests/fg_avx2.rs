//! P39 (F4-FG a) — 8 bpc AVX2 film grain gövdeleri.
//!
//! - **Eşitlik (`fgy`/`fguv`):** `pw` 1…255 (16'nın altı, 16, 17, 31, 32, 33 ve
//!   çok bloklu kısmi genişlikler) × `bh` 1…32 (kromada `32 >> sy`'ye kadar) ×
//!   `row_num` 0, 1, 2, 7 × örtüşme × kırpma × `chroma_scaling_from_luma` ×
//!   `uv` × `is_id` × üç düzen; rastgele ve uç girdiler (piksel 0/255, ölçekleme
//!   0/255, grain −128/127, `uv_*mult` −128/127, `uv_offset` −256/255,
//!   `scaling_shift` 8…11) için AVX2 çıktısı skaler gövdeyle bit-exact.
//! - **Sentinel:** `src`/`dst` dilimleri `(bh − 1)·stride + pw` baytta, luma
//!   `((bh − 1) << sy)·luma_stride + ((pw − 1) << sx) + sx + 1` baytta biter
//!   (dilim dışı okuma panik olurdu); `dst`'nin satır arası baytları nöbetçi
//!   kalır (yazma 0).
//! - **Eşitlik (`generate_grain_*`):** bütün `ar_coeff_lag`, `grain_scale_shift`,
//!   `ar_coeff_shift` 6…9, uç ve rastgele katsayılar, `num_y_points` 0 / > 0,
//!   tohum 0 / 0xFFFF / rastgele; LUT'un bütün 73 × 82 değeri.
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; ölçekleme
//!   tablosu 259 bayttan kısa (`fgy`/`fguv`) ya da `seed ≥ 65536` (`generate_grain`)
//!   skalere devreder, sayılır ve skalerle aynı çıktıyı verir.
//!
//! AVX2 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

mod common;

use av1oxide::bitstream::frame_header::FilmGrainData;
use av1oxide::dsp::fg::{
    EMPTY_LUT, FgBitDepth, FgDsp, FguvFn, FgyFn, GRAIN_HEIGHT, GRAIN_WIDTH, GenerateGrainUvFn,
    GenerateGrainYFn, GrainLut, SCALING_SIZE_MAX, fg_skaler_devir_sayisi,
};
use common::on_kosul::{self, Neden};

const NOBETCI: u8 = 0xA5;
const BD: FgBitDepth = FgBitDepth::B8;

/// Sayaç denetimleri aynı süreçte koşan testlerden yalıtılır.
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
    fn sec<T: Copy>(&mut self, v: &[T]) -> T {
        v[(self.u64() >> 33) as usize % v.len()]
    }
}

struct Avx2 {
    grain_y: GenerateGrainYFn,
    grain_uv: [GenerateGrainUvFn; 3],
    fgy: FgyFn,
    fguv: [FguvFn; 3],
}

fn avx2_yuvalari(test: &str) -> Option<Avx2> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        use av1oxide::dsp::x86::fg;
        return Some(Avx2 {
            grain_y: fg::generate_grain_y_fn(t),
            grain_uv: [0, 1, 2].map(|d| fg::generate_grain_uv_fn(t, d)),
            fgy: fg::fgy_fn(t),
            fguv: [0, 1, 2].map(|d| fg::fguv_fn(t, d)),
        });
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P39] 8 bpc AVX2 film grain gövdeleri AVX2 ister",
    );
    None
}

/// Rastgele film grain parametreleri (`uc`: uç değerler ağırlıklı).
fn veri(r: &mut Rng, ov: bool, clip: bool, csfl: bool, uc: bool) -> FilmGrainData {
    let rastgele = r.aralik(0, 0xFFFF) as u32;
    let mut d = FilmGrainData {
        seed: r.sec(&[0, 0xFFFF, rastgele]),
        num_y_points: r.aralik(1, 14),
        chroma_scaling_from_luma: csfl,
        scaling_shift: r.aralik(8, 11),
        ar_coeff_lag: r.aralik(0, 3),
        ar_coeff_shift: r.aralik(6, 9) as u64,
        grain_scale_shift: r.aralik(0, 3),
        overlap_flag: ov,
        clip_to_restricted_range: clip,
        ..FilmGrainData::default()
    };
    let k = |r: &mut Rng| {
        if uc {
            r.sec(&[-128, 127, 0, -1, 1]) as i8
        } else {
            r.aralik(-128, 127) as i8
        }
    };
    for v in &mut d.ar_coeffs_y {
        *v = k(r);
    }
    for p in 0..2 {
        for v in &mut d.ar_coeffs_uv[p][..25] {
            *v = k(r);
        }
        d.uv_mult[p] = if uc {
            r.sec(&[-128, 127])
        } else {
            r.aralik(-128, 127)
        };
        d.uv_luma_mult[p] = if uc {
            r.sec(&[-128, 127])
        } else {
            r.aralik(-128, 127)
        };
        d.uv_offset[p] = if uc {
            r.sec(&[-256, 255])
        } else {
            r.aralik(-256, 255)
        };
    }
    d
}

fn lut(r: &mut Rng, uc: bool) -> GrainLut {
    let mut l = EMPTY_LUT;
    for v in l.iter_mut().flatten() {
        *v = if uc {
            r.sec(&[-128, 127])
        } else {
            r.aralik(-128, 127) as i16
        };
    }
    l
}

fn baytlar(r: &mut Rng, n: usize, kip: u32) -> Vec<u8> {
    (0..n)
        .map(|_| match kip {
            0 => 0,
            1 => 255,
            2 => r.sec(&[0, 255, 16, 235, 240]),
            _ => r.aralik(0, 255) as u8,
        })
        .collect()
}

fn olcekleme(r: &mut Rng, kip: u32) -> Vec<u8> {
    let mut v = baytlar(r, 256, kip);
    v.resize(SCALING_SIZE_MAX, 0);
    v
}

/// `dst` (nöbetçi dolu) ve kesin uzunluklu `src`: `(bh − 1)·stride + pw`.
fn tampon(r: &mut Rng, pw: usize, bh: usize, stride: usize, kip: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (bh - 1) * stride + pw;
    (baytlar(r, n, kip), vec![NOBETCI; n])
}

const PW: [usize; 22] = [
    1, 2, 7, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49, 63, 64, 65, 96, 128, 145, 160, 191, 255,
];

#[test]
fn p39_fgy_fguv_avx2_skalerle_bit_exact_ve_sentinel() {
    let Some(a) = avx2_yuvalari("fg_avx2::p39_fgy_fguv_avx2_skalerle_bit_exact_ve_sentinel") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = FgDsp::scalar_for(BD);
    let devir0 = fg_skaler_devir_sayisi();
    let mut r = Rng(0x5039_0001);
    let mut sayi = [0usize; 4];
    for vaka in 0..2400 {
        let uc = vaka % 5 == 0;
        let kip = (vaka % 7) as u32;
        let (ov, clip) = (vaka % 2 == 1, vaka % 3 == 1);
        let pw = PW[vaka % PW.len()];
        let row_num = r.sec(&[0, 1, 2, 7]);
        let stride = pw + r.sec(&[0, 1, 9]);
        let scaling = olcekleme(&mut r, kip);
        let l = lut(&mut r, uc);
        // fgy
        let bh = r.sec(&[1, 2, 3, 15, 16, 31, 32]);
        let d = veri(&mut r, ov, clip, false, uc);
        let (src, mut dst_s) = tampon(&mut r, pw, bh, stride, kip);
        let mut dst_a = dst_s.clone();
        (sk.fgy_32x32xn)(
            &mut dst_s, &src, stride, &d, pw, &scaling, &l, bh, row_num, BD,
        );
        (a.fgy)(
            &mut dst_a, &src, stride, &d, pw, &scaling, &l, bh, row_num, BD,
        );
        assert_eq!(
            dst_a, dst_s,
            "fgy vaka {vaka}: pw {pw} bh {bh} stride {stride} row {row_num}"
        );
        sayi[3] += 1;
        // fguv: üç düzen
        for duzen in 0..3 {
            let (sx, sy) = [(1, 1), (1, 0), (0, 0)][duzen];
            let bh = (r.sec(&[1usize, 2, 3, 15, 16, 31, 32]) >> sy).max(1);
            let csfl = r.u64() & 1 == 1;
            let d = veri(&mut r, ov, clip, csfl, uc);
            let (uv, is_id) = ((r.u64() & 1) as usize, r.u64() & 1 == 1);
            let (src, mut dst_s) = tampon(&mut r, pw, bh, stride, kip);
            let mut dst_a = dst_s.clone();
            let luma_son = ((pw - 1) << sx) + sx + 1;
            let ls = luma_son + r.sec(&[0, 1, 13]);
            let luma = baytlar(&mut r, ((bh - 1) << sy) * ls + luma_son, kip);
            let args = |f: FguvFn, dst: &mut Vec<u8>| {
                f(
                    dst, &src, stride, &d, pw, &scaling, &l, bh, row_num, &luma, ls, uv, is_id, BD,
                );
            };
            args(sk.fguv_32x32xn[duzen], &mut dst_s);
            args(a.fguv[duzen], &mut dst_a);
            assert_eq!(
                dst_a, dst_s,
                "fguv[{duzen}] vaka {vaka}: pw {pw} bh {bh} stride {stride} ls {ls} row {row_num} csfl {csfl}"
            );
            sayi[duzen] += 1;
        }
    }
    assert_eq!(
        fg_skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrıda skaler devir olmamalı"
    );
    assert!(sayi.iter().all(|&n| n == 2400), "{sayi:?}");
}

#[test]
fn p39_grain_avx2_skalerle_bit_exact() {
    let Some(a) = avx2_yuvalari("fg_avx2::p39_grain_avx2_skalerle_bit_exact") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = FgDsp::scalar_for(BD);
    let devir0 = fg_skaler_devir_sayisi();
    let mut r = Rng(0x5039_0002);
    let mut sayi = 0;
    for vaka in 0..480 {
        let uc = vaka % 4 == 0;
        let mut d = veri(&mut r, false, false, false, uc);
        d.ar_coeff_lag = vaka % 4;
        d.grain_scale_shift = (vaka / 4) % 4;
        d.ar_coeff_shift = 6 + ((vaka / 16) % 4) as u64;
        if vaka % 3 == 0 {
            d.num_y_points = 0;
        }
        let (mut ys, mut ya) = (EMPTY_LUT, EMPTY_LUT);
        (sk.generate_grain_y)(&mut ys, &d, BD);
        (a.grain_y)(&mut ya, &d, BD);
        assert_eq!(ya, ys, "grain_y vaka {vaka}");
        // Kroma: luma girdisi gerçek luma grain'i ya da rastgele LUT.
        let by = if vaka % 2 == 0 { ys } else { lut(&mut r, uc) };
        for duzen in 0..3 {
            for uv in 0..2 {
                let (mut us, mut ua) = (EMPTY_LUT, EMPTY_LUT);
                (sk.generate_grain_uv[duzen])(&mut us, &by, &d, uv, BD);
                (a.grain_uv[duzen])(&mut ua, &by, &d, uv, BD);
                assert_eq!(ua, us, "grain_uv[{duzen}] uv {uv} vaka {vaka}");
                sayi += 1;
            }
        }
    }
    assert_eq!(fg_skaler_devir_sayisi(), devir0);
    assert_eq!(sayi, 480 * 6);
    assert_eq!(GRAIN_HEIGHT * GRAIN_WIDTH, 5986);
}

#[test]
fn p39_sozlesme_disi_skalere_devreder_ve_sayilir() {
    let Some(a) = avx2_yuvalari("fg_avx2::p39_sozlesme_disi_skalere_devreder_ve_sayilir") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sk = FgDsp::scalar_for(BD);
    let mut r = Rng(0x5039_0003);
    let d = veri(&mut r, true, false, false, false);
    let l = lut(&mut r, false);
    let (pw, bh, stride) = (64, 32, 64);
    let (src, mut dst_s) = tampon(&mut r, pw, bh, stride, 3);
    let mut dst_a = dst_s.clone();
    // Ölçekleme tablosu 256 bayt (< 259): skalere devir.
    let kisa: Vec<u8> = baytlar(&mut r, 256, 3);
    let devir0 = fg_skaler_devir_sayisi();
    (sk.fgy_32x32xn)(&mut dst_s, &src, stride, &d, pw, &kisa, &l, bh, 1, BD);
    (a.fgy)(&mut dst_a, &src, stride, &d, pw, &kisa, &l, bh, 1, BD);
    assert_eq!(dst_a, dst_s);
    assert_eq!(fg_skaler_devir_sayisi(), devir0 + 1);
    // `seed ≥ 65536`: skalere devir (16 bitlik LFSR çözümü kapsam dışı).
    let mut d2 = d;
    d2.seed = 0x1_2345;
    d2.ar_coeff_lag = 2;
    let (mut ys, mut ya) = (EMPTY_LUT, EMPTY_LUT);
    (sk.generate_grain_y)(&mut ys, &d2, BD);
    (a.grain_y)(&mut ya, &d2, BD);
    assert_eq!(ya, ys);
    assert_eq!(fg_skaler_devir_sayisi(), devir0 + 2);
}
