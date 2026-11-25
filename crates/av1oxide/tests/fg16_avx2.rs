//! P42 (F4-FG c, Adım 2) — 16 bpc (10/12 bit) AVX2 film grain gövdeleri.
//!
//! - **Eşitlik (`fgy`/`fguv`):** 10 ve 12 bpc × `pw` 1…255 (1…64 kısmi genişlikler
//!   ve çok bloklu genişlikler) × `bh` 1…32 (kromada `32 >> sy`'ye kadar) ×
//!   `row_num` 0, 1, 2, 7 × örtüşme × kırpma × `chroma_scaling_from_luma` ×
//!   `uv` × `is_id` × üç düzen; rastgele ve uç girdiler (piksel `0`/`bitdepth_max`,
//!   ölçekleme 0/255, grain `[-gctr, gctr - 1]`, `uv_*mult` −128/127, `uv_offset`
//!   −256/255, `scaling_shift` 8…11) için AVX2 çıktısı skaler gövdeyle bit-exact.
//! - **Sentinel ve `vpgatherdd` sınır güvenliği:** `src`/`dst` dilimleri
//!   `(bh − 1)·stride + pw·2` baytta, luma
//!   `((bh − 1) << sy)·luma_stride + (((pw − 1) << sx) + sx + 1)·2` baytta, ölçekleme
//!   tablosu tam 1024 (10 bpc) / 4096 (12 bpc) baytta (ön/arka 64 bayt nöbetçiyle
//!   çevrili dilim olarak) biter; `0` ve `1023`/`4095` uç dizinlerinde tablo dışına
//!   (`-2` veya `+3` bayt) okuma yapılmaz, `dst`'nin satır arası baytları nöbetçi
//!   (`0xA5`) kalır.
//! - **Eşitlik (`generate_grain_*`):** 10 ve 12 bpc × bütün `ar_coeff_lag` 0…3 ×
//!   `grain_scale_shift` 0…3 (12 bpc `grain_scale_shift == 0` → `shift == 0` dahil) ×
//!   `ar_coeff_shift` 6…9, uç ve rastgele katsayılar, `num_y_points` 0 / > 0,
//!   tohum 0 / 0xFFFF / rastgele; LUT'un bütün 73 × 82 değeri.
//! - **Devir sayacı:** sözleşmedeki çağrılarda skaler devir 0; `uv_mult > 127` (`fguv`),
//!   `bd = FgBitDepth::B8` (`fgy`) ya da `seed ≥ 65536` (`generate_grain`) skalere
//!   devreder, sayılır ve skalerle aynı çıktıyı verir.
//!
//! AVX2 yoksa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

mod common;

use av1oxide::bitstream::frame_header::FilmGrainData;
use av1oxide::dsp::fg::{
    EMPTY_LUT, FgBitDepth, FgDsp, FguvFn, FgyFn, GRAIN_HEIGHT, GRAIN_WIDTH, GenerateGrainUvFn,
    GenerateGrainYFn, GrainLut, fg_skaler_devir_sayisi,
};
use common::on_kosul::{self, Neden};

const NOBETCI: u8 = 0xA5;

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
        use av1oxide::dsp::x86::fg16;
        return Some(Avx2 {
            grain_y: fg16::generate_grain_y_fn(t),
            grain_uv: [0, 1, 2].map(|d| fg16::generate_grain_uv_fn(t, d)),
            fgy: fg16::fgy_fn(t),
            fguv: [0, 1, 2].map(|d| fg16::fguv_fn(t, d)),
        });
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P42] 16 bpc AVX2 film grain gövdeleri AVX2 ister",
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

fn lut(r: &mut Rng, bd: FgBitDepth, uc: bool) -> GrainLut {
    let mut l = EMPTY_LUT;
    let gctr = 128i32 << (bd.bits() - 8);
    let (lo, hi) = (-gctr, gctr - 1);
    for v in l.iter_mut().flatten() {
        *v = if uc {
            r.sec(&[lo, hi, 0, -1, 1]) as i16
        } else {
            r.aralik(lo, hi) as i16
        };
    }
    l
}

/// 16 bpc AVX2 ölçekleme tamponu: ön/arka 64 bayt nöbetçi (`0xFF` / `0x00` zıt değerli)
/// ile çevrili; çağrıya yalnız ortadaki tam `1024` (10 bpc) / `4096` (12 bpc) baytlık
/// dilim verilir (`vpgatherdd` `-2` veya `+3` bayt taşarsa nöbetçi bayt sonucu bozar).
fn olcekleme_tamponu(r: &mut Rng, bd: FgBitDepth, kip: u32) -> (Vec<u8>, usize) {
    let n = (bd.bitdepth_max() + 1) as usize;
    let nob = if kip == 0 { 0xFFu8 } else { 0x00u8 };
    let mut b = vec![nob; n + 128];
    for x in &mut b[64..64 + n] {
        *x = match kip {
            0 => 0,
            1 => 255,
            2 => r.sec(&[0, 255, 16, 235, 240]),
            _ => r.aralik(0, 255) as u8,
        };
    }
    (b, n)
}

fn pikseller_16(
    r: &mut Rng,
    pw: usize,
    bh: usize,
    stride: usize,
    bd: FgBitDepth,
    kip: u32,
) -> Vec<u8> {
    let max = bd.bitdepth_max();
    let shift = bd.bits() - 8;
    let n = (bh - 1) * stride + pw * 2;
    let mut b = vec![NOBETCI; n];
    for y in 0..bh {
        let satir = &mut b[y * stride..y * stride + pw * 2];
        for x in 0..pw {
            let px = match kip {
                0 => 0,
                1 => max,
                2 => r.sec(&[
                    0,
                    max,
                    16 << shift,
                    235 << shift,
                    240 << shift,
                    (16 << shift) - 1,
                    (235 << shift) + 1,
                ]),
                _ => r.aralik(0, max),
            } as u16;
            satir[x * 2..x * 2 + 2].copy_from_slice(&px.to_le_bytes());
        }
    }
    b
}

/// `dst` (nöbetçi dolu) ve kesin uzunluklu `src`: `(bh − 1)·stride + pw·2`.
fn tampon(
    r: &mut Rng,
    pw: usize,
    bh: usize,
    stride: usize,
    bd: FgBitDepth,
    kip: u32,
) -> (Vec<u8>, Vec<u8>) {
    let n = (bh - 1) * stride + pw * 2;
    (pikseller_16(r, pw, bh, stride, bd, kip), vec![NOBETCI; n])
}

const PW: [usize; 26] = [
    1, 2, 3, 7, 8, 9, 10, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49, 63, 64, 96, 128, 145,
    191, 255,
];

#[test]
fn p42_fgy16_fguv16_avx2_skalerle_bit_exact_ve_sentinel() {
    let Some(s) = avx2_yuvalari("fg16_avx2::p42_fgy16_fguv16_avx2_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = fg_skaler_devir_sayisi();
    let mut r = Rng(0x5042_0201);
    let mut sayi = [[0usize; 4]; 2];
    for (bdi, bd) in [FgBitDepth::B10, FgBitDepth::B12].into_iter().enumerate() {
        let sk = FgDsp::scalar_for(bd);
        for vaka in 0..1200 {
            let uc = vaka % 5 == 0;
            let kip = (vaka % 7) as u32;
            let (ov, clip) = (vaka % 2 == 1, vaka % 3 == 1);
            let pw = PW[vaka % PW.len()];
            let row_num = r.sec(&[0, 1, 2, 7]);
            let stride = (pw + r.sec(&[0, 1, 9])) * 2;
            let (sc_buf, sc_n) = olcekleme_tamponu(&mut r, bd, kip);
            let scaling = &sc_buf[64..64 + sc_n];
            let l = lut(&mut r, bd, uc);
            // fgy
            let bh = r.sec(&[1, 2, 3, 15, 16, 31, 32]);
            let d = veri(&mut r, ov, clip, false, uc);
            let (src, mut dst_s) = tampon(&mut r, pw, bh, stride, bd, kip);
            let mut dst_a = dst_s.clone();
            (sk.fgy_32x32xn)(
                &mut dst_s, &src, stride, &d, pw, scaling, &l, bh, row_num, bd,
            );
            (s.fgy)(
                &mut dst_a, &src, stride, &d, pw, scaling, &l, bh, row_num, bd,
            );
            assert_eq!(
                dst_a, dst_s,
                "fgy {bd:?} vaka {vaka}: pw {pw} bh {bh} stride {stride} row {row_num}"
            );
            sayi[bdi][3] += 1;
            // fguv: üç düzen
            for duzen in 0..3 {
                let (sx, sy) = [(1, 1), (1, 0), (0, 0)][duzen];
                let bh = (r.sec(&[1usize, 2, 3, 15, 16, 31, 32]) >> sy).max(1);
                let csfl = r.u64() & 1 == 1;
                let d = veri(&mut r, ov, clip, csfl, uc);
                let (uv, is_id) = ((r.u64() & 1) as usize, r.u64() & 1 == 1);
                let (src, mut dst_s) = tampon(&mut r, pw, bh, stride, bd, kip);
                let mut dst_a = dst_s.clone();
                let luma_w = ((pw - 1) << sx) + sx + 1;
                let luma_h = ((bh - 1) << sy) + 1;
                let ls = (luma_w + r.sec(&[0, 1, 13]) as usize) * 2;
                let luma = pikseller_16(&mut r, luma_w, luma_h, ls, bd, kip);
                let args = |f: FguvFn, dst: &mut Vec<u8>| {
                    f(
                        dst, &src, stride, &d, pw, scaling, &l, bh, row_num, &luma, ls, uv, is_id,
                        bd,
                    );
                };
                args(sk.fguv_32x32xn[duzen], &mut dst_s);
                args(s.fguv[duzen], &mut dst_a);
                assert_eq!(
                    dst_a, dst_s,
                    "fguv[{duzen}] {bd:?} vaka {vaka}: pw {pw} bh {bh} stride {stride} ls {ls} row {row_num} csfl {csfl}"
                );
                sayi[bdi][duzen] += 1;
            }
        }
    }
    assert_eq!(
        fg_skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrıda skaler devir olmamalı"
    );
    assert!(
        sayi.iter().flatten().all(|&n| n == 1200),
        "beklenen çağrı sayıları: {sayi:?}"
    );
}

#[test]
fn p42_grain16_avx2_skalerle_bit_exact() {
    let Some(s) = avx2_yuvalari("fg16_avx2::p42_grain16_avx2_skalerle_bit_exact") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = fg_skaler_devir_sayisi();
    let mut r = Rng(0x5042_0202);
    let mut sayi = 0;
    for bd in [FgBitDepth::B10, FgBitDepth::B12] {
        let sk = FgDsp::scalar_for(bd);
        for vaka in 0..320 {
            let uc = vaka % 4 == 0;
            let mut d = veri(&mut r, false, false, false, uc);
            d.ar_coeff_lag = vaka % 4;
            d.grain_scale_shift = (vaka / 4) % 4;
            d.ar_coeff_shift = 6 + ((vaka / 16) % 4) as u64;
            if vaka % 3 == 0 {
                d.num_y_points = 0;
            }
            let (mut ys, mut ya) = (EMPTY_LUT, EMPTY_LUT);
            (sk.generate_grain_y)(&mut ys, &d, bd);
            (s.grain_y)(&mut ya, &d, bd);
            assert_eq!(
                ya, ys,
                "grain_y {bd:?} vaka {vaka} lag={} gss={} acs={}",
                d.ar_coeff_lag, d.grain_scale_shift, d.ar_coeff_shift
            );
            // Kroma: luma girdisi gerçek luma grain'i ya da rastgele LUT.
            let by = if vaka % 2 == 0 {
                ys
            } else {
                lut(&mut r, bd, uc)
            };
            for duzen in 0..3 {
                for uv in 0..2 {
                    let (mut us, mut ua) = (EMPTY_LUT, EMPTY_LUT);
                    (sk.generate_grain_uv[duzen])(&mut us, &by, &d, uv, bd);
                    (s.grain_uv[duzen])(&mut ua, &by, &d, uv, bd);
                    assert_eq!(
                        ua, us,
                        "grain_uv[{duzen}] {bd:?} uv {uv} vaka {vaka} lag={} gss={}",
                        d.ar_coeff_lag, d.grain_scale_shift
                    );
                    sayi += 1;
                }
            }
        }
    }
    assert_eq!(fg_skaler_devir_sayisi(), devir0);
    assert_eq!(sayi, 2 * 320 * 6);
    assert_eq!(GRAIN_HEIGHT * GRAIN_WIDTH, 5986);
}

#[test]
fn p42_sozlesme_disi_skalere_devreder_ve_sayilir() {
    let Some(s) = avx2_yuvalari("fg16_avx2::p42_sozlesme_disi_skalere_devreder_ve_sayilir") else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for bd in [FgBitDepth::B10, FgBitDepth::B12] {
        let sk = FgDsp::scalar_for(bd);
        let mut r = Rng(0x5042_0203);
        let mut d = veri(&mut r, true, false, false, false);
        d.uv_mult[0] = 200; // (-128..=127) aralığı dışında: skalere devir.
        let l = lut(&mut r, bd, false);
        let (pw, bh, stride) = (32, 16, 64);
        let (src, mut dst_s) = tampon(&mut r, pw, bh, stride, bd, 3);
        let mut dst_a = dst_s.clone();
        let luma = pikseller_16(&mut r, 64, 32, 128, bd, 3);
        let (sc_buf, sc_n) = olcekleme_tamponu(&mut r, bd, 3);
        let scaling = &sc_buf[64..64 + sc_n];
        let devir0 = fg_skaler_devir_sayisi();
        (sk.fguv_32x32xn[0])(
            &mut dst_s, &src, stride, &d, pw, scaling, &l, bh, 1, &luma, 128, 0, false, bd,
        );
        (s.fguv[0])(
            &mut dst_a, &src, stride, &d, pw, scaling, &l, bh, 1, &luma, 128, 0, false, bd,
        );
        assert_eq!(dst_a, dst_s);
        assert_eq!(fg_skaler_devir_sayisi(), devir0 + 1);

        // `bd = FgBitDepth::B8` 16 bpc girişine verilirse `fgy16_sozlesmesi` reddeder ve skalere devreder.
        let mut dst_s2 = dst_s.clone();
        let mut dst_a2 = dst_a.clone();
        (FgDsp::scalar_for(FgBitDepth::B8).fgy_32x32xn)(
            &mut dst_s2,
            &src,
            stride,
            &d,
            pw,
            scaling,
            &l,
            bh,
            1,
            FgBitDepth::B8,
        );
        (s.fgy)(
            &mut dst_a2,
            &src,
            stride,
            &d,
            pw,
            scaling,
            &l,
            bh,
            1,
            FgBitDepth::B8,
        );
        assert_eq!(dst_a2, dst_s2);
        assert_eq!(fg_skaler_devir_sayisi(), devir0 + 2);

        // `seed ≥ 65536`: skalere devir (16 bitlik LFSR çözümü kapsam dışı).
        let mut d2 = d;
        d2.seed = 0x1_2345;
        d2.ar_coeff_lag = 2;
        let (mut ys, mut ya) = (EMPTY_LUT, EMPTY_LUT);
        (sk.generate_grain_y)(&mut ys, &d2, bd);
        (s.grain_y)(&mut ya, &d2, bd);
        assert_eq!(ya, ys);
        assert_eq!(fg_skaler_devir_sayisi(), devir0 + 3);
    }
}
