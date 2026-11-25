//! ARM (`aarch64` NEON / DOTPROD) SIMD aktarım sözlüğünün derleme ve lane davranışı
//! testleri (`P54` / `F5d`, `SMD-03` / `SMD-04`).
//!
//! İki parça:
//! 1. **Derleme ve belge senkronu:** `dsp/arm/` üretim çekirdeklerinde ve sözlükte yer alan
//!    tüm kararlı `core::arch::aarch64` vektör türleri ve intrinsic işlevleri (`209` sembol)
//!    `rustc 1.100.0` altında içe aktarılıp derlenir; [`arm_sozluk_belgesi_intrinsicleri_kapsar`]
//!    testi bu adların tamamının `docs/simd-sozluk/arm.md` içinde kayıtlı olduğunu doğrular.
//! 2. **Negatif denetimli lane ve semantik davranış testleri:** `dav1d` AArch64 asm kodunun
//!    dayandığı ve x86 SSE/AVX semantiğinden ayrıştığı 9 kritik tuzak (`vqtbl1q_u8` `>= 16`
//!    sıfırlaması, `vextq_u8` ters argüman sırası, `vtrn1q_s32`/`vtrn2q_s32` çift/tek şerit
//!    devriği, `vuzp1q_u8` 128-bit tam ayırma, `vzip1q_u8` alt yarı serpiştirme,
//!    `vqrdmulhq_s16` `-32768 * -32768 -> +32767` doyumu, `vrshlq_s32` negatif üs yuvarlamalı
//!    kaydırma, `vqrshrun_n_s16` yuvarlama + işaretsiz daraltma doyumu ve `DotprodToken`
//!    kapılı `vdotq_s32`/`vdotq_u32` 4'lü grup iç çarpımı) 520 belirlenimci + uç girdide
//!    hem doğru skaler referansla hem de tuzağı yok sayan yanlış referansla (`ayrisan > 0`)
//!    sınanır.

#![forbid(unsafe_code)]

use core::arch::aarch64::{
    int8x16_t, int16x8_t, int32x4_t, uint8x16_t, uint32x4_t, vcombine_u8, vdotq_s32, vdotq_u32,
    vdup_n_u8, vextq_u8, vqrdmulhq_s16, vqrshrun_n_s16, vqtbl1q_u8, vreinterpretq_s8_u8,
    vreinterpretq_s16_u8, vreinterpretq_s32_u8, vreinterpretq_u8_s16, vreinterpretq_u8_s32,
    vreinterpretq_u8_u32, vreinterpretq_u32_u8, vrshlq_s32, vtrn1q_s32, vtrn2q_s32, vuzp1q_u8,
    vzip1q_u8,
};

use super::mem::{load_u8x16, store_u8x16};
use crate::cpu::{CpuFlags, DotprodToken, NeonToken};

macro_rules! arm_sozluk_sembol_listesi {
    ($($ad:ident),* $(,)?) => {
        mod derleme_kaniti {
            #[allow(unused_imports)]
            use core::arch::aarch64::{$($ad),*};
            pub(super) const ADLAR: &[&str] = &[$(stringify!($ad)),*];
        }
    };
}

arm_sozluk_sembol_listesi!(
    int8x16_t,
    int16x4_t,
    int16x8_t,
    int32x4_t,
    uint8x8_t,
    uint8x16_t,
    uint16x4_t,
    uint16x8_t,
    uint32x4_t,
    vabdq_s16,
    vabdq_s32,
    vabs_s16,
    vabsq_s32,
    vadd_u16,
    vaddl_high_s16,
    vaddl_s16,
    vaddlvq_s16,
    vaddlvq_u8,
    vaddlvq_u16,
    vaddq_s16,
    vaddq_s32,
    vaddq_u8,
    vaddq_u16,
    vaddq_u32,
    vaddvq_s16,
    vaddvq_s32,
    vand_u16,
    vandq_s32,
    vandq_u8,
    vandq_u16,
    vandq_u32,
    vbicq_u32,
    vbslq_s16,
    vbslq_s32,
    vcge_u16,
    vcgeq_u16,
    vcgtq_s32,
    vcleq_s16,
    vcleq_s32,
    vcltq_s32,
    vcombine_s16,
    vcombine_u8,
    vcombine_u16,
    vcreate_u16,
    vdotq_s32,
    vdotq_u32,
    vdup_n_u8,
    vdup_n_u16,
    vdup_n_u32,
    vdup_n_u64,
    vdupq_n_s16,
    vdupq_n_s32,
    vdupq_n_u8,
    vdupq_n_u16,
    vdupq_n_u32,
    veorq_u8,
    vextq_s16,
    vextq_u8,
    vextq_u16,
    vget_high_s16,
    vget_high_u8,
    vget_high_u16,
    vget_lane_u16,
    vget_lane_u32,
    vget_lane_u64,
    vget_low_s16,
    vget_low_u8,
    vget_low_u16,
    vld1_s16,
    vld1_u8,
    vld1_u16,
    vld1q_s16,
    vld1q_s32,
    vld1q_u8,
    vld1q_u16,
    vmaxq_s16,
    vmaxq_s32,
    vmaxvq_u8,
    vmaxvq_u32,
    vminq_s16,
    vminq_s32,
    vminq_u16,
    vminq_u32,
    vmlal_high_n_s16,
    vmlal_high_n_u16,
    vmlal_high_s16,
    vmlal_high_u16,
    vmlal_n_s16,
    vmlal_n_u16,
    vmlal_s16,
    vmlal_u8,
    vmlal_u16,
    vmlaq_n_s16,
    vmlaq_n_u16,
    vmovl_high_s16,
    vmovl_high_u8,
    vmovl_s16,
    vmovl_u8,
    vmovl_u16,
    vmovn_high_u32,
    vmovn_s32,
    vmovn_u16,
    vmovn_u32,
    vmull_high_n_s16,
    vmull_high_n_u16,
    vmull_high_s16,
    vmull_high_u16,
    vmull_n_s16,
    vmull_n_u16,
    vmull_s16,
    vmull_u8,
    vmull_u16,
    vmulq_n_s16,
    vmulq_n_s32,
    vmulq_n_u16,
    vmulq_n_u32,
    vmulq_s32,
    vmulq_u32,
    vnegq_s32,
    vorr_u8,
    vorrq_u32,
    vpaddlq_u8,
    vpaddlq_u16,
    vpaddq_u16,
    vqmovn_u16,
    vqmovn_u32,
    vqmovun_s32,
    vqrdmulhq_s16,
    vqrshrun_n_s16,
    vqrshrun_n_s32,
    vqsubq_u8,
    vqtbl1q_s8,
    vqtbl1q_u8,
    vreinterpret_s16_u16,
    vreinterpret_s16_u64,
    vreinterpret_u8_u32,
    vreinterpret_u16_s16,
    vreinterpret_u16_u8,
    vreinterpret_u32_u8,
    vreinterpret_u64_u8,
    vreinterpret_u64_u16,
    vreinterpretq_s8_u8,
    vreinterpretq_s16_u8,
    vreinterpretq_s16_u16,
    vreinterpretq_s32_s64,
    vreinterpretq_s32_u8,
    vreinterpretq_s32_u32,
    vreinterpretq_s64_s32,
    vreinterpretq_u8_s16,
    vreinterpretq_u8_s32,
    vreinterpretq_u8_u16,
    vreinterpretq_u8_u32,
    vreinterpretq_u16_s16,
    vreinterpretq_u16_u8,
    vreinterpretq_u32_s32,
    vreinterpretq_u32_u8,
    vrhadd_u8,
    vrhaddq_u16,
    vrshlq_s32,
    vrshlq_u16,
    vrshlq_u32,
    vrshrn_high_n_u32,
    vrshrn_n_s32,
    vrshrn_n_u16,
    vrshrn_n_u32,
    vrshrq_n_s16,
    vrshrq_n_s32,
    vrshrq_n_u16,
    vrshrq_n_u32,
    vshl_n_u8,
    vshlq_n_s16,
    vshlq_n_s32,
    vshlq_n_u8,
    vshlq_s16,
    vshlq_s32,
    vshlq_u16,
    vshr_n_u16,
    vshrn_high_n_u32,
    vshrn_n_u16,
    vshrn_n_u32,
    vshrq_n_s32,
    vshrq_n_u8,
    vshrq_n_u16,
    vshrq_n_u32,
    vst1_u8,
    vst1_u16,
    vst1q_s16,
    vst1q_s32,
    vst1q_u8,
    vst1q_u16,
    vsub_u8,
    vsubl_high_s16,
    vsubl_s16,
    vsubl_u8,
    vsubq_s16,
    vsubq_s32,
    vsubq_u16,
    vtbl1_u8,
    vtbl2_u8,
    vtrn1q_s32,
    vtrn1q_s64,
    vtrn2q_s32,
    vtrn2q_s64,
    vuzp1_u8,
    vuzp1q_s16,
    vuzp1q_u8,
    vuzp1q_u16,
    vuzp2_u8,
    vuzp2q_s16,
    vuzp2q_u8,
    vuzp2q_u16,
    vzip1q_u8,
);

#[test]
fn arm_sozluk_intrinsicleri_tutarli() {
    assert!(
        derleme_kaniti::ADLAR.len() >= 200,
        "ARM sözlük sembol listesi beklenenden küçük: {}",
        derleme_kaniti::ADLAR.len()
    );
}

pub(super) type V16 = [u8; 16];
type Ref16<'a> = &'a dyn Fn(&V16, &V16) -> V16;

#[derive(Clone, Copy, Debug)]
pub(super) enum NeonKip {
    Qtbl1U8,
    ExtU8(u8),
    Trn1S32,
    Trn2S32,
    Uzp1U8,
    Zip1U8,
    QrdmulhS16,
    RshlS32,
    QrshrunS16N4,
}

#[target_feature(enable = "neon")]
pub(super) fn neon_govde(kip: NeonKip, a: &V16, b: &V16) -> V16 {
    let va: uint8x16_t = load_u8x16(a);
    let vb: uint8x16_t = load_u8x16(b);
    let vr: uint8x16_t = match kip {
        NeonKip::Qtbl1U8 => vqtbl1q_u8(va, vb),
        NeonKip::ExtU8(n) => match n {
            0 => vextq_u8::<0>(va, vb),
            1 => vextq_u8::<1>(va, vb),
            4 => vextq_u8::<4>(va, vb),
            8 => vextq_u8::<8>(va, vb),
            15 => vextq_u8::<15>(va, vb),
            _ => panic!("desteklenmeyen ext imm: {n}"),
        },
        NeonKip::Trn1S32 => {
            let xa: int32x4_t = vreinterpretq_s32_u8(va);
            let xb: int32x4_t = vreinterpretq_s32_u8(vb);
            vreinterpretq_u8_s32(vtrn1q_s32(xa, xb))
        }
        NeonKip::Trn2S32 => {
            let xa: int32x4_t = vreinterpretq_s32_u8(va);
            let xb: int32x4_t = vreinterpretq_s32_u8(vb);
            vreinterpretq_u8_s32(vtrn2q_s32(xa, xb))
        }
        NeonKip::Uzp1U8 => vuzp1q_u8(va, vb),
        NeonKip::Zip1U8 => vzip1q_u8(va, vb),
        NeonKip::QrdmulhS16 => {
            let xa: int16x8_t = vreinterpretq_s16_u8(va);
            let xb: int16x8_t = vreinterpretq_s16_u8(vb);
            vreinterpretq_u8_s16(vqrdmulhq_s16(xa, xb))
        }
        NeonKip::RshlS32 => {
            let xa: int32x4_t = vreinterpretq_s32_u8(va);
            // Kayma miktarlarını [-16, +8] aralığına sınırla ki UB / aşırı kayma olmasın.
            let sb_raw = i32x4_oku(b);
            let sb_clamped: [i32; 4] = core::array::from_fn(|i| sb_raw[i].rem_euclid(25) - 16);
            let sb_bytes = i32x4_yaz(sb_clamped);
            let xb: int32x4_t = vreinterpretq_s32_u8(load_u8x16(&sb_bytes));
            vreinterpretq_u8_s32(vrshlq_s32(xa, xb))
        }
        NeonKip::QrshrunS16N4 => {
            let xa: int16x8_t = vreinterpretq_s16_u8(va);
            let lo = vqrshrun_n_s16::<4>(xa);
            vcombine_u8(lo, vdup_n_u8(0))
        }
    };
    let mut out = [0u8; 16];
    store_u8x16(&mut out, vr);
    out
}

#[target_feature(enable = "neon,dotprod")]
pub(super) fn dotprod_govde(acc: &V16, a: &V16, b: &V16, signed: bool) -> V16 {
    let vacc = load_u8x16(acc);
    let va = load_u8x16(a);
    let vb = load_u8x16(b);
    let vr: uint8x16_t = if signed {
        let acc_s: int32x4_t = vreinterpretq_s32_u8(vacc);
        let as8: int8x16_t = vreinterpretq_s8_u8(va);
        let bs8: int8x16_t = vreinterpretq_s8_u8(vb);
        vreinterpretq_u8_s32(vdotq_s32(acc_s, as8, bs8))
    } else {
        let acc_u: uint32x4_t = vreinterpretq_u32_u8(vacc);
        vreinterpretq_u8_u32(vdotq_u32(acc_u, va, vb))
    };
    let mut out = [0u8; 16];
    store_u8x16(&mut out, vr);
    out
}

fn i16x8_oku(v: &V16) -> [i16; 8] {
    core::array::from_fn(|i| i16::from_le_bytes([v[2 * i], v[2 * i + 1]]))
}

fn i16x8_yaz(a: [i16; 8]) -> V16 {
    let mut o = [0u8; 16];
    for (i, x) in a.iter().enumerate() {
        o[2 * i..2 * i + 2].copy_from_slice(&x.to_le_bytes());
    }
    o
}

fn i32x4_oku(v: &V16) -> [i32; 4] {
    core::array::from_fn(|i| {
        i32::from_le_bytes([v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]])
    })
}

fn i32x4_yaz(a: [i32; 4]) -> V16 {
    let mut o = [0u8; 16];
    for (i, x) in a.iter().enumerate() {
        o[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
    }
    o
}

fn girdiler() -> Vec<(V16, V16)> {
    let mut s: u64 = 0x2026_1123_5454_0001;
    let mut rnd = || -> V16 {
        let mut v = [0u8; 16];
        for ch in v.chunks_mut(8) {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            ch.copy_from_slice(&s.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
        }
        v
    };
    let mut out: Vec<(V16, V16)> = (0..512).map(|_| (rnd(), rnd())).collect();
    out.push((i16x8_yaz([i16::MIN; 8]), i16x8_yaz([i16::MIN; 8])));
    out.push((i16x8_yaz([i16::MAX; 8]), i16x8_yaz([i16::MAX; 8])));
    out.push((
        i16x8_yaz([-32768, 32767, -1, 0, 4095, -4096, 255, -256]),
        [0x10; 16],
    ));
    out.push(([0xff; 16], [0x7f; 16]));
    out.push(([0xff; 16], [0x80; 16]));
    out.push((
        core::array::from_fn(|i| i as u8),
        core::array::from_fn(|i| (i as u8) + 8),
    ));
    out
}

fn sina_neon(kip: NeonKip, dogru: Ref16<'_>, yanlis: Option<Ref16<'_>>) {
    let token =
        NeonToken::new(CpuFlags::detect()).expect("aarch64 üzerinde NeonToken mevcut olmalı");
    let mut ayrisan = 0usize;
    for (a, b) in girdiler() {
        let got = super::giris::sozluk_neon(token, kip, &a, &b);
        let want = dogru(&a, &b);
        assert_eq!(
            got, want,
            "{kip:?}: NEON ≠ skaler referans; a={a:02x?} b={b:02x?}"
        );
        if yanlis.is_some_and(|y| got != y(&a, &b)) {
            ayrisan += 1;
        }
    }
    assert!(
        yanlis.is_none() || ayrisan > 0,
        "{kip:?}: negatif denetim — tuzağı yok sayan yanlış referans hiç ayrışmadı!"
    );
}

#[test]
fn lane_vqtbl1q_u8_sinir_disi_16_ustu_sifirlar() {
    let dogru = |a: &V16, b: &V16| -> V16 {
        core::array::from_fn(|i| if b[i] < 16 { a[usize::from(b[i])] } else { 0 })
    };
    // Tuzak: x86 `pshufb` gibi yalnız bit 7'ye bakıp `16..127` indekslerini `b[i] & 15` sarmak yanlıştır!
    let yanlis_pshufb = |a: &V16, b: &V16| -> V16 {
        core::array::from_fn(|i| {
            if b[i] & 0x80 != 0 {
                0
            } else {
                a[usize::from(b[i] & 15)]
            }
        })
    };
    sina_neon(NeonKip::Qtbl1U8, &dogru, Some(&yanlis_pshufb));
}

#[test]
fn lane_vextq_u8_arguman_sirasi_palignr_tersidir() {
    for n in [0u8, 1, 4, 8, 15] {
        let dogru = |a: &V16, b: &V16| -> V16 {
            let k = usize::from(n);
            core::array::from_fn(|i| if i + k < 16 { a[i + k] } else { b[i + k - 16] })
        };
        // Tuzak: x86 `_mm_alignr_epi8(a, b, n)` `b`'yi alt, `a`'yı üst alır (`vextq_u8` tam tersidir!).
        let yanlis_palignr = |a: &V16, b: &V16| -> V16 {
            let k = usize::from(n);
            core::array::from_fn(|i| if i + k < 16 { b[i + k] } else { a[i + k - 16] })
        };
        sina_neon(NeonKip::ExtU8(n), &dogru, Some(&yanlis_palignr));
    }
}

#[test]
fn lane_vtrn1q_ve_vtrn2q_s32_serit_eslemesi() {
    let r_trn1 = |a: &V16, b: &V16| -> V16 {
        let (xa, xb) = (i32x4_oku(a), i32x4_oku(b));
        i32x4_yaz([xa[0], xb[0], xa[2], xb[2]])
    };
    let r_trn2 = |a: &V16, b: &V16| -> V16 {
        let (xa, xb) = (i32x4_oku(a), i32x4_oku(b));
        i32x4_yaz([xa[1], xb[1], xa[3], xb[3]])
    };
    // Tuzak: `zip1` (`[xa[0], xb[0], xa[1], xb[1]]`) ile `trn1`'i karıştırmak!
    let yanlis_zip1 = |a: &V16, b: &V16| -> V16 {
        let (xa, xb) = (i32x4_oku(a), i32x4_oku(b));
        i32x4_yaz([xa[0], xb[0], xa[1], xb[1]])
    };
    sina_neon(NeonKip::Trn1S32, &r_trn1, Some(&yanlis_zip1));
    sina_neon(NeonKip::Trn2S32, &r_trn2, Some(&r_trn1));
}

#[test]
fn lane_vuzp1q_ve_vzip1q_u8_128bit_serit_duzeni() {
    let r_uzp1 = |a: &V16, b: &V16| -> V16 {
        core::array::from_fn(|i| if i < 8 { a[2 * i] } else { b[2 * (i - 8)] })
    };
    let yanlis_uzp = |a: &V16, b: &V16| -> V16 {
        core::array::from_fn(|i| if i % 2 == 0 { a[i] } else { b[i - 1] })
    };
    sina_neon(NeonKip::Uzp1U8, &r_uzp1, Some(&yanlis_uzp));

    let r_zip1 = |a: &V16, b: &V16| -> V16 {
        core::array::from_fn(|i| if i % 2 == 0 { a[i / 2] } else { b[i / 2] })
    };
    let yanlis_zip2 = |a: &V16, b: &V16| -> V16 {
        core::array::from_fn(|i| {
            if i % 2 == 0 {
                a[8 + i / 2]
            } else {
                b[8 + i / 2]
            }
        })
    };
    sina_neon(NeonKip::Zip1U8, &r_zip1, Some(&yanlis_zip2));
}

#[test]
fn lane_vqrdmulhq_s16_yuvarlama_ve_min_kare_doyumu() {
    let dogru = |a: &V16, b: &V16| -> V16 {
        let (xa, xb) = (i16x8_oku(a), i16x8_oku(b));
        i16x8_yaz(core::array::from_fn(|i| {
            let p = (2 * i64::from(xa[i]) * i64::from(xb[i]) + (1 << 15)) >> 16;
            p.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
        }))
    };
    // Tuzak: x86 `_mm_mulhrs_epi16` `-32768 * -32768`'de taşarak `-32768` döner; NEON `sqrdmulh` `+32767`'ye doyurur!
    let yanlis_mulhrs = |a: &V16, b: &V16| -> V16 {
        let (xa, xb) = (i16x8_oku(a), i16x8_oku(b));
        i16x8_yaz(core::array::from_fn(|i| {
            ((((i32::from(xa[i]) * i32::from(xb[i])) >> 14) + 1) >> 1) as i16
        }))
    };
    sina_neon(NeonKip::QrdmulhS16, &dogru, Some(&yanlis_mulhrs));
}

#[test]
fn lane_vrshlq_s32_negatif_us_yuvarlamali_kaydirma() {
    let dogru = |a: &V16, b: &V16| -> V16 {
        let xa = i32x4_oku(a);
        let sb_raw = i32x4_oku(b);
        i32x4_yaz(core::array::from_fn(|i| {
            let sh = sb_raw[i].rem_euclid(25) - 16;
            if sh >= 0 {
                (i64::from(xa[i]) << sh) as i32
            } else {
                let r = (-sh) as u32;
                ((i64::from(xa[i]) + (1i64 << (r - 1))) >> r) as i32
            }
        }))
    };
    let yanlis_yuvarlamasiz = |a: &V16, b: &V16| -> V16 {
        let xa = i32x4_oku(a);
        let sb_raw = i32x4_oku(b);
        i32x4_yaz(core::array::from_fn(|i| {
            let sh = sb_raw[i].rem_euclid(25) - 16;
            if sh >= 0 {
                (i64::from(xa[i]) << sh) as i32
            } else {
                (i64::from(xa[i]) >> (-sh)) as i32
            }
        }))
    };
    sina_neon(NeonKip::RshlS32, &dogru, Some(&yanlis_yuvarlamasiz));
}

#[test]
fn lane_vqrshrun_n_s16_yuvarlama_ve_isaretsiz_doyum() {
    let dogru = |a: &V16, _b: &V16| -> V16 {
        let xa = i16x8_oku(a);
        let mut o = [0u8; 16];
        for i in 0..8 {
            let v = (i32::from(xa[i]) + 8) >> 4;
            o[i] = v.clamp(0, 255) as u8;
        }
        o
    };
    let yanlis_doyumsuz = |a: &V16, _b: &V16| -> V16 {
        let xa = i16x8_oku(a);
        let mut o = [0u8; 16];
        for i in 0..8 {
            o[i] = ((i32::from(xa[i]) + 8) >> 4) as u8;
        }
        o
    };
    sina_neon(NeonKip::QrshrunS16N4, &dogru, Some(&yanlis_doyumsuz));
}

#[test]
fn lane_vdotq_s32_ve_u32_dortlu_ic_carpim() {
    let Some(dot) = DotprodToken::new(CpuFlags::detect()) else {
        return;
    };
    let mut ayrisan_s = 0usize;
    let mut ayrisan_u = 0usize;
    let acc0 = i32x4_yaz([1000, -2000, 0, i32::MAX - 500_000]);
    for (a, b) in girdiler() {
        let got_s = super::giris::sozluk_dotprod(dot, &acc0, &a, &b, true);
        let acc_s = i32x4_oku(&acc0);
        let want_s = i32x4_yaz(core::array::from_fn(|g| {
            let mut sum = acc_s[g];
            for k in 0..4 {
                let prod = i32::from(a[4 * g + k] as i8) * i32::from(b[4 * g + k] as i8);
                sum = sum.wrapping_add(prod);
            }
            sum
        }));
        assert_eq!(got_s, want_s, "vdotq_s32 ≠ skaler");

        // Tuzak: `pmaddubsw` gibi ilk işleneni `u8` sanmak veya ara `i16` doyumu yapmak!
        let yanlis_us = i32x4_yaz(core::array::from_fn(|g| {
            let mut sum = acc_s[g];
            for k in 0..4 {
                let prod = i32::from(a[4 * g + k]) * i32::from(b[4 * g + k] as i8);
                sum = sum.wrapping_add(prod);
            }
            sum
        }));
        if got_s != yanlis_us {
            ayrisan_s += 1;
        }

        let acc_u0 = [1u8; 16];
        let got_u = super::giris::sozluk_dotprod(dot, &acc_u0, &a, &b, false);
        let acc_u = i32x4_oku(&acc_u0).map(|x| x as u32);
        let want_u_arr: [i32; 4] = core::array::from_fn(|g| {
            let mut sum = acc_u[g];
            for k in 0..4 {
                sum = sum.wrapping_add(u32::from(a[4 * g + k]) * u32::from(b[4 * g + k]));
            }
            sum as i32
        });
        assert_eq!(got_u, i32x4_yaz(want_u_arr), "vdotq_u32 ≠ skaler");
        if got_u != got_s {
            ayrisan_u += 1;
        }
    }
    assert!(
        ayrisan_s > 0 && ayrisan_u > 0,
        "vdotq negatif denetim ayrışmalı"
    );
}
