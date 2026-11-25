//! x86 SIMD aktarım sözlüğünün testleri (F3d / P27, SMD-03/04).
//!
//! İki parça:
//!
//! 1. **Derleme:** `sozluk_liste.in` (`docs/simd-sozluk/x86.csv`'den
//!    `scripts/inventory/x86_sozluk.py` ile üretilir) sözlükteki bütün
//!    intrinsic adlarını `core::arch::x86_64`'ten içe aktarır. Ad yoksa ya da
//!    Rust 1.99.1'de kararsızsa test ikilisi **derlenmez**; liste ayrıca CSV
//!    sütunuyla karşılaştırılır ([`sozluk_listesi_csv_ile_ayni`]).
//! 2. **Lane davranışı:** dav1d asm'inin dayandığı lane tuzakları (pshufb lane
//!    içi + bit 7, pack lane sırası + doyum, `vperm2i128`/`vpermq` imm8,
//!    `palignr` lane, `pmaddubsw` ara doyum, `pmulhrsw` yuvarlama, bayt
//!    kaydırma lane'i) skaler referans işlevlerle sınanır. Her testin bir
//!    **negatif denetimi** vardır: tuzağı yok sayan (çapraz lane, doyumsuz,
//!    yuvarlamasız) referans en az bir girdide SIMD sonucundan ayrışmalıdır;
//!    ayrışmazsa test girdileri tuzağa duyarsızdır ve test düşer.
//!
//! ## Yetki ve `unsafe`
//!
//! AVX2 gövdesi ([`avx2_govde`]) güvenli bir `#[target_feature]` işlevidir;
//! AVX2'siz test bağlamından çağırmak Rust 1.99.1'de `unsafe` ister. Tek çağrı
//! [`avx2_ile`] içindedir ve [`Avx2Token`] ister (belirteç gerçek tespit +
//! maske ile kurulur, `cpu::token`). AVX2'siz makinede lane testleri
//! `on_kosul::eksik` ile bildirilir (P29: `AV1OXIDE_ORACLES_REQUIRED` ayarlıysa FAIL). Modül yalnız `cfg(test)` ile derlenir; üretim ikilisine
//! girmez.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m256i, _mm256_alignr_epi8, _mm256_bsrli_epi128, _mm256_extract_epi64, _mm256_maddubs_epi16,
    _mm256_mulhrs_epi16, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_setr_epi64x, _mm256_shuffle_epi8, _mm256_unpacklo_epi8,
};

use crate::cpu::{Avx2Token, CpuFlags};

/// Ön koşul yardımcısı (P29): entegrasyon testleriyle aynı kaynak dosyası.
#[allow(dead_code)]
#[path = "../../../tests/common/on_kosul.rs"]
mod on_kosul;
use on_kosul::Neden;

/// Üretilmiş listeyi içe aktarma + ad dizisine açar.
macro_rules! sozluk_intrinsic_listesi {
    ($($ad:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use core::arch::x86_64::{$($ad),*};
        /// Sözlükteki intrinsic adları (üretilmiş listeden, sıralı).
        pub(super) const ADLAR: &[&str] = &[$(stringify!($ad)),*];
    };
}

/// Derleme testi: içe aktarma lane testlerinin adlarıyla çakışmasın diye ayrı modülde.
mod derleme {
    include!("sozluk_liste.in");
}

#[test]
fn sozluk_listesi_tutarli() {
    let liste: Vec<&str> = derleme::ADLAR.to_vec();
    assert!(
        liste.len() > 300,
        "sözlük beklenenden küçük: {}",
        liste.len()
    );
}

// --- lane davranışı ----------------------------------------------------------

/// 256 bitlik değer, bayt dizisi olarak (küçük uçlu).
type V = [u8; 32];

/// İki girdili skaler referans işlevi.
type Ref<'a> = &'a dyn Fn(&V, &V) -> V;

/// Sınanan intrinsic (imm8 biçimleri sabit kümeden).
#[derive(Clone, Copy, Debug)]
enum Kip {
    Pshufb,
    Packuswb,
    Packssdw,
    Punpcklbw,
    Perm2i128(u8),
    Permq(u8),
    Palignr(u8),
    Maddubs,
    Mulhrs,
    Bsrli(u8),
}

const PERM2I128_IMM: [u8; 10] = [0x00, 0x01, 0x03, 0x08, 0x12, 0x20, 0x21, 0x31, 0x80, 0x83];
const PERMQ_IMM: [u8; 5] = [0x00, 0x1b, 0x4e, 0xd8, 0xe4];
const PALIGNR_IMM: [u8; 7] = [0, 1, 4, 8, 15, 16, 17];
const BSRLI_IMM: [u8; 5] = [1, 3, 8, 15, 16];

/// imm8'i sabit kümeden const generic'e çevirir.
macro_rules! imm {
    ($v:expr; $($i:literal),*; $f:ident $args:tt) => {
        match $v {
            $($i => $f::<$i> $args,)*
            v => panic!("sınanmayan imm8 {v:#x}"),
        }
    };
}

#[target_feature(enable = "avx2")]
fn yukle(v: &V) -> __m256i {
    let q = |i: usize| i64::from_le_bytes(*v[i * 8..].first_chunk::<8>().expect("8 bayt"));
    _mm256_setr_epi64x(q(0), q(1), q(2), q(3))
}

#[target_feature(enable = "avx2")]
fn al(m: __m256i) -> V {
    let q = [
        _mm256_extract_epi64::<0>(m),
        _mm256_extract_epi64::<1>(m),
        _mm256_extract_epi64::<2>(m),
        _mm256_extract_epi64::<3>(m),
    ];
    let mut o = [0u8; 32];
    for (i, x) in q.iter().enumerate() {
        o[i * 8..i * 8 + 8].copy_from_slice(&x.to_le_bytes());
    }
    o
}

/// AVX2 gövdesi: `kip` intrinsic'ini `a`, `b` üzerinde çalıştırır.
#[target_feature(enable = "avx2")]
fn avx2_govde(_avx2: Avx2Token, kip: Kip, a: &V, b: &V) -> V {
    let (x, y) = (yukle(a), yukle(b));
    al(match kip {
        Kip::Pshufb => _mm256_shuffle_epi8(x, y),
        Kip::Packuswb => _mm256_packus_epi16(x, y),
        Kip::Packssdw => _mm256_packs_epi32(x, y),
        Kip::Punpcklbw => _mm256_unpacklo_epi8(x, y),
        Kip::Perm2i128(i) => imm!(i; 0x00, 0x01, 0x03, 0x08, 0x12, 0x20, 0x21, 0x31, 0x80, 0x83;
            _mm256_permute2x128_si256(x, y)),
        Kip::Permq(i) => imm!(i; 0x00, 0x1b, 0x4e, 0xd8, 0xe4; _mm256_permute4x64_epi64(x)),
        Kip::Palignr(i) => imm!(i; 0, 1, 4, 8, 15, 16, 17; _mm256_alignr_epi8(x, y)),
        Kip::Maddubs => _mm256_maddubs_epi16(x, y),
        Kip::Mulhrs => _mm256_mulhrs_epi16(x, y),
        Kip::Bsrli(i) => imm!(i; 1, 3, 8, 15, 16; _mm256_bsrli_epi128(x)),
    })
}

/// Belirteçli giriş: AVX2 gövdesine giden tek yol.
fn avx2_ile(avx2: Avx2Token, kip: Kip, a: &V, b: &V) -> V {
    // SAFETY: (1) ISA: `avx2_govde` `#[target_feature(enable = "avx2")]`
    // taşır; `Avx2Token` yalnız AVX2 tespiti + maske izniyle kurulur
    // (`cpu::token`). (2) sınır: gövde yalnız `&[u8; 32]` değerlerini kopyalar,
    // bellek erişimli intrinsic kullanmaz. (3) hizalama/takma ad: yazmaç
    // değerleriyle çalışır; paylaşılan referanslar yalnız okunur.
    unsafe { avx2_govde(avx2, kip, a, b) }
}

fn i16ler(v: &V) -> [i16; 16] {
    core::array::from_fn(|i| i16::from_le_bytes([v[2 * i], v[2 * i + 1]]))
}

fn i16den(a: [i16; 16]) -> V {
    let mut o = [0u8; 32];
    for (i, x) in a.iter().enumerate() {
        o[2 * i..2 * i + 2].copy_from_slice(&x.to_le_bytes());
    }
    o
}

fn i32ler(v: &V) -> [i32; 8] {
    core::array::from_fn(|i| i32::from_le_bytes(*v[4 * i..].first_chunk::<4>().expect("4 bayt")))
}

fn sat_u8(x: i16) -> u8 {
    x.clamp(0, 255) as u8
}

fn sat_i16(x: i32) -> i16 {
    x.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// 128 bit yarılar: `[a.lo, a.hi, b.lo, b.hi]`.
fn yarilar<'a>(a: &'a V, b: &'a V) -> [&'a [u8]; 4] {
    [&a[..16], &a[16..], &b[..16], &b[16..]]
}

// Skaler referanslar (Intel SDM komut tanımları).

fn r_pshufb(a: &V, b: &V) -> V {
    core::array::from_fn(|i| {
        if b[i] & 0x80 != 0 {
            0
        } else {
            a[(i & !15) + usize::from(b[i] & 15)]
        }
    })
}

fn r_packuswb(a: &V, b: &V) -> V {
    let (x, y) = (i16ler(a), i16ler(b));
    core::array::from_fn(|i| {
        let (lane, j) = (i / 16, i % 16);
        sat_u8(if j < 8 {
            x[lane * 8 + j]
        } else {
            y[lane * 8 + j - 8]
        })
    })
}

fn r_packssdw(a: &V, b: &V) -> V {
    let (x, y) = (i32ler(a), i32ler(b));
    i16den(core::array::from_fn(|i| {
        let (lane, j) = (i / 8, i % 8);
        sat_i16(if j < 4 {
            x[lane * 4 + j]
        } else {
            y[lane * 4 + j - 4]
        })
    }))
}

fn r_punpcklbw(a: &V, b: &V) -> V {
    core::array::from_fn(|i| {
        let (lane, j) = (i / 16, i % 16);
        let src = if j % 2 == 0 { a } else { b };
        src[lane * 16 + j / 2]
    })
}

fn r_perm2i128(imm: u8, a: &V, b: &V) -> V {
    let h = yarilar(a, b);
    core::array::from_fn(|i| {
        let s = (imm >> (4 * (i / 16))) & 0xf;
        if s & 8 != 0 {
            0
        } else {
            h[usize::from(s & 3)][i % 16]
        }
    })
}

fn r_permq(imm: u8, a: &V) -> V {
    core::array::from_fn(|i| {
        let q = usize::from((imm >> (2 * (i / 8))) & 3);
        a[q * 8 + i % 8]
    })
}

fn r_palignr(n: u8, a: &V, b: &V) -> V {
    core::array::from_fn(|i| {
        let (lane, j) = (i / 16, i % 16);
        let k = usize::from(n) + j;
        // Lane başına `a_lane : b_lane` (a üstte) birleşimi, `n` bayt sağa.
        match k {
            0..16 => b[lane * 16 + k],
            16..32 => a[lane * 16 + k - 16],
            _ => 0,
        }
    })
}

fn r_maddubs(a: &V, b: &V) -> V {
    i16den(core::array::from_fn(|i| {
        let p = |k: usize| i32::from(a[k]) * i32::from(b[k] as i8);
        sat_i16(p(2 * i) + p(2 * i + 1))
    }))
}

fn r_mulhrs(a: &V, b: &V) -> V {
    let (x, y) = (i16ler(a), i16ler(b));
    i16den(core::array::from_fn(|i| {
        ((((i32::from(x[i]) * i32::from(y[i])) >> 14) + 1) >> 1) as i16
    }))
}

fn r_bsrli(n: u8, a: &V) -> V {
    core::array::from_fn(|i| {
        let (lane, j) = (i / 16, i % 16);
        let k = j + usize::from(n);
        if k < 16 { a[lane * 16 + k] } else { 0 }
    })
}

// Negatif denetim referansları: tuzağı yok sayan (yanlış) tanımlar.

/// 32 baytı tek tablo sayar (lane yok).
fn y_pshufb_capraz(a: &V, b: &V) -> V {
    core::array::from_fn(|i| {
        if b[i] & 0x80 != 0 {
            0
        } else {
            a[usize::from(b[i] & 31)]
        }
    })
}

/// Önce bütün `a`, sonra bütün `b` (çapraz lane sırası).
fn y_packuswb_capraz(a: &V, b: &V) -> V {
    let (x, y) = (i16ler(a), i16ler(b));
    core::array::from_fn(|i| sat_u8(if i < 16 { x[i] } else { y[i - 16] }))
}

fn y_packssdw_capraz(a: &V, b: &V) -> V {
    let (x, y) = (i32ler(a), i32ler(b));
    i16den(core::array::from_fn(|i| {
        sat_i16(if i < 8 { x[i] } else { y[i - 8] })
    }))
}

/// Bütün vektörün alt yarısını serpiştirir (çapraz lane).
fn y_punpcklbw_capraz(a: &V, b: &V) -> V {
    core::array::from_fn(|i| if i % 2 == 0 { a[i / 2] } else { b[i / 2] })
}

/// Sıfırlama bitini (bit 3/7) yok sayar.
fn y_perm2i128_sifirsiz(imm: u8, a: &V, b: &V) -> V {
    let h = yarilar(a, b);
    core::array::from_fn(|i| h[usize::from((imm >> (4 * (i / 16))) & 3)][i % 16])
}

/// imm8 alanlarını ters sırada okur.
fn y_permq_ters(imm: u8, a: &V) -> V {
    core::array::from_fn(|i| {
        let q = usize::from((imm >> (6 - 2 * (i / 8))) & 3);
        a[q * 8 + i % 8]
    })
}

/// 256 bitlik `a : b` birleşimini tek parça kaydırır (çapraz lane).
fn y_palignr_capraz(n: u8, a: &V, b: &V) -> V {
    core::array::from_fn(|i| {
        let k = usize::from(n) + i;
        if k < 32 { b[k] } else { a[k - 32] }
    })
}

/// Ara toplamı doyurmaz (i16'ya sarar).
fn y_maddubs_doyumsuz(a: &V, b: &V) -> V {
    i16den(core::array::from_fn(|i| {
        let p = |k: usize| i32::from(a[k]) * i32::from(b[k] as i8);
        (p(2 * i) + p(2 * i + 1)) as i16
    }))
}

/// Yuvarlamasız `(a*b) >> 15`.
fn y_mulhrs_yuvarlamasiz(a: &V, b: &V) -> V {
    let (x, y) = (i16ler(a), i16ler(b));
    i16den(core::array::from_fn(|i| {
        ((i32::from(x[i]) * i32::from(y[i])) >> 15) as i16
    }))
}

/// 256 biti tek parça kaydırır (çapraz lane).
fn y_bsrli_capraz(n: u8, a: &V) -> V {
    core::array::from_fn(|i| {
        let k = i + usize::from(n);
        if k < 32 { a[k] } else { 0 }
    })
}

/// Belirlenimci girdi üreticisi (xorshift64*) + uç değer çiftleri.
fn girdiler() -> Vec<(V, V)> {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut rasgele = || -> V {
        let mut v = [0u8; 32];
        for c in v.chunks_mut(8) {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            c.copy_from_slice(&s.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
        }
        v
    };
    let mut out: Vec<(V, V)> = (0..512).map(|_| (rasgele(), rasgele())).collect();
    // Uçlar: doyum (255·127·2, 255·−128·2), −32768², sıfır, işaret sınırları.
    let i16_dolu = |x: i16| i16den([x; 16]);
    out.push(([0xff; 32], [0x7f; 32]));
    out.push(([0xff; 32], [0x80; 32]));
    out.push((i16_dolu(i16::MIN), i16_dolu(i16::MIN)));
    out.push((i16_dolu(i16::MAX), i16_dolu(i16::MAX)));
    out.push((i16_dolu(-1), i16_dolu(1)));
    out.push(([0; 32], [0; 32]));
    out.push((
        core::array::from_fn(|i| i as u8),
        core::array::from_fn(|i| 31 - i as u8),
    ));
    out
}

/// `kip`'i bütün girdilerde doğru referansla karşılaştırır. `yanlis` verilirse
/// tuzağı yok sayan referansın en az bir girdide ayrıştığını (negatif denetim)
/// doğrular; tuzağın görünmediği imm8 biçimlerinde (`None`) yalnız doğruluk
/// sınanır.
fn sina(kip: Kip, dogru: Ref<'_>, yanlis: Option<Ref<'_>>) {
    let Some(avx2) = Avx2Token::new(CpuFlags::detect()) else {
        on_kosul::eksik(
            &format!("av1oxide::dsp::x86::sozluk::{kip:?}"),
            Neden::avx2(),
            "lane testi AVX2 ister",
        );
        return;
    };
    let mut ayrisan = 0usize;
    for (a, b) in girdiler() {
        let s = avx2_ile(avx2, kip, &a, &b);
        assert_eq!(
            s,
            dogru(&a, &b),
            "{kip:?}: SIMD ≠ skaler referans; a={a:02x?} b={b:02x?}"
        );
        if yanlis.is_some_and(|y| s != y(&a, &b)) {
            ayrisan += 1;
        }
    }
    assert!(
        yanlis.is_none() || ayrisan > 0,
        "{kip:?}: negatif denetim — tuzağı yok sayan referans hiç ayrışmadı (girdiler duyarsız)"
    );
}

#[test]
fn lane_pshufb_lane_ici_ve_bit7() {
    sina(Kip::Pshufb, &r_pshufb, Some(&y_pshufb_capraz));
}

#[test]
fn lane_packuswb_sira_ve_doyum() {
    sina(Kip::Packuswb, &r_packuswb, Some(&y_packuswb_capraz));
}

#[test]
fn lane_packssdw_sira_ve_doyum() {
    sina(Kip::Packssdw, &r_packssdw, Some(&y_packssdw_capraz));
}

#[test]
fn lane_punpcklbw_lane_ici() {
    sina(Kip::Punpcklbw, &r_punpcklbw, Some(&y_punpcklbw_capraz));
}

#[test]
fn lane_vperm2i128_imm8() {
    for imm in PERM2I128_IMM {
        let yanlis = |a: &V, b: &V| y_perm2i128_sifirsiz(imm, a, b);
        // Sıfırlama biti (3/7) yoksa yanlış referans doğruyla aynıdır.
        let y: Option<Ref<'_>> = (imm & 0x88 != 0).then_some(&yanlis);
        sina(Kip::Perm2i128(imm), &|a, b| r_perm2i128(imm, a, b), y);
    }
}

#[test]
fn lane_vpermq_imm8() {
    for imm in PERMQ_IMM {
        let yanlis = |a: &V, _: &V| y_permq_ters(imm, a);
        // imm8 = 0x00 bütün alanlarda aynıdır; ters okuma ayrışmaz.
        let y: Option<Ref<'_>> = (imm != 0).then_some(&yanlis);
        sina(Kip::Permq(imm), &|a, _| r_permq(imm, a), y);
    }
}

#[test]
fn lane_vpalignr_lane_ici() {
    for n in PALIGNR_IMM {
        let yanlis = |a: &V, b: &V| y_palignr_capraz(n, a, b);
        // n = 0 → sonuç `b`; lane'ler arası taşıma yoktur.
        let y: Option<Ref<'_>> = (n != 0).then_some(&yanlis);
        sina(Kip::Palignr(n), &|a, b| r_palignr(n, a, b), y);
    }
}

#[test]
fn lane_pmaddubsw_ara_doyum() {
    sina(Kip::Maddubs, &r_maddubs, Some(&y_maddubs_doyumsuz));
}

#[test]
fn lane_pmulhrsw_yuvarlama() {
    sina(Kip::Mulhrs, &r_mulhrs, Some(&y_mulhrs_yuvarlamasiz));
}

#[test]
fn lane_vpsrldq_lane_kaydirma() {
    for n in BSRLI_IMM {
        sina(
            Kip::Bsrli(n),
            &|a, _| r_bsrli(n, a),
            Some(&|a, _| y_bsrli_capraz(n, a)),
        );
    }
}
