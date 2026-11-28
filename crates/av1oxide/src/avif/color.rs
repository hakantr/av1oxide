//! ITU-T H.273 / CICP renk uzayı ve kroma alt örnekleme dönüşümleri.

#![forbid(unsafe_code)]

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::avif::image::{AvifImage, PlaneStorage, Rgb16Image, RgbImage, Rgba16Image, RgbaImage};
use crate::error::{Error, Result, SizeStage};
use crate::frame::butce::{Butce, ButceTutucu};
use crate::frame::{BitDepth, ChromaFormat};

/// `std::thread::scope` şerit paralelliği için minimum piksel eşiği (`512 × 512`).
#[cfg(not(target_arch = "wasm32"))]
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
const PARALLEL_PIXEL_THRESHOLD: usize = 262_144;

/// `color_primaries` kod noktasına karşılık gelen CIE 1931 `(rX, rY, gX, gY, bX, bY, wX, wY)`
/// birincil renk ve beyaz nokta koordinatlarını döndürür (`ITU-T H.273` Tablo 2).
#[inline]
fn color_primaries_cie_xy(color_primaries: u16) -> [f32; 8] {
    match color_primaries {
        1 => [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290], // BT.709
        4 => [0.67, 0.33, 0.21, 0.71, 0.14, 0.08, 0.310, 0.316],   // BT.470M
        5 => [0.64, 0.33, 0.29, 0.60, 0.15, 0.06, 0.3127, 0.3290], // BT.470BG
        6 | 7 => [0.630, 0.340, 0.310, 0.595, 0.155, 0.070, 0.3127, 0.3290], // BT.601 / SMPTE 240M
        8 => [0.681, 0.319, 0.243, 0.692, 0.145, 0.049, 0.310, 0.316], // Generic film
        9 => [0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290], // BT.2020
        10 => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3333, 0.3333],      // SMPTE ST 428-1 (XYZ)
        11 => [0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.314, 0.351], // SMPTE RP 431-2 (DCI P3)
        12 => [0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290], // SMPTE EG 432-1 (Display P3)
        22 => [0.630, 0.340, 0.295, 0.605, 0.155, 0.077, 0.3127, 0.3290], // EBU Tech. 3213-E
        _ => [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290],        // Varsayılan (BT.709)
    }
}

/// Birincil renk koordinatlarından parlaklık katsayılarını `(Kr, Kb)` türetir (`ITU-T H.273` Denkl. 32–37).
#[inline]
fn chroma_derived_kr_kb(color_primaries: u16) -> (f32, f32) {
    let [rx, ry, gx, gy, bx, by, wx, wy] = color_primaries_cie_xy(color_primaries);
    let rz = 1.0f32 - (rx + ry);
    let gz = 1.0f32 - (gx + gy);
    let bz = 1.0f32 - (bx + by);
    let wz = 1.0f32 - (wx + wy);
    let denom =
        wy * (rx * (gy * bz - by * gz) + gx * (by * rz - ry * bz) + bx * (ry * gz - gy * rz));
    let kr = (ry
        * (wx * (gy * bz - by * gz) + wy * (bx * gz - gx * bz) + wz * (gx * by - bx * gy)))
        / denom;
    let kb = (by
        * (wx * (ry * gz - gy * rz) + wy * (gx * rz - rx * gz) + wz * (rx * gy - gx * ry)))
        / denom;
    (kr, kb)
}

/// Desteklenmeyen veya ayrılmış `ITU-T H.273` `matrix_coefficients` kod noktalarını denetler.
///
/// Doğrusal matrisler (`0`, `1`, `2`, `4..=9`, `12`, `16`, `17`) ile doğrusal olmayan HDR
/// matrisleri (`10` = `BT.2020 Constant Luminance`, `14` = `ICtCp`) desteklenir; rezerve ve
/// tanımlanmamış kod noktaları (`3`, `11`, `13`, `15`, `>= 18`) [`Error::Unsupported`] döner.
#[inline]
fn validate_matrix_coefficients(matrix_coefficients: u16) -> Result<()> {
    match matrix_coefficients {
        0 | 1 | 2 | 4..=10 | 12 | 14 | 16 | 17 => Ok(()),
        _ => Err(Error::Unsupported(
            "desteklenmeyen veya ayrılmış ITU-T H.273 matrix_coefficients kod noktası",
        )),
    }
}

/// `matrix_coefficients` değerine karşılık gelen `(Kr, Kb)` katsayılarını döndürür (`ITU-T H.273`).
#[inline]
fn matrix_kr_kb(matrix_coefficients: u16, color_primaries: u16) -> (f32, f32) {
    match matrix_coefficients {
        1 => (0.2126, 0.0722),   // BT.709
        4 => (0.30, 0.11),       // FCC
        5 | 6 => (0.299, 0.114), // BT.470BG / BT.601
        7 => (0.212, 0.087),     // SMPTE ST 240
        9 => (0.2627, 0.0593),   // BT.2020 NCL
        12 => chroma_derived_kr_kb(color_primaries),
        _ => (0.299, 0.114), // Varsayılan (2 = Unspecified -> MIAF §7.3.6.4 BT.601)
    }
}

/// Y ve UV örneklerini `[0.0, 1.0]` / `[-0.5, 0.5]` aralığına dönüştüren arama tablolarını oluşturur.
fn build_unorm_tables(
    bit_depth: BitDepth,
    full_range: bool,
    identity_mode: bool,
) -> (Vec<f32>, Vec<f32>) {
    let bits = bit_depth.bits();
    let count = 1usize << bits;
    let max_chan = ((1u32 << bits) - 1) as f32;
    let shift = bits - 8;

    let bias_y = if full_range {
        0.0f32
    } else {
        (16u32 << shift) as f32
    };
    let range_y = if full_range {
        max_chan
    } else {
        (219u32 << shift) as f32
    };

    let mut table_y = Vec::with_capacity(count);
    for cp in 0..count {
        table_y.push(((cp as f32) - bias_y) / range_y);
    }

    if identity_mode {
        let table_uv = table_y.clone();
        (table_y, table_uv)
    } else {
        let bias_uv = (1u32 << (bits - 1)) as f32;
        let range_uv = if full_range {
            max_chan
        } else {
            (224u32 << shift) as f32
        };
        let mut table_uv = Vec::with_capacity(count);
        for cp in 0..count {
            table_uv.push(((cp as f32) - bias_uv) / range_uv);
        }
        (table_y, table_uv)
    }
}

/// 8-bit düzlemler için yığın üzerinde sabit `[f32; 256]` normalizasyon tabloları oluşturur.
fn build_unorm_tables_8bit(full_range: bool, identity_mode: bool) -> ([f32; 256], [f32; 256]) {
    let bias_y = if full_range { 0.0f32 } else { 16.0f32 };
    let range_y = if full_range { 255.0f32 } else { 219.0f32 };

    let mut table_y = [0.0f32; 256];
    for (cp, slot) in table_y.iter_mut().enumerate() {
        *slot = ((cp as f32) - bias_y) / range_y;
    }

    if identity_mode {
        (table_y, table_y)
    } else {
        let bias_uv = 128.0f32;
        let range_uv = if full_range { 255.0f32 } else { 224.0f32 };
        let mut table_uv = [0.0f32; 256];
        for (cp, slot) in table_uv.iter_mut().enumerate() {
            *slot = ((cp as f32) - bias_uv) / range_uv;
        }
        (table_y, table_uv)
    }
}

/// 10/12-bit alfa kanalı → 8-bit alfa dönüşümü için arama tablosu (`libavif` uyumlu).
fn build_alpha_u8_lut(bit_depth: BitDepth) -> Vec<u8> {
    let max_val = bit_depth.max_sample() as usize;
    let src_max_f = max_val as f32;
    let mut lut = Vec::with_capacity(max_val + 1);
    for a in 0..=max_val {
        let alpha_f = (a as f32) / src_max_f;
        lut.push(((0.5f32 + (alpha_f * 255.0f32)) as i32).clamp(0, 255) as u8);
    }
    lut
}

/// NaN içermeyen sonlu `f32` değerlerini dallanmasız `[0.0, 1.0]` aralığına kırpar.
#[inline(always)]
#[allow(clippy::manual_clamp)]
fn clamp01(v: f32) -> f32 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// IEEE-754 `f32` toplama sırasını koruyan `(9, 3, 3, 1) / 16` iki doğrusal kroma bileşimi.
#[inline(always)]
fn bilinear_4(t00: f32, t10: f32, t01: f32, t11: f32) -> f32 {
    (t00 * (9.0 / 16.0)) + (t10 * (3.0 / 16.0)) + (t01 * (3.0 / 16.0)) + (t11 * (1.0 / 16.0))
}

/// Bit derinliğine özgü örnek tablosu erişimi (`U8` sınır-denetimsiz, `U16` doyurmalı).
trait SampleSource: Copy + Send + Sync {
    type Sample: Copy + Send + Sync;

    fn lookup_y(&self, sample: Self::Sample) -> (u16, f32);
    fn lookup_uv(&self, sample: Self::Sample) -> f32;
}

#[derive(Clone, Copy)]
struct TablesU8<'a> {
    table_y: &'a [f32; 256],
    table_uv: &'a [f32; 256],
}

impl SampleSource for TablesU8<'_> {
    type Sample = u8;

    #[inline(always)]
    fn lookup_y(&self, sample: u8) -> (u16, f32) {
        (u16::from(sample), self.table_y[usize::from(sample)])
    }

    #[inline(always)]
    fn lookup_uv(&self, sample: u8) -> f32 {
        self.table_uv[usize::from(sample)]
    }
}

#[derive(Clone, Copy)]
struct TablesU16<'a> {
    table_y: &'a [f32],
    table_uv: &'a [f32],
    max_val: usize,
}

impl SampleSource for TablesU16<'_> {
    type Sample = u16;

    #[inline(always)]
    fn lookup_y(&self, sample: u16) -> (u16, f32) {
        let clamped = usize::from(sample).min(self.max_val);
        (clamped as u16, self.table_y[clamped])
    }

    #[inline(always)]
    fn lookup_uv(&self, sample: u16) -> f32 {
        self.table_uv[usize::from(sample).min(self.max_val)]
    }
}

/// H.273 renk matrisi çekirdeği.
trait MatrixKernel: Copy + Send + Sync {
    fn to_rgb_f32(&self, unorm_y: u16, y_norm: f32, cb: f32, cr: f32) -> (f32, f32, f32);

    #[inline(always)]
    fn to_rgb_f32x4(&self, y: [(u16, f32); 4], cb: [f32; 4], cr: [f32; 4]) -> [(f32, f32, f32); 4] {
        core::array::from_fn(|i| self.to_rgb_f32(y[i].0, y[i].1, cb[i], cr[i]))
    }
}

/// Standart H.273 `(Kr, Kb)` doğrusal matris katsayıları (önceden hesaplanmış IEEE-754 özdeş terimler).
#[derive(Clone, Copy)]
struct StandardMatrix {
    c_r: f32,
    c_b: f32,
    k_kr: f32,
    k_kb: f32,
    kg: f32,
}

impl StandardMatrix {
    fn new(kr: f32, kb: f32) -> Self {
        Self {
            c_r: 2.0 * (1.0 - kr),
            c_b: 2.0 * (1.0 - kb),
            k_kr: kr * (1.0 - kr),
            k_kb: kb * (1.0 - kb),
            kg: 1.0 - kr - kb,
        }
    }
}

impl MatrixKernel for StandardMatrix {
    #[inline(always)]
    fn to_rgb_f32(&self, _unorm_y: u16, y_norm: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
        let r = y_norm + self.c_r * cr;
        let b = y_norm + self.c_b * cb;
        let g = y_norm - ((2.0 * ((self.k_kr * cr) + (self.k_kb * cb))) / self.kg);
        (clamp01(r), clamp01(g), clamp01(b))
    }

    #[inline(always)]
    fn to_rgb_f32x4(&self, y: [(u16, f32); 4], cb: [f32; 4], cr: [f32; 4]) -> [(f32, f32, f32); 4] {
        let y = y.map(|(_, normalized)| normalized);
        // Aynı IEEE-754 işlem sırasını her piksel için koruyarak kanalları ayrı hesapla.
        let red: [f32; 4] = core::array::from_fn(|i| y[i] + self.c_r * cr[i]);
        let green: [f32; 4] = core::array::from_fn(|i| {
            y[i] - ((2.0 * ((self.k_kr * cr[i]) + (self.k_kb * cb[i]))) / self.kg)
        });
        let blue: [f32; 4] = core::array::from_fn(|i| y[i] + self.c_b * cb[i]);
        core::array::from_fn(|i| (clamp01(red[i]), clamp01(green[i]), clamp01(blue[i])))
    }
}

/// `no_std` + `alloc` kipinde `%100 #![forbid(unsafe_code)]` IEEE-754 `f64` matematik yardımcıları.
///
/// `std` açıkken doğrudan standart kütüphane (`f64::sqrt`, `f64::ln`, `f64::exp`, `f64::powf`,
/// `f32::round`) çağrılır; `no_std` kipinde `< 1 ULP` hassasiyetli saf polinom + Newton-Raphson
/// yaklaşımları kullanılır.
mod math {
    #[cfg(any(not(feature = "std"), test))]
    pub(super) mod nostd {
        const LN2_HI: f64 = core::f64::consts::LN_2;
        const LN2_LO: f64 = 2.319_046_813_846_299_6e-17;
        const INV_LN2: f64 = core::f64::consts::LOG2_E;
        const SQRT_2: f64 = core::f64::consts::SQRT_2;

        /// `x >= 0` için 6 adımlı Newton-Raphson IEEE-754 `f64` karekök (`< 0.5 ULP`).
        #[inline]
        pub(crate) fn sqrt(x: f64) -> f64 {
            if x <= 0.0 || x.is_nan() || !x.is_finite() {
                return if x == 0.0 { x } else { f64::NAN };
            }
            let bits = x.to_bits();
            let mut g = f64::from_bits((bits >> 1) + (0x3ff0_0000_0000_0000u64 >> 1));
            for _ in 0..6 {
                g = 0.5 * (g + x / g);
            }
            g
        }

        /// `x > 0` için doğal logaritma (`ln(x)`, `< 1 ULP`).
        #[inline]
        pub(crate) fn ln(x: f64) -> f64 {
            if x <= 0.0 || x.is_nan() || !x.is_finite() {
                return if x == 0.0 {
                    f64::NEG_INFINITY
                } else if x.is_infinite() && x > 0.0 {
                    f64::INFINITY
                } else {
                    f64::NAN
                };
            }
            let mut bits = x.to_bits();
            let mut exp = ((bits >> 52) & 0x7ff) as i32 - 1023;
            // Alt-normal (subnormal) normalize etme.
            if exp == -1023 {
                let scaled = x * f64::from_bits((1023 + 54) << 52);
                bits = scaled.to_bits();
                exp = ((bits >> 52) & 0x7ff) as i32 - 1023 - 54;
            }
            let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
            if m > SQRT_2 {
                m *= 0.5;
                exp += 1;
            }
            // m ∈ [sqrt(0.5), sqrt(2)] ⇒ |u| <= 0.1716, u^2 <= 0.02945
            let u = (m - 1.0) / (m + 1.0);
            let u2 = u * u;
            // 2u * (1 + u^2/3 + u^4/5 + ... + u^18/19)
            let poly = 1.0 / 19.0;
            let poly = 1.0 / 17.0 + u2 * poly;
            let poly = 1.0 / 15.0 + u2 * poly;
            let poly = 1.0 / 13.0 + u2 * poly;
            let poly = 1.0 / 11.0 + u2 * poly;
            let poly = 1.0 / 9.0 + u2 * poly;
            let poly = 1.0 / 7.0 + u2 * poly;
            let poly = 1.0 / 5.0 + u2 * poly;
            let poly = 1.0 / 3.0 + u2 * poly;
            let poly = 1.0 + u2 * poly;
            let ln_m = 2.0 * u * poly;
            let ef = f64::from(exp);
            (ef * LN2_HI) + (ln_m + ef * LN2_LO)
        }

        /// `x` için üstel fonksiyon (`exp(x)`, `< 1 ULP`).
        #[inline]
        pub(crate) fn exp(x: f64) -> f64 {
            if x.is_nan() {
                return f64::NAN;
            }
            if x > 709.782_712_893_384 {
                return f64::INFINITY;
            }
            if x < -745.133_219_101_941_2 {
                return 0.0;
            }
            let kf = x * INV_LN2;
            let k = if kf >= 0.0 {
                (kf + 0.5) as i32
            } else {
                (kf - 0.5) as i32
            };
            let kd = f64::from(k);
            let r = (x - kd * LN2_HI) - kd * LN2_LO;
            // |r| <= ln(2)/2 ≈ 0.3466 ⇒ 13. derece Taylor < 4e-18
            let mut p = 1.0 / 6_227_020_800.0; // 1/13!
            p = 1.0 / 479_001_600.0 + r * p; // 1/12!
            p = 1.0 / 39_916_800.0 + r * p; // 1/11!
            p = 1.0 / 3_628_800.0 + r * p; // 1/10!
            p = 1.0 / 362_880.0 + r * p; // 1/9!
            p = 1.0 / 40_320.0 + r * p; // 1/8!
            p = 1.0 / 5_040.0 + r * p; // 1/7!
            p = 1.0 / 720.0 + r * p; // 1/6!
            p = 1.0 / 120.0 + r * p; // 1/5!
            p = 1.0 / 24.0 + r * p; // 1/4!
            p = 1.0 / 6.0 + r * p; // 1/3!
            p = 0.5 + r * p; // 1/2!
            p = 1.0 + r * p;
            let er = 1.0 + r * p;
            // 2^k ile çarp (alt-normal aralığı için iki aşamalı ölçekleme).
            if k >= -1022 {
                er * f64::from_bits(((k + 1023) as u64) << 52)
            } else {
                let half = k / 2;
                let rem = k - half;
                er * f64::from_bits(((half + 1023) as u64) << 52)
                    * f64::from_bits(((rem + 1023) as u64) << 52)
            }
        }

        /// `x >= 0` için kuvvet (`x^y`, `< 1 ULP` tipik hassasiyet).
        #[inline]
        pub(crate) fn powf(x: f64, y: f64) -> f64 {
            if y == 0.0 || x == 1.0 {
                return 1.0;
            }
            if x == 0.0 {
                return if y > 0.0 { 0.0 } else { f64::INFINITY };
            }
            if x <= 0.0 || x.is_nan() {
                return f64::NAN;
            }
            exp(y * ln(x))
        }

        /// Sıfırdan uzağa yarı-yuvarlama (`f32::round` karşılığı).
        #[inline]
        pub(crate) fn round_f32(x: f32) -> f32 {
            if !x.is_finite() || x.abs() >= 8_388_608.0 {
                return x;
            }
            if x >= 0.0 {
                ((x + 0.5) as i64) as f32
            } else {
                ((x - 0.5) as i64) as f32
            }
        }
    }

    #[inline]
    pub(super) fn sqrt(x: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            x.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            nostd::sqrt(x)
        }
    }

    #[inline]
    pub(super) fn ln(x: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            x.ln()
        }
        #[cfg(not(feature = "std"))]
        {
            nostd::ln(x)
        }
    }

    #[inline]
    pub(super) fn exp(x: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            x.exp()
        }
        #[cfg(not(feature = "std"))]
        {
            nostd::exp(x)
        }
    }

    #[inline]
    pub(super) fn powf(x: f64, y: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            x.powf(y)
        }
        #[cfg(not(feature = "std"))]
        {
            nostd::powf(x, y)
        }
    }

    #[inline]
    pub(super) fn round_f32(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.round()
        }
        #[cfg(not(feature = "std"))]
        {
            nostd::round_f32(x)
        }
    }
}

/// SMPTE ST 2084 (PQ) ters EOTF: doğrusal olmayan `E' ∈ [0, 1]` → doğrusal `E ∈ [0, 1]`.
#[inline]
fn pq_eotf(e_prime: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = (2523.0 / 4096.0) * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = (2413.0 / 4096.0) * 32.0;
    const C3: f64 = (2392.0 / 4096.0) * 32.0;

    let ep = e_prime.clamp(0.0, 1.0);
    if ep == 0.0 {
        return 0.0;
    }
    let p = math::powf(ep, 1.0 / M2);
    let num = (p - C1).max(0.0);
    let den = C2 - C3 * p;
    if den <= 0.0 {
        1.0
    } else {
        math::powf(num / den, 1.0 / M1).clamp(0.0, 1.0)
    }
}

/// SMPTE ST 2084 (PQ) ileri OETF: doğrusal `E ∈ [0, 1]` → doğrusal olmayan `E' ∈ [0, 1]`.
#[inline]
fn pq_oetf(linear: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = (2523.0 / 4096.0) * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = (2413.0 / 4096.0) * 32.0;
    const C3: f64 = (2392.0 / 4096.0) * 32.0;

    let lin = linear.clamp(0.0, 1.0);
    if lin == 0.0 {
        return 0.0;
    }
    let p = math::powf(lin, M1);
    math::powf((C1 + C2 * p) / (1.0 + C3 * p), M2).clamp(0.0, 1.0)
}

/// ITU-R BT.2100 HLG ters OETF: doğrusal olmayan `E' ∈ [0, 1]` → doğrusal `E ∈ [0, 1]`.
#[inline]
fn hlg_inv_oetf(e_prime: f64) -> f64 {
    const A: f64 = 0.17883277;
    const B: f64 = 0.28466892;
    const C: f64 = 0.55991073;

    let ep = e_prime.clamp(0.0, 1.0);
    if ep <= 0.5 {
        (ep * ep) / 3.0
    } else {
        ((math::exp((ep - C) / A) + B) / 12.0).clamp(0.0, 1.0)
    }
}

/// ITU-R BT.2100 HLG ileri OETF: doğrusal `E ∈ [0, 1]` → doğrusal olmayan `E' ∈ [0, 1]`.
#[inline]
fn hlg_oetf(linear: f64) -> f64 {
    const A: f64 = 0.17883277;
    const B: f64 = 0.28466892;
    const C: f64 = 0.55991073;

    let lin = linear.clamp(0.0, 1.0);
    if lin <= 1.0 / 12.0 {
        math::sqrt(3.0 * lin)
    } else {
        (A * math::ln(12.0 * lin - B) + C).clamp(0.0, 1.0)
    }
}

/// ITU-R BT.2020 OETF (`transfer_characteristics` `14` / `15`): doğrusal `E ∈ [0, 1]` → `E' ∈ [0, 1]`.
#[inline]
fn bt2020_oetf(linear: f64) -> f64 {
    const ALPHA: f64 = 1.09929682680944;
    const BETA: f64 = 0.018053968510807;

    let lin = linear.clamp(0.0, 1.0);
    if lin < BETA {
        4.5 * lin
    } else {
        (ALPHA * math::powf(lin, 0.45) - (ALPHA - 1.0)).clamp(0.0, 1.0)
    }
}

/// ITU-R BT.2020 ters OETF: doğrusal olmayan `E' ∈ [0, 1]` → doğrusal `E ∈ [0, 1]`.
#[inline]
fn bt2020_inv_oetf(e_prime: f64) -> f64 {
    const ALPHA: f64 = 1.09929682680944;
    const BETA_PRIME: f64 = 4.5 * 0.018053968510807;

    let ep = e_prime.clamp(0.0, 1.0);
    if ep < BETA_PRIME {
        ep / 4.5
    } else {
        math::powf((ep + (ALPHA - 1.0)) / ALPHA, 1.0 / 0.45).clamp(0.0, 1.0)
    }
}

/// `Identity` (`0`), `YCgCo` (`8`), `YCgCo-Re/Ro` (`16`/`17`), `BT.2020 CL` (`10`) ve
/// `ICtCp` (`14`) matrisleri.
#[derive(Clone, Copy)]
enum SpecialMatrix {
    Identity,
    YCgCo,
    YCgCoR {
        yuv_max_f: f32,
        rgb_max_f: f32,
        max_rgb: i32,
    },
    /// ITU-T H.273 Denklem 58–67 (`matrix_coefficients == 10`, BT.2020 Constant Luminance).
    Bt2020Cl {
        is_pq: bool,
        two_nb: f64,
        two_pb: f64,
        two_nr: f64,
        two_pr: f64,
    },
    /// ITU-T H.273 §8.3 / ITU-R BT.2100 (`matrix_coefficients == 14`, ICtCp PQ/HLG).
    ICtCp {
        is_hlg: bool,
    },
}

impl SpecialMatrix {
    fn bt2020_cl(transfer_characteristics: u16) -> Self {
        const KR: f64 = 0.2627;
        const KB: f64 = 0.0593;
        let is_pq = transfer_characteristics == 16;
        let oetf = if is_pq { pq_oetf } else { bt2020_oetf };
        let nb = oetf(1.0 - KB);
        let pb = 1.0 - oetf(KB);
        let nr = oetf(1.0 - KR);
        let pr = 1.0 - oetf(KR);
        Self::Bt2020Cl {
            is_pq,
            two_nb: 2.0 * nb,
            two_pb: 2.0 * pb,
            two_nr: 2.0 * nr,
            two_pr: 2.0 * pr,
        }
    }
}

impl MatrixKernel for SpecialMatrix {
    #[inline(always)]
    fn to_rgb_f32(&self, unorm_y: u16, y_norm: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
        let (r, g, b) = match *self {
            Self::Identity => (cr, y_norm, cb),
            Self::YCgCo => {
                let t = y_norm - cb;
                (t + cr, y_norm + cb, t - cr)
            }
            Self::YCgCoR {
                yuv_max_f,
                rgb_max_f,
                max_rgb,
            } => {
                let yy = i32::from(unorm_y);
                let cg = math::round_f32(cb * yuv_max_f) as i32;
                let co = math::round_f32(cr * yuv_max_f) as i32;
                let t = yy - (cg >> 1);
                let g_i = (t + cg).clamp(0, max_rgb);
                let b_i = (t - (co >> 1)).clamp(0, max_rgb);
                let r_i = (b_i + co).clamp(0, max_rgb);
                (
                    (r_i as f32) / rgb_max_f,
                    (g_i as f32) / rgb_max_f,
                    (b_i as f32) / rgb_max_f,
                )
            }
            Self::Bt2020Cl {
                is_pq,
                two_nb,
                two_pb,
                two_nr,
                two_pr,
            } => {
                const KR: f64 = 0.2627;
                const KB: f64 = 0.0593;
                const KG: f64 = 0.6780;

                let yc_p = f64::from(y_norm);
                let cbc = f64::from(cb);
                let crc = f64::from(cr);

                let b_p = if cbc <= 0.0 {
                    cbc * two_nb + yc_p
                } else {
                    cbc * two_pb + yc_p
                }
                .clamp(0.0, 1.0);
                let r_p = if crc <= 0.0 {
                    crc * two_nr + yc_p
                } else {
                    crc * two_pr + yc_p
                }
                .clamp(0.0, 1.0);

                let (y_lin, b_lin, r_lin) = if is_pq {
                    (pq_eotf(yc_p), pq_eotf(b_p), pq_eotf(r_p))
                } else {
                    (
                        bt2020_inv_oetf(yc_p),
                        bt2020_inv_oetf(b_p),
                        bt2020_inv_oetf(r_p),
                    )
                };
                let g_lin = ((y_lin - KR * r_lin - KB * b_lin) / KG).clamp(0.0, 1.0);
                let g_p = if is_pq {
                    pq_oetf(g_lin)
                } else {
                    bt2020_oetf(g_lin)
                };
                (r_p as f32, g_p as f32, b_p as f32)
            }
            Self::ICtCp { is_hlg } => {
                let i = f64::from(y_norm);
                let ct = f64::from(cb);
                let cp = f64::from(cr);

                let (l_p, m_p, s_p) = if is_hlg {
                    (
                        i + 0.015_718_580_108_730_4 * ct + 0.209_581_068_116_405_5 * cp,
                        i - 0.015_718_580_108_730_4 * ct - 0.209_581_068_116_405_5 * cp,
                        i + 1.021_271_079_842_234_2 * ct - 0.605_274_490_992_431_5 * cp,
                    )
                } else {
                    (
                        i + 0.008_609_037_037_932_8 * ct + 0.111_029_625_003_026 * cp,
                        i - 0.008_609_037_037_932_8 * ct - 0.111_029_625_003_026 * cp,
                        i + 0.560_031_335_710_679_1 * ct - 0.320_627_174_987_318_9 * cp,
                    )
                };

                let (l, m, s) = if is_hlg {
                    (hlg_inv_oetf(l_p), hlg_inv_oetf(m_p), hlg_inv_oetf(s_p))
                } else {
                    (pq_eotf(l_p), pq_eotf(m_p), pq_eotf(s_p))
                };

                // Doğrusal LMS -> doğrusal BT.2020 RGB (ITU-R BT.2100-2 matrisinin tam tersi)
                let r_lin = (3.436_606_694_333_078_4 * l - 2.506_452_118_656_27 * m
                    + 0.069_845_424_323_191_5 * s)
                    .clamp(0.0, 1.0);
                let g_lin = (-0.791_329_555_598_928_7 * l + 1.983_600_451_792_290_7 * m
                    - 0.192_270_896_193_362 * s)
                    .clamp(0.0, 1.0);
                let b_lin = (-0.025_949_899_690_592_7 * l - 0.098_913_714_711_726_4 * m
                    + 1.124_863_614_402_319_2 * s)
                    .clamp(0.0, 1.0);

                if is_hlg {
                    (
                        hlg_oetf(r_lin) as f32,
                        hlg_oetf(g_lin) as f32,
                        hlg_oetf(b_lin) as f32,
                    )
                } else {
                    (
                        pq_oetf(r_lin) as f32,
                        pq_oetf(g_lin) as f32,
                        pq_oetf(b_lin) as f32,
                    )
                }
            }
        };
        (clamp01(r), clamp01(g), clamp01(b))
    }
}

/// Paketlenmiş çıktı tamponu piksel yazıcısı (`RGB8`, `RGBA8`, `RGB16`, `RGBA16`).
trait PixelSink: Copy + Send + Sync {
    type Channel: Copy + Default + Send;
    type RowAlpha<'r>: Copy
    where
        Self: 'r;

    const CHANNELS: usize;

    fn row_alpha(&self, row: usize) -> Self::RowAlpha<'_>;

    fn write_rgb(
        &self,
        dst: &mut [Self::Channel],
        r: f32,
        g: f32,
        b: f32,
        row_alpha: Self::RowAlpha<'_>,
        x: usize,
    );

    fn write_gray(
        &self,
        dst: &mut [Self::Channel],
        gray: f32,
        row_alpha: Self::RowAlpha<'_>,
        x: usize,
    );
}

#[derive(Clone, Copy)]
struct Rgb8Sink;

impl PixelSink for Rgb8Sink {
    type Channel = u8;
    type RowAlpha<'r> = ();
    const CHANNELS: usize = 3;

    #[inline(always)]
    fn row_alpha(&self, _row: usize) {}

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u8], r: f32, g: f32, b: f32, _row_alpha: (), _x: usize) {
        dst[0] = (0.5f32 + (r * 255.0f32)) as u8;
        dst[1] = (0.5f32 + (g * 255.0f32)) as u8;
        dst[2] = (0.5f32 + (b * 255.0f32)) as u8;
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u8], gray: f32, _row_alpha: (), _x: usize) {
        let q = (0.5f32 + (gray * 255.0f32)) as u8;
        dst[0] = q;
        dst[1] = q;
        dst[2] = q;
    }
}

#[derive(Clone, Copy)]
struct Rgba8OpaqueSink;

impl PixelSink for Rgba8OpaqueSink {
    type Channel = u8;
    type RowAlpha<'r> = ();
    const CHANNELS: usize = 4;

    #[inline(always)]
    fn row_alpha(&self, _row: usize) {}

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u8], r: f32, g: f32, b: f32, _row_alpha: (), _x: usize) {
        dst[0] = (0.5f32 + (r * 255.0f32)) as u8;
        dst[1] = (0.5f32 + (g * 255.0f32)) as u8;
        dst[2] = (0.5f32 + (b * 255.0f32)) as u8;
        dst[3] = 255;
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u8], gray: f32, _row_alpha: (), _x: usize) {
        let q = (0.5f32 + (gray * 255.0f32)) as u8;
        dst[0] = q;
        dst[1] = q;
        dst[2] = q;
        dst[3] = 255;
    }
}

#[derive(Clone, Copy)]
struct Rgba8AlphaU8Sink<'a> {
    alpha: &'a [u8],
    stride: usize,
}

impl<'a> PixelSink for Rgba8AlphaU8Sink<'a> {
    type Channel = u8;
    type RowAlpha<'r>
        = &'a [u8]
    where
        Self: 'r;
    const CHANNELS: usize = 4;

    #[inline(always)]
    fn row_alpha(&self, row: usize) -> &'a [u8] {
        let start = row * self.stride;
        &self.alpha[start..start + self.stride]
    }

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u8], r: f32, g: f32, b: f32, row_alpha: &'a [u8], x: usize) {
        dst[0] = (0.5f32 + (r * 255.0f32)) as u8;
        dst[1] = (0.5f32 + (g * 255.0f32)) as u8;
        dst[2] = (0.5f32 + (b * 255.0f32)) as u8;
        dst[3] = row_alpha[x];
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u8], gray: f32, row_alpha: &'a [u8], x: usize) {
        let q = (0.5f32 + (gray * 255.0f32)) as u8;
        dst[0] = q;
        dst[1] = q;
        dst[2] = q;
        dst[3] = row_alpha[x];
    }
}

#[derive(Clone, Copy)]
struct Rgba8AlphaU16Sink<'a> {
    alpha: &'a [u16],
    stride: usize,
    max_val: usize,
    lut: &'a [u8],
}

impl<'a> PixelSink for Rgba8AlphaU16Sink<'a> {
    type Channel = u8;
    type RowAlpha<'r>
        = &'a [u16]
    where
        Self: 'r;
    const CHANNELS: usize = 4;

    #[inline(always)]
    fn row_alpha(&self, row: usize) -> &'a [u16] {
        let start = row * self.stride;
        &self.alpha[start..start + self.stride]
    }

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u8], r: f32, g: f32, b: f32, row_alpha: &'a [u16], x: usize) {
        dst[0] = (0.5f32 + (r * 255.0f32)) as u8;
        dst[1] = (0.5f32 + (g * 255.0f32)) as u8;
        dst[2] = (0.5f32 + (b * 255.0f32)) as u8;
        dst[3] = self.lut[usize::from(row_alpha[x]).min(self.max_val)];
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u8], gray: f32, row_alpha: &'a [u16], x: usize) {
        let q = (0.5f32 + (gray * 255.0f32)) as u8;
        dst[0] = q;
        dst[1] = q;
        dst[2] = q;
        dst[3] = self.lut[usize::from(row_alpha[x]).min(self.max_val)];
    }
}

#[derive(Clone, Copy)]
struct Rgb16Sink {
    max_f: f32,
}

impl PixelSink for Rgb16Sink {
    type Channel = u16;
    type RowAlpha<'r> = ();
    const CHANNELS: usize = 3;

    #[inline(always)]
    fn row_alpha(&self, _row: usize) {}

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u16], r: f32, g: f32, b: f32, _row_alpha: (), _x: usize) {
        dst[0] = (0.5f32 + (r * self.max_f)) as u16;
        dst[1] = (0.5f32 + (g * self.max_f)) as u16;
        dst[2] = (0.5f32 + (b * self.max_f)) as u16;
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u16], gray: f32, _row_alpha: (), _x: usize) {
        let q = (0.5f32 + (gray * self.max_f)) as u16;
        dst[0] = q;
        dst[1] = q;
        dst[2] = q;
    }
}

#[derive(Clone, Copy)]
struct Rgba16OpaqueSink {
    max_f: f32,
    opaque_alpha: u16,
}

impl PixelSink for Rgba16OpaqueSink {
    type Channel = u16;
    type RowAlpha<'r> = ();
    const CHANNELS: usize = 4;

    #[inline(always)]
    fn row_alpha(&self, _row: usize) {}

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u16], r: f32, g: f32, b: f32, _row_alpha: (), _x: usize) {
        dst[0] = (0.5f32 + (r * self.max_f)) as u16;
        dst[1] = (0.5f32 + (g * self.max_f)) as u16;
        dst[2] = (0.5f32 + (b * self.max_f)) as u16;
        dst[3] = self.opaque_alpha;
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u16], gray: f32, _row_alpha: (), _x: usize) {
        let q = (0.5f32 + (gray * self.max_f)) as u16;
        dst[0] = q;
        dst[1] = q;
        dst[2] = q;
        dst[3] = self.opaque_alpha;
    }
}

#[derive(Clone, Copy)]
struct Rgba16AlphaU8Sink<'a> {
    max_f: f32,
    alpha: &'a [u8],
    stride: usize,
}

impl<'a> PixelSink for Rgba16AlphaU8Sink<'a> {
    type Channel = u16;
    type RowAlpha<'r>
        = &'a [u8]
    where
        Self: 'r;
    const CHANNELS: usize = 4;

    #[inline(always)]
    fn row_alpha(&self, row: usize) -> &'a [u8] {
        let start = row * self.stride;
        &self.alpha[start..start + self.stride]
    }

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u16], r: f32, g: f32, b: f32, row_alpha: &'a [u8], x: usize) {
        dst[0] = (0.5f32 + (r * self.max_f)) as u16;
        dst[1] = (0.5f32 + (g * self.max_f)) as u16;
        dst[2] = (0.5f32 + (b * self.max_f)) as u16;
        dst[3] = u16::from(row_alpha[x]);
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u16], gray: f32, row_alpha: &'a [u8], x: usize) {
        let q = (0.5f32 + (gray * self.max_f)) as u16;
        dst[0] = q;
        dst[1] = q;
        dst[2] = q;
        dst[3] = u16::from(row_alpha[x]);
    }
}

#[derive(Clone, Copy)]
struct Rgba16AlphaU16Sink<'a> {
    max_f: f32,
    opaque_alpha: u16,
    alpha: &'a [u16],
    stride: usize,
}

impl<'a> PixelSink for Rgba16AlphaU16Sink<'a> {
    type Channel = u16;
    type RowAlpha<'r>
        = &'a [u16]
    where
        Self: 'r;
    const CHANNELS: usize = 4;

    #[inline(always)]
    fn row_alpha(&self, row: usize) -> &'a [u16] {
        let start = row * self.stride;
        &self.alpha[start..start + self.stride]
    }

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u16], r: f32, g: f32, b: f32, row_alpha: &'a [u16], x: usize) {
        dst[0] = (0.5f32 + (r * self.max_f)) as u16;
        dst[1] = (0.5f32 + (g * self.max_f)) as u16;
        dst[2] = (0.5f32 + (b * self.max_f)) as u16;
        dst[3] = row_alpha[x].min(self.opaque_alpha);
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u16], gray: f32, row_alpha: &'a [u16], x: usize) {
        let q = (0.5f32 + (gray * self.max_f)) as u16;
        dst[0] = q;
        dst[1] = q;
        dst[2] = q;
        dst[3] = row_alpha[x].min(self.opaque_alpha);
    }
}

/// Etkin bit derinliğindeki (`8`/`10`/`12`-bit) örneği tam `[0..=65535]` 16-bit aralığına
/// yuvarlayarak ölçekler (`image::ColorType::Rgb16` / `Rgba16` sözleşmesi).
#[inline(always)]
const fn scale_sample_to_u16_full(bit_depth: BitDepth, v: u16) -> u16 {
    match bit_depth {
        BitDepth::Eight => v.saturating_mul(257),
        BitDepth::Ten => {
            let clamped = if v > 1023 { 1023 } else { v };
            ((clamped as u32 * 65_535 + 511) / 1023) as u16
        }
        BitDepth::Twelve => {
            let clamped = if v > 4095 { 4095 } else { v };
            ((clamped as u32 * 65_535 + 2047) / 4095) as u16
        }
    }
}

#[derive(Clone, Copy)]
struct Rgb16ScaledNeSink {
    max_f: f32,
    bit_depth: BitDepth,
}

impl PixelSink for Rgb16ScaledNeSink {
    type Channel = u8;
    type RowAlpha<'r> = ();
    const CHANNELS: usize = 6;

    #[inline(always)]
    fn row_alpha(&self, _row: usize) {}

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u8], r: f32, g: f32, b: f32, _row_alpha: (), _x: usize) {
        let rq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (r * self.max_f)) as u16)
            .to_ne_bytes();
        let gq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (g * self.max_f)) as u16)
            .to_ne_bytes();
        let bq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (b * self.max_f)) as u16)
            .to_ne_bytes();
        dst[0] = rq[0];
        dst[1] = rq[1];
        dst[2] = gq[0];
        dst[3] = gq[1];
        dst[4] = bq[0];
        dst[5] = bq[1];
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u8], gray: f32, _row_alpha: (), _x: usize) {
        let q = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (gray * self.max_f)) as u16)
            .to_ne_bytes();
        dst[0] = q[0];
        dst[1] = q[1];
        dst[2] = q[0];
        dst[3] = q[1];
        dst[4] = q[0];
        dst[5] = q[1];
    }
}

#[derive(Clone, Copy)]
struct Rgba16ScaledNeOpaqueSink {
    max_f: f32,
    bit_depth: BitDepth,
}

impl PixelSink for Rgba16ScaledNeOpaqueSink {
    type Channel = u8;
    type RowAlpha<'r> = ();
    const CHANNELS: usize = 8;

    #[inline(always)]
    fn row_alpha(&self, _row: usize) {}

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u8], r: f32, g: f32, b: f32, _row_alpha: (), _x: usize) {
        let rq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (r * self.max_f)) as u16)
            .to_ne_bytes();
        let gq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (g * self.max_f)) as u16)
            .to_ne_bytes();
        let bq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (b * self.max_f)) as u16)
            .to_ne_bytes();
        let aq = 65_535u16.to_ne_bytes();
        dst[0] = rq[0];
        dst[1] = rq[1];
        dst[2] = gq[0];
        dst[3] = gq[1];
        dst[4] = bq[0];
        dst[5] = bq[1];
        dst[6] = aq[0];
        dst[7] = aq[1];
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u8], gray: f32, _row_alpha: (), _x: usize) {
        let q = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (gray * self.max_f)) as u16)
            .to_ne_bytes();
        let aq = 65_535u16.to_ne_bytes();
        dst[0] = q[0];
        dst[1] = q[1];
        dst[2] = q[0];
        dst[3] = q[1];
        dst[4] = q[0];
        dst[5] = q[1];
        dst[6] = aq[0];
        dst[7] = aq[1];
    }
}

#[derive(Clone, Copy)]
struct Rgba16ScaledNeAlphaU8Sink<'a> {
    max_f: f32,
    bit_depth: BitDepth,
    alpha: &'a [u8],
    stride: usize,
}

impl<'a> PixelSink for Rgba16ScaledNeAlphaU8Sink<'a> {
    type Channel = u8;
    type RowAlpha<'r>
        = &'a [u8]
    where
        Self: 'r;
    const CHANNELS: usize = 8;

    #[inline(always)]
    fn row_alpha(&self, row: usize) -> &'a [u8] {
        let start = row * self.stride;
        &self.alpha[start..start + self.stride]
    }

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u8], r: f32, g: f32, b: f32, row_alpha: &'a [u8], x: usize) {
        let rq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (r * self.max_f)) as u16)
            .to_ne_bytes();
        let gq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (g * self.max_f)) as u16)
            .to_ne_bytes();
        let bq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (b * self.max_f)) as u16)
            .to_ne_bytes();
        let aq = (u16::from(row_alpha[x]) * 257).to_ne_bytes();
        dst[0] = rq[0];
        dst[1] = rq[1];
        dst[2] = gq[0];
        dst[3] = gq[1];
        dst[4] = bq[0];
        dst[5] = bq[1];
        dst[6] = aq[0];
        dst[7] = aq[1];
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u8], gray: f32, row_alpha: &'a [u8], x: usize) {
        let q = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (gray * self.max_f)) as u16)
            .to_ne_bytes();
        let aq = (u16::from(row_alpha[x]) * 257).to_ne_bytes();
        dst[0] = q[0];
        dst[1] = q[1];
        dst[2] = q[0];
        dst[3] = q[1];
        dst[4] = q[0];
        dst[5] = q[1];
        dst[6] = aq[0];
        dst[7] = aq[1];
    }
}

#[derive(Clone, Copy)]
struct Rgba16ScaledNeAlphaU16Sink<'a> {
    max_f: f32,
    bit_depth: BitDepth,
    opaque_alpha: u16,
    alpha: &'a [u16],
    stride: usize,
}

impl<'a> PixelSink for Rgba16ScaledNeAlphaU16Sink<'a> {
    type Channel = u8;
    type RowAlpha<'r>
        = &'a [u16]
    where
        Self: 'r;
    const CHANNELS: usize = 8;

    #[inline(always)]
    fn row_alpha(&self, row: usize) -> &'a [u16] {
        let start = row * self.stride;
        &self.alpha[start..start + self.stride]
    }

    #[inline(always)]
    fn write_rgb(&self, dst: &mut [u8], r: f32, g: f32, b: f32, row_alpha: &'a [u16], x: usize) {
        let rq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (r * self.max_f)) as u16)
            .to_ne_bytes();
        let gq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (g * self.max_f)) as u16)
            .to_ne_bytes();
        let bq = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (b * self.max_f)) as u16)
            .to_ne_bytes();
        let aq = scale_sample_to_u16_full(self.bit_depth, row_alpha[x].min(self.opaque_alpha))
            .to_ne_bytes();
        dst[0] = rq[0];
        dst[1] = rq[1];
        dst[2] = gq[0];
        dst[3] = gq[1];
        dst[4] = bq[0];
        dst[5] = bq[1];
        dst[6] = aq[0];
        dst[7] = aq[1];
    }

    #[inline(always)]
    fn write_gray(&self, dst: &mut [u8], gray: f32, row_alpha: &'a [u16], x: usize) {
        let q = scale_sample_to_u16_full(self.bit_depth, (0.5f32 + (gray * self.max_f)) as u16)
            .to_ne_bytes();
        let aq = scale_sample_to_u16_full(self.bit_depth, row_alpha[x].min(self.opaque_alpha))
            .to_ne_bytes();
        dst[0] = q[0];
        dst[1] = q[1];
        dst[2] = q[0];
        dst[3] = q[1];
        dst[4] = q[0];
        dst[5] = q[1];
        dst[6] = aq[0];
        dst[7] = aq[1];
    }
}

#[inline(always)]
fn bilinear_col_indices(i: usize, w: usize, cw: usize) -> (usize, usize) {
    let uv_i = (i >> 1).min(cw - 1);
    let adj_col: isize = if i == 0 || (i == w - 1 && (i & 1) != 0) {
        0
    } else if (i & 1) != 0 {
        if uv_i + 1 < cw { 1 } else { 0 }
    } else if uv_i > 0 {
        -1
    } else {
        0
    };
    (uv_i, ((uv_i as isize) + adj_col) as usize)
}

#[inline(always)]
fn bilinear_row_indices(j: usize, h: usize, ch: usize) -> (usize, usize) {
    let uv_j = (j >> 1).min(ch - 1);
    let adj_row: isize = if j == 0 || (j == h - 1 && (j & 1) != 0) {
        0
    } else if (j & 1) != 0 {
        if uv_j + 1 < ch { 1 } else { 0 }
    } else if uv_j > 0 {
        -1
    } else {
        0
    };
    (uv_j, ((uv_j as isize) + adj_row) as usize)
}

/// Tek bir `4:2:0` veya `4:2:2` satırını önceden çözülmüş iki kroma satırı (`u_r0`/`u_r1`, `v_r0`/`v_r1`)
/// üzerinden sınır-denetimsiz iç sütun çiftleriyle dönüştürür (`(9, 3, 3, 1)/16` IEEE-754 özdeş).
#[allow(clippy::too_many_arguments)]
fn convert_bilinear_single_row<S: SampleSource, M: MatrixKernel, O: PixelSink>(
    y_row: &[S::Sample],
    u_r0: &[S::Sample],
    u_r1: &[S::Sample],
    v_r0: &[S::Sample],
    v_r1: &[S::Sample],
    out_row: &mut [O::Channel],
    j: usize,
    w: usize,
    cw: usize,
    src: S,
    matrix: M,
    sink: O,
) {
    let a_row = sink.row_alpha(j);

    // Sol kenar sütunu (i = 0, c0 = 0, c1 = 0).
    let u00 = src.lookup_uv(u_r0[0]);
    let u01 = src.lookup_uv(u_r1[0]);
    let v00 = src.lookup_uv(v_r0[0]);
    let v01 = src.lookup_uv(v_r1[0]);
    let (uy0, yn0) = src.lookup_y(y_row[0]);
    let cb0 = bilinear_4(u00, u00, u01, u01);
    let cr0 = bilinear_4(v00, v00, v01, v01);
    let (r0, g0, b0) = matrix.to_rgb_f32(uy0, yn0, cb0, cr0);
    sink.write_rgb(&mut out_row[..O::CHANNELS], r0, g0, b0, a_row, 0);

    let interior_pairs = if w >= 3 && cw >= 2 {
        ((w - 1) / 2).min(cw - 1)
    } else {
        0
    };

    if interior_pairs > 0 {
        let u0_l = &u_r0[..interior_pairs];
        let u0_r = &u_r0[1..=interior_pairs];
        let u1_l = &u_r1[..interior_pairs];
        let u1_r = &u_r1[1..=interior_pairs];
        let v0_l = &v_r0[..interior_pairs];
        let v0_r = &v_r0[1..=interior_pairs];
        let v1_l = &v_r1[..interior_pairs];
        let v1_r = &v_r1[1..=interior_pairs];
        let y_mid = &y_row[1..1 + interior_pairs * 2];
        let out_mid = &mut out_row[O::CHANNELS..(1 + interior_pairs * 2) * O::CHANNELS];

        for (k, (&y_pair, out_pair)) in y_mid
            .as_chunks::<2>()
            .0
            .iter()
            .zip(out_mid.chunks_exact_mut(2 * O::CHANNELS))
            .enumerate()
        {
            let tu00 = src.lookup_uv(u0_l[k]);
            let tu10 = src.lookup_uv(u0_r[k]);
            let tu01 = src.lookup_uv(u1_l[k]);
            let tu11 = src.lookup_uv(u1_r[k]);

            let tv00 = src.lookup_uv(v0_l[k]);
            let tv10 = src.lookup_uv(v0_r[k]);
            let tv01 = src.lookup_uv(v1_l[k]);
            let tv11 = src.lookup_uv(v1_r[k]);

            let x_l = 2 * k + 1;
            let (out_l, out_r) = out_pair.split_at_mut(O::CHANNELS);

            let (uy_l, yn_l) = src.lookup_y(y_pair[0]);
            let cb_l = bilinear_4(tu00, tu10, tu01, tu11);
            let cr_l = bilinear_4(tv00, tv10, tv01, tv11);
            let (rl, gl, bl) = matrix.to_rgb_f32(uy_l, yn_l, cb_l, cr_l);
            sink.write_rgb(out_l, rl, gl, bl, a_row, x_l);

            let (uy_r, yn_r) = src.lookup_y(y_pair[1]);
            let cb_r = bilinear_4(tu10, tu00, tu11, tu01);
            let cr_r = bilinear_4(tv10, tv00, tv11, tv01);
            let (rr, gr, br) = matrix.to_rgb_f32(uy_r, yn_r, cb_r, cr_r);
            sink.write_rgb(out_r, rr, gr, br, a_row, x_l + 1);
        }
    }

    // Sağ kenar artığı (ör. çift genişlikte i = w - 1).
    let tail_start = 1 + interior_pairs * 2;
    for (offset, &ys) in y_row[tail_start..w].iter().enumerate() {
        let i = tail_start + offset;
        let (c0, c1) = bilinear_col_indices(i, w, cw);
        let cb = bilinear_4(
            src.lookup_uv(u_r0[c0]),
            src.lookup_uv(u_r0[c1]),
            src.lookup_uv(u_r1[c0]),
            src.lookup_uv(u_r1[c1]),
        );
        let cr = bilinear_4(
            src.lookup_uv(v_r0[c0]),
            src.lookup_uv(v_r0[c1]),
            src.lookup_uv(v_r1[c0]),
            src.lookup_uv(v_r1[c1]),
        );
        let (uy, yn) = src.lookup_y(ys);
        let (r, g, b) = matrix.to_rgb_f32(uy, yn, cb, cr);
        let o = i * O::CHANNELS;
        sink.write_rgb(&mut out_row[o..o + O::CHANNELS], r, g, b, a_row, i);
    }
}

/// Ardışık `4:2:0` satır çiftini (`j_top = 2*r + 1`, `j_bot = 2*r + 2`) aynı 2×2 kroma
/// örneklerini ve önceden ölçeklenmiş ağırlık çarpımlarını paylaşarak dönüştürür.
#[allow(clippy::too_many_arguments)]
fn convert_c420_row_pair<S: SampleSource, M: MatrixKernel, O: PixelSink>(
    y_top: &[S::Sample],
    y_bot: &[S::Sample],
    u_r0: &[S::Sample],
    u_r1: &[S::Sample],
    v_r0: &[S::Sample],
    v_r1: &[S::Sample],
    out_top: &mut [O::Channel],
    out_bot: &mut [O::Channel],
    j_top: usize,
    w: usize,
    cw: usize,
    src: S,
    matrix: M,
    sink: O,
) {
    let a_top = sink.row_alpha(j_top);
    let a_bot = sink.row_alpha(j_top + 1);

    // Sol kenar sütunu (i = 0): r0 ve r1 satırları paylaşılır.
    let tu00 = src.lookup_uv(u_r0[0]);
    let tu01 = src.lookup_uv(u_r1[0]);
    let tv00 = src.lookup_uv(v_r0[0]);
    let tv01 = src.lookup_uv(v_r1[0]);

    let (uyt0, ynt0) = src.lookup_y(y_top[0]);
    let (rt0, gt0, bt0) = matrix.to_rgb_f32(
        uyt0,
        ynt0,
        bilinear_4(tu00, tu00, tu01, tu01),
        bilinear_4(tv00, tv00, tv01, tv01),
    );
    sink.write_rgb(&mut out_top[..O::CHANNELS], rt0, gt0, bt0, a_top, 0);

    let (uyb0, ynb0) = src.lookup_y(y_bot[0]);
    let (rb0, gb0, bb0) = matrix.to_rgb_f32(
        uyb0,
        ynb0,
        bilinear_4(tu01, tu01, tu00, tu00),
        bilinear_4(tv01, tv01, tv00, tv00),
    );
    sink.write_rgb(&mut out_bot[..O::CHANNELS], rb0, gb0, bb0, a_bot, 0);

    let interior_pairs = if w >= 3 && cw >= 2 {
        ((w - 1) / 2).min(cw - 1)
    } else {
        0
    };

    if interior_pairs > 0 {
        let u0_l = &u_r0[..interior_pairs];
        let u0_r = &u_r0[1..=interior_pairs];
        let u1_l = &u_r1[..interior_pairs];
        let u1_r = &u_r1[1..=interior_pairs];
        let v0_l = &v_r0[..interior_pairs];
        let v0_r = &v_r0[1..=interior_pairs];
        let v1_l = &v_r1[..interior_pairs];
        let v1_r = &v_r1[1..=interior_pairs];

        let y_top_mid = &y_top[1..1 + interior_pairs * 2];
        let y_bot_mid = &y_bot[1..1 + interior_pairs * 2];
        let out_top_mid = &mut out_top[O::CHANNELS..(1 + interior_pairs * 2) * O::CHANNELS];
        let out_bot_mid = &mut out_bot[O::CHANNELS..(1 + interior_pairs * 2) * O::CHANNELS];

        for (k, (((&yt_pair, &yb_pair), ot_pair), ob_pair)) in y_top_mid
            .as_chunks::<2>()
            .0
            .iter()
            .zip(y_bot_mid.as_chunks::<2>().0.iter())
            .zip(out_top_mid.chunks_exact_mut(2 * O::CHANNELS))
            .zip(out_bot_mid.chunks_exact_mut(2 * O::CHANNELS))
            .enumerate()
        {
            const W9: f32 = 9.0 / 16.0;
            const W3: f32 = 3.0 / 16.0;
            const W1: f32 = 1.0 / 16.0;

            let u00 = src.lookup_uv(u0_l[k]);
            let u10 = src.lookup_uv(u0_r[k]);
            let u01 = src.lookup_uv(u1_l[k]);
            let u11 = src.lookup_uv(u1_r[k]);

            let u00_9 = u00 * W9;
            let u00_3 = u00 * W3;
            let u00_1 = u00 * W1;
            let u10_9 = u10 * W9;
            let u10_3 = u10 * W3;
            let u10_1 = u10 * W1;
            let u01_9 = u01 * W9;
            let u01_3 = u01 * W3;
            let u01_1 = u01 * W1;
            let u11_9 = u11 * W9;
            let u11_3 = u11 * W3;
            let u11_1 = u11 * W1;

            let v00 = src.lookup_uv(v0_l[k]);
            let v10 = src.lookup_uv(v0_r[k]);
            let v01 = src.lookup_uv(v1_l[k]);
            let v11 = src.lookup_uv(v1_r[k]);

            let v00_9 = v00 * W9;
            let v00_3 = v00 * W3;
            let v00_1 = v00 * W1;
            let v10_9 = v10 * W9;
            let v10_3 = v10 * W3;
            let v10_1 = v10 * W1;
            let v01_9 = v01 * W9;
            let v01_3 = v01 * W3;
            let v01_1 = v01 * W1;
            let v11_9 = v11 * W9;
            let v11_3 = v11 * W3;
            let v11_1 = v11 * W1;

            let x_l = 2 * k + 1;
            let x_r = x_l + 1;
            let (ot_l, ot_r) = ot_pair.split_at_mut(O::CHANNELS);
            let (ob_l, ob_r) = ob_pair.split_at_mut(O::CHANNELS);

            // Piksel sırası üst-sol, üst-sağ, alt-sol, alt-sağ.
            let cb = [
                ((u00_9 + u10_3) + u01_3) + u11_1,
                ((u10_9 + u00_3) + u11_3) + u01_1,
                ((u01_9 + u11_3) + u00_3) + u10_1,
                ((u11_9 + u01_3) + u10_3) + u00_1,
            ];
            let cr = [
                ((v00_9 + v10_3) + v01_3) + v11_1,
                ((v10_9 + v00_3) + v11_3) + v01_1,
                ((v01_9 + v11_3) + v00_3) + v10_1,
                ((v11_9 + v01_3) + v10_3) + v00_1,
            ];
            let y = [
                src.lookup_y(yt_pair[0]),
                src.lookup_y(yt_pair[1]),
                src.lookup_y(yb_pair[0]),
                src.lookup_y(yb_pair[1]),
            ];
            let [
                (rtl, gtl, btl),
                (rtr, gtr, btr),
                (rbl, gbl, bbl),
                (rbr, gbr, bbr),
            ] = matrix.to_rgb_f32x4(y, cb, cr);
            sink.write_rgb(ot_l, rtl, gtl, btl, a_top, x_l);
            sink.write_rgb(ot_r, rtr, gtr, btr, a_top, x_r);
            sink.write_rgb(ob_l, rbl, gbl, bbl, a_bot, x_l);
            sink.write_rgb(ob_r, rbr, gbr, bbr, a_bot, x_r);
        }
    }

    // Sağ kenar artığı.
    let tail_start = 1 + interior_pairs * 2;
    for i in tail_start..w {
        let (c0, c1) = bilinear_col_indices(i, w, cw);
        let u00 = src.lookup_uv(u_r0[c0]);
        let u10 = src.lookup_uv(u_r0[c1]);
        let u01 = src.lookup_uv(u_r1[c0]);
        let u11 = src.lookup_uv(u_r1[c1]);
        let v00 = src.lookup_uv(v_r0[c0]);
        let v10 = src.lookup_uv(v_r0[c1]);
        let v01 = src.lookup_uv(v_r1[c0]);
        let v11 = src.lookup_uv(v_r1[c1]);

        let o = i * O::CHANNELS;
        let (uyt, ynt) = src.lookup_y(y_top[i]);
        let (rt, gt, bt) = matrix.to_rgb_f32(
            uyt,
            ynt,
            bilinear_4(u00, u10, u01, u11),
            bilinear_4(v00, v10, v01, v11),
        );
        sink.write_rgb(&mut out_top[o..o + O::CHANNELS], rt, gt, bt, a_top, i);

        let (uyb, ynb) = src.lookup_y(y_bot[i]);
        let (rb, gb, bb) = matrix.to_rgb_f32(
            uyb,
            ynb,
            bilinear_4(u01, u11, u00, u10),
            bilinear_4(v01, v11, v00, v10),
        );
        sink.write_rgb(&mut out_bot[o..o + O::CHANNELS], rb, gb, bb, a_bot, i);
    }
}

/// Bir şeritteki (`y_start..y_end`) tüm satırları kroma biçimine göre özelleşmiş yoldan dönüştürür.
#[allow(clippy::too_many_arguments)]
fn convert_strip<S: SampleSource, M: MatrixKernel, O: PixelSink>(
    y_plane: &[S::Sample],
    u_plane: &[S::Sample],
    v_plane: &[S::Sample],
    out_strip: &mut [O::Channel],
    y_start: usize,
    y_end: usize,
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    chroma_format: ChromaFormat,
    src: S,
    matrix: M,
    sink: O,
) {
    let row_stride = w * O::CHANNELS;
    if chroma_format == ChromaFormat::Monochrome {
        for (local_y, out_row) in out_strip.chunks_exact_mut(row_stride).enumerate() {
            let j = y_start + local_y;
            let y_row = &y_plane[j * w..(j + 1) * w];
            let a_row = sink.row_alpha(j);
            for (x, (&ys, out_px)) in y_row
                .iter()
                .zip(out_row.chunks_exact_mut(O::CHANNELS))
                .enumerate()
            {
                let (_, yn) = src.lookup_y(ys);
                sink.write_gray(out_px, clamp01(yn), a_row, x);
            }
        }
        return;
    }

    if cw == 0 || ch == 0 {
        for (local_y, out_row) in out_strip.chunks_exact_mut(row_stride).enumerate() {
            let j = y_start + local_y;
            let y_row = &y_plane[j * w..(j + 1) * w];
            let a_row = sink.row_alpha(j);
            for (x, (&ys, out_px)) in y_row
                .iter()
                .zip(out_row.chunks_exact_mut(O::CHANNELS))
                .enumerate()
            {
                let (uy, yn) = src.lookup_y(ys);
                let (r, g, b) = matrix.to_rgb_f32(uy, yn, 0.0, 0.0);
                sink.write_rgb(out_px, r, g, b, a_row, x);
            }
        }
        return;
    }

    match chroma_format {
        ChromaFormat::C444 => {
            for (local_y, out_row) in out_strip.chunks_exact_mut(row_stride).enumerate() {
                let j = y_start + local_y;
                let uv_j = j.min(ch - 1);
                let y_row = &y_plane[j * w..(j + 1) * w];
                let u_row = &u_plane[uv_j * cw..(uv_j + 1) * cw];
                let v_row = &v_plane[uv_j * cw..(uv_j + 1) * cw];
                let a_row = sink.row_alpha(j);
                if cw >= w {
                    for (x, (((&ys, &us), &vs), out_px)) in y_row
                        .iter()
                        .zip(&u_row[..w])
                        .zip(&v_row[..w])
                        .zip(out_row.chunks_exact_mut(O::CHANNELS))
                        .enumerate()
                    {
                        let (uy, yn) = src.lookup_y(ys);
                        let cb = src.lookup_uv(us);
                        let cr = src.lookup_uv(vs);
                        let (r, g, b) = matrix.to_rgb_f32(uy, yn, cb, cr);
                        sink.write_rgb(out_px, r, g, b, a_row, x);
                    }
                } else {
                    for (x, (&ys, out_px)) in y_row
                        .iter()
                        .zip(out_row.chunks_exact_mut(O::CHANNELS))
                        .enumerate()
                    {
                        let uv_i = x.min(cw - 1);
                        let (uy, yn) = src.lookup_y(ys);
                        let cb = src.lookup_uv(u_row[uv_i]);
                        let cr = src.lookup_uv(v_row[uv_i]);
                        let (r, g, b) = matrix.to_rgb_f32(uy, yn, cb, cr);
                        sink.write_rgb(out_px, r, g, b, a_row, x);
                    }
                }
            }
        }
        ChromaFormat::C422 => {
            for (local_y, out_row) in out_strip.chunks_exact_mut(row_stride).enumerate() {
                let j = y_start + local_y;
                let uv_j = j.min(ch - 1);
                let y_row = &y_plane[j * w..(j + 1) * w];
                let u_row = &u_plane[uv_j * cw..(uv_j + 1) * cw];
                let v_row = &v_plane[uv_j * cw..(uv_j + 1) * cw];
                convert_bilinear_single_row(
                    y_row, u_row, u_row, v_row, v_row, out_row, j, w, cw, src, matrix, sink,
                );
            }
        }
        ChromaFormat::C420 => {
            let mut j = y_start;
            let mut out_rem = out_strip;

            // Başlangıç satırı çiftse tek satır olarak işle; ardından tek-çift (2r+1, 2r+2) çiftlerine geç.
            if j < y_end && (j & 1) == 0 {
                let (out_row, rest) = out_rem.split_at_mut(row_stride);
                let (r0, r1) = bilinear_row_indices(j, h, ch);
                convert_bilinear_single_row(
                    &y_plane[j * w..(j + 1) * w],
                    &u_plane[r0 * cw..(r0 + 1) * cw],
                    &u_plane[r1 * cw..(r1 + 1) * cw],
                    &v_plane[r0 * cw..(r0 + 1) * cw],
                    &v_plane[r1 * cw..(r1 + 1) * cw],
                    out_row,
                    j,
                    w,
                    cw,
                    src,
                    matrix,
                    sink,
                );
                j += 1;
                out_rem = rest;
            }

            while j + 1 < y_end {
                let r = j >> 1;
                if r + 1 < ch {
                    let (pair_buf, rest) = out_rem.split_at_mut(row_stride * 2);
                    let (out_top, out_bot) = pair_buf.split_at_mut(row_stride);
                    convert_c420_row_pair(
                        &y_plane[j * w..(j + 1) * w],
                        &y_plane[(j + 1) * w..(j + 2) * w],
                        &u_plane[r * cw..(r + 1) * cw],
                        &u_plane[(r + 1) * cw..(r + 2) * cw],
                        &v_plane[r * cw..(r + 1) * cw],
                        &v_plane[(r + 1) * cw..(r + 2) * cw],
                        out_top,
                        out_bot,
                        j,
                        w,
                        cw,
                        src,
                        matrix,
                        sink,
                    );
                    j += 2;
                    out_rem = rest;
                } else {
                    break;
                }
            }

            while j < y_end {
                let (out_row, rest) = out_rem.split_at_mut(row_stride);
                let (r0, r1) = bilinear_row_indices(j, h, ch);
                convert_bilinear_single_row(
                    &y_plane[j * w..(j + 1) * w],
                    &u_plane[r0 * cw..(r0 + 1) * cw],
                    &u_plane[r1 * cw..(r1 + 1) * cw],
                    &v_plane[r0 * cw..(r0 + 1) * cw],
                    &v_plane[r1 * cw..(r1 + 1) * cw],
                    out_row,
                    j,
                    w,
                    cw,
                    src,
                    matrix,
                    sink,
                );
                j += 1;
                out_rem = rest;
            }
        }
        ChromaFormat::Monochrome => {}
    }
}

/// Tüm görüntüyü tek iş parçacığıyla veya büyük görüntülerde `std::thread::scope` şeritleriyle dönüştürür.
#[allow(clippy::too_many_arguments)]
fn convert_image_strips<S: SampleSource, M: MatrixKernel, O: PixelSink>(
    y_plane: &[S::Sample],
    u_plane: &[S::Sample],
    v_plane: &[S::Sample],
    pixels: &mut [O::Channel],
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    chroma_format: ChromaFormat,
    effective_threads: usize,
    src: S,
    matrix: M,
    sink: O,
) {
    if w == 0 || h == 0 {
        return;
    }

    #[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
    let _ = effective_threads;

    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    {
        let num_pixels = w * h;
        let threads = effective_threads.max(1);
        if threads > 1 && num_pixels >= PARALLEL_PIXEL_THRESHOLD && h >= 4 {
            let num_strips = threads.min(h / 2);
            let row_stride = w * O::CHANNELS;
            // İç şerit sınırlarını tek satır indeksine (2k + 1) hizalayarak 4:2:0 satır çifti
            // hizalamasını koru.
            let mut strips: Vec<(usize, usize, &mut [O::Channel])> = Vec::with_capacity(num_strips);
            let mut rem: &mut [O::Channel] = pixels;
            let mut cur_y = 0usize;
            for s_idx in 0..num_strips {
                let next_y = if s_idx + 1 == num_strips {
                    h
                } else {
                    let raw = ((s_idx + 1) * h) / num_strips;
                    let aligned = if (raw & 1) == 0 { raw + 1 } else { raw };
                    aligned.min(h)
                };
                if next_y <= cur_y {
                    continue;
                }
                let rows = next_y - cur_y;
                let (head, tail) = rem.split_at_mut(rows * row_stride);
                strips.push((cur_y, next_y, head));
                rem = tail;
                cur_y = next_y;
            }

            if strips.len() > 1 {
                let last = strips.pop();
                std::thread::scope(|scope| {
                    for (ys, ye, strip_buf) in strips {
                        scope.spawn(move || {
                            convert_strip(
                                y_plane,
                                u_plane,
                                v_plane,
                                strip_buf,
                                ys,
                                ye,
                                w,
                                h,
                                cw,
                                ch,
                                chroma_format,
                                src,
                                matrix,
                                sink,
                            );
                        });
                    }
                    if let Some((ys, ye, strip_buf)) = last {
                        convert_strip(
                            y_plane,
                            u_plane,
                            v_plane,
                            strip_buf,
                            ys,
                            ye,
                            w,
                            h,
                            cw,
                            ch,
                            chroma_format,
                            src,
                            matrix,
                            sink,
                        );
                    }
                });
                return;
            }
            if let Some((ys, ye, strip_buf)) = strips.into_iter().next() {
                convert_strip(
                    y_plane,
                    u_plane,
                    v_plane,
                    strip_buf,
                    ys,
                    ye,
                    w,
                    h,
                    cw,
                    ch,
                    chroma_format,
                    src,
                    matrix,
                    sink,
                );
                return;
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    let _ = effective_threads;

    convert_strip(
        y_plane,
        u_plane,
        v_plane,
        pixels,
        0,
        h,
        w,
        h,
        cw,
        ch,
        chroma_format,
        src,
        matrix,
        sink,
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_matrix<S: SampleSource, O: PixelSink>(
    y_plane: &[S::Sample],
    u_plane: &[S::Sample],
    v_plane: &[S::Sample],
    pixels: &mut [O::Channel],
    img: &AvifImage,
    yuv_max_f: f32,
    rgb_max_f: f32,
    src: S,
    sink: O,
) {
    let w = img.width as usize;
    let h = img.height as usize;
    let cw = img.chroma_width as usize;
    let ch = img.chroma_height as usize;
    let mtrx = img.color.matrix_coefficients;

    match mtrx {
        0 => convert_image_strips(
            y_plane,
            u_plane,
            v_plane,
            pixels,
            w,
            h,
            cw,
            ch,
            img.chroma_format,
            img.effective_threads,
            src,
            SpecialMatrix::Identity,
            sink,
        ),
        8 => convert_image_strips(
            y_plane,
            u_plane,
            v_plane,
            pixels,
            w,
            h,
            cw,
            ch,
            img.chroma_format,
            img.effective_threads,
            src,
            SpecialMatrix::YCgCo,
            sink,
        ),
        10 => convert_image_strips(
            y_plane,
            u_plane,
            v_plane,
            pixels,
            w,
            h,
            cw,
            ch,
            img.chroma_format,
            img.effective_threads,
            src,
            SpecialMatrix::bt2020_cl(img.color.transfer_characteristics),
            sink,
        ),
        14 => convert_image_strips(
            y_plane,
            u_plane,
            v_plane,
            pixels,
            w,
            h,
            cw,
            ch,
            img.chroma_format,
            img.effective_threads,
            src,
            SpecialMatrix::ICtCp {
                is_hlg: img.color.transfer_characteristics == 18,
            },
            sink,
        ),
        16 | 17 => convert_image_strips(
            y_plane,
            u_plane,
            v_plane,
            pixels,
            w,
            h,
            cw,
            ch,
            img.chroma_format,
            img.effective_threads,
            src,
            SpecialMatrix::YCgCoR {
                yuv_max_f,
                rgb_max_f,
                max_rgb: rgb_max_f as i32,
            },
            sink,
        ),
        _ => {
            let (kr, kb) = matrix_kr_kb(mtrx, img.color.color_primaries);
            convert_image_strips(
                y_plane,
                u_plane,
                v_plane,
                pixels,
                w,
                h,
                cw,
                ch,
                img.chroma_format,
                img.effective_threads,
                src,
                StandardMatrix::new(kr, kb),
                sink,
            );
        }
    }
}

fn alloc_output_vec<T: Copy + Default>(
    count: usize,
    butce: Option<&Arc<Butce>>,
) -> Result<(Vec<T>, Option<ButceTutucu>)> {
    let byte_len = count
        .checked_mul(core::mem::size_of::<T>())
        .ok_or(Error::SizeOverflow {
            stage: SizeStage::TotalAllocation,
            lhs: count,
            rhs: core::mem::size_of::<T>(),
        })?;
    let guard = butce.map(|b| Butce::tut(b, byte_len)).transpose()?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: byte_len,
        })?;
    pixels.resize(count, T::default());
    Ok((pixels, guard))
}

fn build_unorm_tables_budgeted(
    bit_depth: BitDepth,
    full_range: bool,
    identity_mode: bool,
    butce: Option<&Arc<Butce>>,
) -> Result<(Vec<f32>, Vec<f32>, Option<ButceTutucu>)> {
    let count = 1usize << bit_depth.bits();
    let table_bytes = count
        .saturating_mul(core::mem::size_of::<f32>())
        .saturating_mul(2);
    let guard = butce.map(|b| Butce::tut(b, table_bytes)).transpose()?;
    let (ty, tuv) = build_unorm_tables(bit_depth, full_range, identity_mode);
    Ok((ty, tuv, guard))
}

fn build_alpha_u8_lut_budgeted(
    bit_depth: BitDepth,
    butce: Option<&Arc<Butce>>,
) -> Result<(Vec<u8>, Option<ButceTutucu>)> {
    let len = (bit_depth.max_sample() as usize) + 1;
    let guard = butce.map(|b| Butce::tut(b, len)).transpose()?;
    Ok((build_alpha_u8_lut(bit_depth), guard))
}

fn expected_output_len(img: &AvifImage, elem_per_pixel: usize) -> Result<usize> {
    let w = img.width as usize;
    let h = img.height as usize;
    let num_pixels = w.checked_mul(h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: w,
        rhs: h,
    })?;
    num_pixels
        .checked_mul(elem_per_pixel)
        .ok_or(Error::SizeOverflow {
            stage: SizeStage::TotalAllocation,
            lhs: num_pixels,
            rhs: elem_per_pixel,
        })
}

/// [`AvifImage`] düzlemlerini doğrudan çağıranın sağladığı 16-bit RGB dilimine yazar (`0` çıktı tahsisi).
pub(crate) fn write_rgb16_into(img: &AvifImage, dst: &mut [u16]) -> Result<()> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let expected = expected_output_len(img, 3)?;
    if dst.len() != expected {
        return Err(Error::InvalidArgument(
            "RGB16 çıktı dilimi uzunluğu width * height * 3 ile eşleşmiyor",
        ));
    }
    let max_val = img.bit_depth.max_sample() as usize;
    let max_f = max_val as f32;
    let sink = Rgb16Sink { max_f };
    let identity = img.color.matrix_coefficients == 0;

    match (&img.y_plane, &img.u_plane, &img.v_plane) {
        (PlaneStorage::U8(yp), PlaneStorage::U8(up), PlaneStorage::U8(vp)) => {
            let (ty, tuv) = build_unorm_tables_8bit(img.color.full_range, identity);
            let src = TablesU8 {
                table_y: &ty,
                table_uv: &tuv,
            };
            dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
        }
        (PlaneStorage::U16(yp), PlaneStorage::U16(up), PlaneStorage::U16(vp)) => {
            let (ty, tuv, _tbl_guard) = build_unorm_tables_budgeted(
                img.bit_depth,
                img.color.full_range,
                identity,
                img.butce.as_ref(),
            )?;
            let src = TablesU16 {
                table_y: &ty,
                table_uv: &tuv,
                max_val,
            };
            dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
        }
        _ => {
            return Err(Error::InvalidState(
                "AVIF düzlem depolama bit derinlikleri tutarsız",
            ));
        }
    }
    Ok(())
}

/// [`AvifImage`] düzlemlerini 16-bit RGB tamponuna dönüştürür.
pub(crate) fn image_to_rgb16(img: &AvifImage) -> Result<Rgb16Image> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let total_samples = expected_output_len(img, 3)?;
    let (mut pixels, _out_guard): (Vec<u16>, _) =
        alloc_output_vec(total_samples, img.butce.as_ref())?;
    write_rgb16_into(img, &mut pixels)?;
    Ok(Rgb16Image {
        width: img.width,
        height: img.height,
        bit_depth: img.bit_depth,
        pixels,
    })
}

/// [`AvifImage`] düzlemlerini doğrudan çağıranın sağladığı 16-bit RGBA dilimine yazar (`0` çıktı tahsisi).
pub(crate) fn write_rgba16_into(img: &AvifImage, dst: &mut [u16]) -> Result<()> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let expected = expected_output_len(img, 4)?;
    if dst.len() != expected {
        return Err(Error::InvalidArgument(
            "RGBA16 çıktı dilimi uzunluğu width * height * 4 ile eşleşmiyor",
        ));
    }
    let w = img.width as usize;
    let max_val = img.bit_depth.max_sample() as usize;
    let max_f = max_val as f32;
    let opaque_alpha = max_val as u16;
    let identity = img.color.matrix_coefficients == 0;

    match (&img.y_plane, &img.u_plane, &img.v_plane) {
        (PlaneStorage::U8(yp), PlaneStorage::U8(up), PlaneStorage::U8(vp)) => {
            let (ty, tuv) = build_unorm_tables_8bit(img.color.full_range, identity);
            let src = TablesU8 {
                table_y: &ty,
                table_uv: &tuv,
            };
            match img.alpha_plane.as_ref() {
                Some(PlaneStorage::U8(ap)) => {
                    let sink = Rgba16AlphaU8Sink {
                        max_f,
                        alpha: ap,
                        stride: w,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
                }
                None => {
                    let sink = Rgba16OpaqueSink {
                        max_f,
                        opaque_alpha,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
                }
                Some(PlaneStorage::U16(_)) => {
                    return Err(Error::InvalidState(
                        "8-bit AVIF görüntüsünde 16-bit alfa düzlemi",
                    ));
                }
            }
        }
        (PlaneStorage::U16(yp), PlaneStorage::U16(up), PlaneStorage::U16(vp)) => {
            let (ty, tuv, _tbl_guard) = build_unorm_tables_budgeted(
                img.bit_depth,
                img.color.full_range,
                identity,
                img.butce.as_ref(),
            )?;
            let src = TablesU16 {
                table_y: &ty,
                table_uv: &tuv,
                max_val,
            };
            match img.alpha_plane.as_ref() {
                Some(PlaneStorage::U16(ap)) => {
                    let sink = Rgba16AlphaU16Sink {
                        max_f,
                        opaque_alpha,
                        alpha: ap,
                        stride: w,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
                }
                None => {
                    let sink = Rgba16OpaqueSink {
                        max_f,
                        opaque_alpha,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
                }
                Some(PlaneStorage::U8(_)) => {
                    return Err(Error::InvalidState(
                        "10/12-bit AVIF görüntüsünde 8-bit alfa düzlemi",
                    ));
                }
            }
        }
        _ => {
            return Err(Error::InvalidState(
                "AVIF düzlem depolama bit derinlikleri tutarsız",
            ));
        }
    }
    Ok(())
}

/// [`AvifImage`] düzlemlerini 16-bit RGBA tamponuna dönüştürür.
pub(crate) fn image_to_rgba16(img: &AvifImage) -> Result<Rgba16Image> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let total_samples = expected_output_len(img, 4)?;
    let (mut pixels, _out_guard): (Vec<u16>, _) =
        alloc_output_vec(total_samples, img.butce.as_ref())?;
    write_rgba16_into(img, &mut pixels)?;
    Ok(Rgba16Image {
        width: img.width,
        height: img.height,
        bit_depth: img.bit_depth,
        pixels,
    })
}

/// [`AvifImage`] düzlemlerini doğrudan çağıranın sağladığı 8-bit RGB dilimine yazar (`0` çıktı tahsisi).
pub(crate) fn write_rgb8_into(img: &AvifImage, dst: &mut [u8]) -> Result<()> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let expected = expected_output_len(img, 3)?;
    if dst.len() != expected {
        return Err(Error::InvalidArgument(
            "RGB8 çıktı dilimi uzunluğu width * height * 3 ile eşleşmiyor",
        ));
    }
    let max_val = img.bit_depth.max_sample() as usize;
    let yuv_max_f = max_val as f32;
    let identity = img.color.matrix_coefficients == 0;

    match (&img.y_plane, &img.u_plane, &img.v_plane) {
        (PlaneStorage::U8(yp), PlaneStorage::U8(up), PlaneStorage::U8(vp)) => {
            let (ty, tuv) = build_unorm_tables_8bit(img.color.full_range, identity);
            let src = TablesU8 {
                table_y: &ty,
                table_uv: &tuv,
            };
            dispatch_matrix(yp, up, vp, dst, img, yuv_max_f, 255.0, src, Rgb8Sink);
        }
        (PlaneStorage::U16(yp), PlaneStorage::U16(up), PlaneStorage::U16(vp)) => {
            let (ty, tuv, _tbl_guard) = build_unorm_tables_budgeted(
                img.bit_depth,
                img.color.full_range,
                identity,
                img.butce.as_ref(),
            )?;
            let src = TablesU16 {
                table_y: &ty,
                table_uv: &tuv,
                max_val,
            };
            dispatch_matrix(yp, up, vp, dst, img, yuv_max_f, 255.0, src, Rgb8Sink);
        }
        _ => {
            return Err(Error::InvalidState(
                "AVIF düzlem depolama bit derinlikleri tutarsız",
            ));
        }
    }
    Ok(())
}

/// [`AvifImage`] düzlemlerini 8-bit RGB tamponuna dönüştürür.
pub(crate) fn image_to_rgb8(img: &AvifImage) -> Result<RgbImage> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let total_bytes = expected_output_len(img, 3)?;
    let (mut pixels, _out_guard): (Vec<u8>, _) = alloc_output_vec(total_bytes, img.butce.as_ref())?;
    write_rgb8_into(img, &mut pixels)?;
    Ok(RgbImage {
        width: img.width,
        height: img.height,
        pixels,
    })
}

/// [`AvifImage`] düzlemlerini doğrudan çağıranın sağladığı 8-bit RGBA dilimine yazar (`0` çıktı tahsisi).
pub(crate) fn write_rgba8_into(img: &AvifImage, dst: &mut [u8]) -> Result<()> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let expected = expected_output_len(img, 4)?;
    if dst.len() != expected {
        return Err(Error::InvalidArgument(
            "RGBA8 çıktı dilimi uzunluğu width * height * 4 ile eşleşmiyor",
        ));
    }
    let w = img.width as usize;
    let max_val = img.bit_depth.max_sample() as usize;
    let yuv_max_f = max_val as f32;
    let identity = img.color.matrix_coefficients == 0;

    match (&img.y_plane, &img.u_plane, &img.v_plane) {
        (PlaneStorage::U8(yp), PlaneStorage::U8(up), PlaneStorage::U8(vp)) => {
            let (ty, tuv) = build_unorm_tables_8bit(img.color.full_range, identity);
            let src = TablesU8 {
                table_y: &ty,
                table_uv: &tuv,
            };
            match img.alpha_plane.as_ref() {
                Some(PlaneStorage::U8(ap)) => {
                    let sink = Rgba8AlphaU8Sink {
                        alpha: ap,
                        stride: w,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, yuv_max_f, 255.0, src, sink);
                }
                None => {
                    dispatch_matrix(yp, up, vp, dst, img, yuv_max_f, 255.0, src, Rgba8OpaqueSink);
                }
                Some(PlaneStorage::U16(_)) => {
                    return Err(Error::InvalidState(
                        "8-bit AVIF görüntüsünde 16-bit alfa düzlemi",
                    ));
                }
            }
        }
        (PlaneStorage::U16(yp), PlaneStorage::U16(up), PlaneStorage::U16(vp)) => {
            let (ty, tuv, _tbl_guard) = build_unorm_tables_budgeted(
                img.bit_depth,
                img.color.full_range,
                identity,
                img.butce.as_ref(),
            )?;
            let src = TablesU16 {
                table_y: &ty,
                table_uv: &tuv,
                max_val,
            };
            match img.alpha_plane.as_ref() {
                Some(PlaneStorage::U16(ap)) => {
                    let (lut, _lut_guard) =
                        build_alpha_u8_lut_budgeted(img.bit_depth, img.butce.as_ref())?;
                    let sink = Rgba8AlphaU16Sink {
                        alpha: ap,
                        stride: w,
                        max_val,
                        lut: &lut,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, yuv_max_f, 255.0, src, sink);
                }
                None => {
                    dispatch_matrix(yp, up, vp, dst, img, yuv_max_f, 255.0, src, Rgba8OpaqueSink);
                }
                Some(PlaneStorage::U8(_)) => {
                    return Err(Error::InvalidState(
                        "10/12-bit AVIF görüntüsünde 8-bit alfa düzlemi",
                    ));
                }
            }
        }
        _ => {
            return Err(Error::InvalidState(
                "AVIF düzlem depolama bit derinlikleri tutarsız",
            ));
        }
    }
    Ok(())
}

/// [`AvifImage`] düzlemlerini 8-bit RGBA tamponuna dönüştürür.
pub(crate) fn image_to_rgba8(img: &AvifImage) -> Result<RgbaImage> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let total_bytes = expected_output_len(img, 4)?;
    let (mut pixels, _out_guard): (Vec<u8>, _) = alloc_output_vec(total_bytes, img.butce.as_ref())?;
    write_rgba8_into(img, &mut pixels)?;
    Ok(RgbaImage {
        width: img.width,
        height: img.height,
        pixels,
    })
}

/// [`AvifImage`] düzlemlerini `[0..=65535]` tam 16-bit aralığına ölçekleyip yerel-endian
/// bayt çiftleri halinde doğrudan çağıranın RGB16 bayt dilimine yazar (`0` ara vektör tahsisi).
pub(crate) fn write_rgb16_scaled_ne_into(img: &AvifImage, dst: &mut [u8]) -> Result<()> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let expected = expected_output_len(img, 6)?;
    if dst.len() != expected {
        return Err(Error::InvalidArgument(
            "ölçekli RGB16 bayt dilimi uzunluğu width * height * 6 ile eşleşmiyor",
        ));
    }
    let max_val = img.bit_depth.max_sample() as usize;
    let max_f = max_val as f32;
    let sink = Rgb16ScaledNeSink {
        max_f,
        bit_depth: img.bit_depth,
    };
    let identity = img.color.matrix_coefficients == 0;

    match (&img.y_plane, &img.u_plane, &img.v_plane) {
        (PlaneStorage::U8(yp), PlaneStorage::U8(up), PlaneStorage::U8(vp)) => {
            let (ty, tuv) = build_unorm_tables_8bit(img.color.full_range, identity);
            let src = TablesU8 {
                table_y: &ty,
                table_uv: &tuv,
            };
            dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
        }
        (PlaneStorage::U16(yp), PlaneStorage::U16(up), PlaneStorage::U16(vp)) => {
            let (ty, tuv, _tbl_guard) = build_unorm_tables_budgeted(
                img.bit_depth,
                img.color.full_range,
                identity,
                img.butce.as_ref(),
            )?;
            let src = TablesU16 {
                table_y: &ty,
                table_uv: &tuv,
                max_val,
            };
            dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
        }
        _ => {
            return Err(Error::InvalidState(
                "AVIF düzlem depolama bit derinlikleri tutarsız",
            ));
        }
    }
    Ok(())
}

/// [`AvifImage`] düzlemlerini `[0..=65535]` tam 16-bit aralığına ölçekleyip yerel-endian
/// bayt çiftleri halinde doğrudan çağıranın RGBA16 bayt dilimine yazar (`0` ara vektör tahsisi).
pub(crate) fn write_rgba16_scaled_ne_into(img: &AvifImage, dst: &mut [u8]) -> Result<()> {
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
    let expected = expected_output_len(img, 8)?;
    if dst.len() != expected {
        return Err(Error::InvalidArgument(
            "ölçekli RGBA16 bayt dilimi uzunluğu width * height * 8 ile eşleşmiyor",
        ));
    }
    let w = img.width as usize;
    let max_val = img.bit_depth.max_sample() as usize;
    let max_f = max_val as f32;
    let opaque_alpha = max_val as u16;
    let identity = img.color.matrix_coefficients == 0;

    match (&img.y_plane, &img.u_plane, &img.v_plane) {
        (PlaneStorage::U8(yp), PlaneStorage::U8(up), PlaneStorage::U8(vp)) => {
            let (ty, tuv) = build_unorm_tables_8bit(img.color.full_range, identity);
            let src = TablesU8 {
                table_y: &ty,
                table_uv: &tuv,
            };
            match img.alpha_plane.as_ref() {
                Some(PlaneStorage::U8(ap)) => {
                    let sink = Rgba16ScaledNeAlphaU8Sink {
                        max_f,
                        bit_depth: img.bit_depth,
                        alpha: ap,
                        stride: w,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
                }
                None => {
                    let sink = Rgba16ScaledNeOpaqueSink {
                        max_f,
                        bit_depth: img.bit_depth,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
                }
                Some(PlaneStorage::U16(_)) => {
                    return Err(Error::InvalidState(
                        "8-bit AVIF görüntüsünde 16-bit alfa düzlemi",
                    ));
                }
            }
        }
        (PlaneStorage::U16(yp), PlaneStorage::U16(up), PlaneStorage::U16(vp)) => {
            let (ty, tuv, _tbl_guard) = build_unorm_tables_budgeted(
                img.bit_depth,
                img.color.full_range,
                identity,
                img.butce.as_ref(),
            )?;
            let src = TablesU16 {
                table_y: &ty,
                table_uv: &tuv,
                max_val,
            };
            match img.alpha_plane.as_ref() {
                Some(PlaneStorage::U16(ap)) => {
                    let sink = Rgba16ScaledNeAlphaU16Sink {
                        max_f,
                        bit_depth: img.bit_depth,
                        opaque_alpha,
                        alpha: ap,
                        stride: w,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
                }
                None => {
                    let sink = Rgba16ScaledNeOpaqueSink {
                        max_f,
                        bit_depth: img.bit_depth,
                    };
                    dispatch_matrix(yp, up, vp, dst, img, max_f, max_f, src, sink);
                }
                Some(PlaneStorage::U8(_)) => {
                    return Err(Error::InvalidState(
                        "10/12-bit AVIF görüntüsünde 8-bit alfa düzlemi",
                    ));
                }
            }
        }
        _ => {
            return Err(Error::InvalidState(
                "AVIF düzlem depolama bit derinlikleri tutarsız",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        MatrixKernel, StandardMatrix, chroma_derived_kr_kb, matrix_kr_kb,
        validate_matrix_coefficients,
    };
    use crate::error::Error;

    #[test]
    fn nostd_math_matches_std_accuracy() {
        use super::math::nostd;

        for i in 1..=200 {
            let x = (i as f64) / 200.0;
            let sq = nostd::sqrt(x);
            assert!((sq * sq - x).abs() <= 1e-15, "sqrt({x})={sq}");
            #[cfg(feature = "std")]
            {
                assert!((sq - x.sqrt()).abs() <= 1e-15, "sqrt std diff at {x}");
                let l = nostd::ln(x);
                assert!(
                    (l - x.ln()).abs() <= 1e-14,
                    "ln({x}) diff: {l} vs {}",
                    x.ln()
                );
                let e = nostd::exp((x - 0.5) * 10.0);
                let es = ((x - 0.5) * 10.0).exp();
                assert!(((e - es) / es).abs() <= 1e-14, "exp diff: {e} vs {es}");
                for &y in &[
                    0.45_f64,
                    1.0 / 0.45,
                    0.1593017578125,
                    78.84375,
                    1.0 / 78.84375,
                ] {
                    let p = nostd::powf(x, y);
                    let ps = x.powf(y);
                    let denom = ps.abs().max(1e-30);
                    assert!(
                        ((p - ps) / denom).abs() <= 1e-12,
                        "powf({x}, {y}) diff: {p} vs {ps}"
                    );
                }
            }
        }
        for v in [-127.5_f32, -2.5, -0.5, -0.49, 0.0, 0.49, 0.5, 2.5, 511.5] {
            let r = nostd::round_f32(v);
            #[cfg(feature = "std")]
            assert_eq!(r, v.round(), "round_f32({v})");
            #[cfg(not(feature = "std"))]
            assert!((r - v).abs() <= 0.5);
        }
    }

    #[test]
    fn h273_chroma_derived_kr_kb_matches_cie_primaries() {
        // BT.709 (1): (0.2126, 0.0722)
        let (kr709, kb709) = chroma_derived_kr_kb(1);
        assert!((kr709 - 0.212639).abs() < 1e-4);
        assert!((kb709 - 0.072192).abs() < 1e-4);

        // BT.2020 (9): (0.2627, 0.0593)
        let (kr2020, kb2020) = chroma_derived_kr_kb(9);
        assert!((kr2020 - 0.262700).abs() < 1e-4);
        assert!((kb2020 - 0.059302).abs() < 1e-4);

        // Display P3 / SMPTE EG 432-1 (12): (0.228975, 0.079287)
        let (kr_p3, kb_p3) = chroma_derived_kr_kb(12);
        assert!((kr_p3 - 0.228975).abs() < 1e-4);
        assert!((kb_p3 - 0.079287).abs() < 1e-4);
    }

    #[test]
    fn h273_nonlinear_and_reserved_matrices_return_unsupported() {
        for supported in [0u16, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17] {
            assert_eq!(validate_matrix_coefficients(supported), Ok(()));
        }
        for unsupported in [3u16, 11, 13, 15, 18, 255] {
            assert!(matches!(
                validate_matrix_coefficients(unsupported),
                Err(Error::Unsupported(_))
            ));
        }
    }

    #[test]
    fn standard_matrix_four_pixel_matches_scalar_bits() {
        // Sınırlı aralık uçları ve kroma, işaretli sıfır, 8-bit yuvarlama eşikleri.
        let cases = core::hint::black_box([
            (
                [
                    (0, -16.0 / 219.0),
                    (128, 112.0 / 219.0),
                    (235, 1.0),
                    (255, 239.0 / 219.0),
                ],
                [-128.0 / 224.0, 127.0 / 224.0, -0.25, 0.25],
                [127.0 / 224.0, 0.175, 0.5, -0.5],
            ),
            (
                [(0, -0.0), (1, 1.0 / 1023.0), (4095, 1.0), (2048, 0.5)],
                [-0.0, -0.375, 0.0, 0.166_666_67],
                [-0.0, 0.125, 0.0, -0.123_456_79],
            ),
            (
                [
                    (0, 0.5 / 255.0),
                    (11, 11.5 / 255.0),
                    (127, 127.5 / 255.0),
                    (254, 254.5 / 255.0),
                ],
                [0.0; 4],
                [0.0; 4],
            ),
        ]);
        let bits = |(r, g, b): (f32, f32, f32)| [r.to_bits(), g.to_bits(), b.to_bits()];

        for (coefficients, primaries) in [(1, 1), (6, 6), (9, 9), (12, 12)] {
            let (kr, kb) = matrix_kr_kb(coefficients, primaries);
            let matrix = StandardMatrix::new(kr, kb);
            for (y, cb, cr) in cases {
                let actual = matrix.to_rgb_f32x4(y, cb, cr).map(bits);
                let expected: [[u32; 3]; 4] = core::array::from_fn(|i| {
                    let (sample, normalized) = y[i];
                    bits(matrix.to_rgb_f32(sample, normalized, cb[i], cr[i]))
                });
                assert_eq!(actual, expected, "matrix={coefficients}, y={y:?}");
            }
        }
    }

    #[test]
    fn strip_parallel_and_single_thread_color_conversion_parity() {
        use crate::avif::image::{AvifImage, AvifPlane, ColorInfo, PlaneStorage};
        use crate::frame::{BitDepth, ChromaFormat};

        let make_image = |w: u32,
                          h: u32,
                          bit_depth: BitDepth,
                          chroma_format: ChromaFormat,
                          mtrx: u16,
                          with_alpha: bool,
                          threads: usize| {
            let (cw, ch) = chroma_format
                .chroma_dimensions(w as usize, h as usize)
                .unwrap()
                .unwrap_or((0, 0));
            let y_len = (w as usize) * (h as usize);
            let c_len = cw * ch;
            let (y_plane, u_plane, v_plane, alpha_plane) = match bit_depth {
                BitDepth::Eight => {
                    let y: Vec<u8> = (0..y_len)
                        .map(|i| ((i.wrapping_mul(37) ^ (i >> 3)) & 0xFF) as u8)
                        .collect();
                    let u: Vec<u8> = (0..c_len)
                        .map(|i| ((i.wrapping_mul(53).wrapping_add(17)) & 0xFF) as u8)
                        .collect();
                    let v: Vec<u8> = (0..c_len)
                        .map(|i| ((i.wrapping_mul(97).wrapping_add(83)) & 0xFF) as u8)
                        .collect();
                    let a = with_alpha.then(|| {
                        PlaneStorage::U8(
                            (0..y_len)
                                .map(|i| ((i.wrapping_mul(19).wrapping_add(7)) & 0xFF) as u8)
                                .collect(),
                        )
                    });
                    (
                        PlaneStorage::U8(y),
                        PlaneStorage::U8(u),
                        PlaneStorage::U8(v),
                        a,
                    )
                }
                BitDepth::Ten | BitDepth::Twelve => {
                    let mask = bit_depth.max_sample() as usize;
                    let y: Vec<u16> = (0..y_len)
                        .map(|i| ((i.wrapping_mul(131) ^ (i >> 2)) & mask) as u16)
                        .collect();
                    let u: Vec<u16> = (0..c_len)
                        .map(|i| ((i.wrapping_mul(211).wrapping_add(31)) & mask) as u16)
                        .collect();
                    let v: Vec<u16> = (0..c_len)
                        .map(|i| ((i.wrapping_mul(307).wrapping_add(59)) & mask) as u16)
                        .collect();
                    let a = with_alpha.then(|| {
                        PlaneStorage::U16(
                            (0..y_len)
                                .map(|i| ((i.wrapping_mul(71).wrapping_add(13)) & mask) as u16)
                                .collect(),
                        )
                    });
                    (
                        PlaneStorage::U16(y),
                        PlaneStorage::U16(u),
                        PlaneStorage::U16(v),
                        a,
                    )
                }
            };
            AvifImage {
                width: w,
                height: h,
                chroma_width: cw as u32,
                chroma_height: ch as u32,
                bit_depth,
                chroma_format,
                y_plane,
                u_plane,
                v_plane,
                alpha_plane,
                premultiplied_alpha: false,
                color: ColorInfo {
                    color_primaries: 1,
                    transfer_characteristics: 13,
                    matrix_coefficients: mtrx,
                    full_range: false,
                },
                icc_profile: None,
                exif: None,
                xmp: None,
                clli: None,
                mdcv: None,
                pasp: None,
                gain_map_metadata: None,
                gain_map: None,
                effective_threads: threads,
                butce: None,
                _butce_tutucu: None,
            }
        };

        // 515 x 513 = 264_195 >= 262_144 (tek/çift kenar + şerit paralelliği eşiği üstü)
        for (bd, cf, mtrx, with_alpha) in [
            (BitDepth::Eight, ChromaFormat::C420, 1u16, true),
            (BitDepth::Eight, ChromaFormat::C420, 1u16, false),
            (BitDepth::Eight, ChromaFormat::C422, 6u16, true),
            (BitDepth::Eight, ChromaFormat::C444, 0u16, false),
            (BitDepth::Eight, ChromaFormat::Monochrome, 1u16, true),
            (BitDepth::Ten, ChromaFormat::C420, 9u16, true),
            (BitDepth::Twelve, ChromaFormat::C420, 8u16, false),
            (BitDepth::Ten, ChromaFormat::C444, 16u16, true),
        ] {
            let img_t1 = make_image(515, 513, bd, cf, mtrx, with_alpha, 1);
            let img_t4 = make_image(515, 513, bd, cf, mtrx, with_alpha, 4);
            // effective_threads farkı PartialEq sonucunu etkilememeli
            assert_eq!(img_t1, img_t4);
            match bd {
                BitDepth::Eight => {
                    assert!(matches!(img_t1.y_plane(), AvifPlane::U8(_)));
                    assert!(img_t1.y_plane_u8().is_some());
                    assert!(img_t1.y_plane_u16().is_none());
                    assert!(img_t1.y_row_u8(0).is_some());
                    assert!(img_t1.y_row_u16(0).is_none());
                }
                BitDepth::Ten | BitDepth::Twelve => {
                    assert!(matches!(img_t1.y_plane(), AvifPlane::U16(_)));
                    assert!(img_t1.y_plane_u16().is_some());
                    assert!(img_t1.y_plane_u8().is_none());
                    assert!(img_t1.y_row_u16(0).is_some());
                    assert!(img_t1.y_row_u8(0).is_none());
                }
            }
            assert_eq!(img_t1.to_rgb8().unwrap(), img_t4.to_rgb8().unwrap());
            assert_eq!(img_t1.to_rgba8().unwrap(), img_t4.to_rgba8().unwrap());
            assert_eq!(img_t1.to_rgb16().unwrap(), img_t4.to_rgb16().unwrap());
            assert_eq!(img_t1.to_rgba16().unwrap(), img_t4.to_rgba16().unwrap());
        }
    }

    #[test]
    fn h273_ictcp_pq_and_hlg_reference_vectors() {
        use super::{MatrixKernel, SpecialMatrix, hlg_inv_oetf, hlg_oetf, pq_eotf, pq_oetf};

        // İleri BT.2100 RGB' -> ICtCp (PQ ve HLG) -> SpecialMatrix::ICtCp geri dönüşümü.
        let encode_ictcp = |r_p: f64, g_p: f64, b_p: f64, is_hlg: bool| -> (f32, f32, f32) {
            let (r, g, b) = if is_hlg {
                (hlg_inv_oetf(r_p), hlg_inv_oetf(g_p), hlg_inv_oetf(b_p))
            } else {
                (pq_eotf(r_p), pq_eotf(g_p), pq_eotf(b_p))
            };
            let l = (1688.0 * r + 2146.0 * g + 262.0 * b) / 4096.0;
            let m = (683.0 * r + 2951.0 * g + 462.0 * b) / 4096.0;
            let s = (99.0 * r + 309.0 * g + 3688.0 * b) / 4096.0;
            let (lp, mp, sp) = if is_hlg {
                (hlg_oetf(l), hlg_oetf(m), hlg_oetf(s))
            } else {
                (pq_oetf(l), pq_oetf(m), pq_oetf(s))
            };
            let i = 0.5 * lp + 0.5 * mp;
            let (ct, cp) = if is_hlg {
                (
                    (3625.0 * lp - 7465.0 * mp + 3840.0 * sp) / 4096.0,
                    (9500.0 * lp - 9212.0 * mp - 288.0 * sp) / 4096.0,
                )
            } else {
                (
                    (6610.0 * lp - 13613.0 * mp + 7003.0 * sp) / 4096.0,
                    (17933.0 * lp - 17390.0 * mp - 543.0 * sp) / 4096.0,
                )
            };
            (i as f32, ct as f32, cp as f32)
        };

        let samples = [
            (0.0, 0.0, 0.0),   // Siyah
            (1.0, 1.0, 1.0),   // Beyaz
            (0.5, 0.5, 0.5),   // Orta gri
            (0.8, 0.2, 0.1),   // Doygun kırmızımsı
            (0.1, 0.85, 0.2),  // Doygun yeşilimsi
            (0.15, 0.25, 0.9), // Doygun mavimsi
        ];
        for is_hlg in [false, true] {
            let kernel = SpecialMatrix::ICtCp { is_hlg };
            for (er, eg, eb) in samples {
                let (i, ct, cp) = encode_ictcp(er, eg, eb, is_hlg);
                let (rr, rg, rb) = kernel.to_rgb_f32(0, i, ct, cp);
                assert!(
                    (f64::from(rr) - er).abs() < 1e-4
                        && (f64::from(rg) - eg).abs() < 1e-4
                        && (f64::from(rb) - eb).abs() < 1e-4,
                    "ICtCp(is_hlg={is_hlg}) sapması: girdi=({er},{eg},{eb}) çıktı=({rr},{rg},{rb})"
                );
            }
        }
    }

    #[test]
    fn h273_bt2020_cl_reference_vectors() {
        use super::{MatrixKernel, SpecialMatrix, bt2020_inv_oetf, bt2020_oetf, pq_eotf, pq_oetf};

        let encode_bt2020_cl = |r_p: f64, g_p: f64, b_p: f64, trc: u16| -> (f32, f32, f32) {
            const KR: f64 = 0.2627;
            const KB: f64 = 0.0593;
            const KG: f64 = 0.6780;
            let is_pq = trc == 16;
            let (r, g, b) = if is_pq {
                (pq_eotf(r_p), pq_eotf(g_p), pq_eotf(b_p))
            } else {
                (
                    bt2020_inv_oetf(r_p),
                    bt2020_inv_oetf(g_p),
                    bt2020_inv_oetf(b_p),
                )
            };
            let oetf = if is_pq { pq_oetf } else { bt2020_oetf };
            let y_lin = KR * r + KG * g + KB * b;
            let yc_p = oetf(y_lin);
            let nb = oetf(1.0 - KB);
            let pb = 1.0 - oetf(KB);
            let nr = oetf(1.0 - KR);
            let pr = 1.0 - oetf(KR);
            let cbc = if b_p <= yc_p {
                (b_p - yc_p) / (2.0 * nb)
            } else {
                (b_p - yc_p) / (2.0 * pb)
            };
            let crc = if r_p <= yc_p {
                (r_p - yc_p) / (2.0 * nr)
            } else {
                (r_p - yc_p) / (2.0 * pr)
            };
            (yc_p as f32, cbc as f32, crc as f32)
        };

        let samples = [
            (0.0, 0.0, 0.0), // Siyah
            (1.0, 1.0, 1.0), // Beyaz
            (0.5, 0.5, 0.5), // Nötr gri
            (1.0, 0.0, 0.0), // Saf kırmızı
            (0.0, 1.0, 0.0), // Saf yeşil
            (0.0, 0.0, 1.0), // Saf mavi
            (1.0, 1.0, 0.0), // Sarı
            (0.0, 1.0, 1.0), // Camgöbeği
        ];
        for trc in [14u16, 16u16] {
            let kernel = SpecialMatrix::bt2020_cl(trc);
            // PQ (trc=16) OETF'nin sıfır civarındaki dik eğimi nedeniyle `f32` giriş yuvarlaması
            // doğrusal ışıkta <1e-5 sapmayla kod uzayında ~3.5e-3 üretebilir.
            let tol = if trc == 16 { 5e-3 } else { 1e-4 };
            for (er, eg, eb) in samples {
                let (yc, cbc, crc) = encode_bt2020_cl(er, eg, eb, trc);
                let (rr, rg, rb) = kernel.to_rgb_f32(0, yc, cbc, crc);
                assert!(
                    (f64::from(rr) - er).abs() < tol
                        && (f64::from(rg) - eg).abs() < tol
                        && (f64::from(rb) - eb).abs() < tol,
                    "BT.2020 CL (trc={trc}) sapması: girdi=({er},{eg},{eb}) çıktı=({rr},{rg},{rb})"
                );
                if trc == 16 {
                    assert!(
                        (pq_eotf(f64::from(rg)) - pq_eotf(eg)).abs() < 1e-5,
                        "BT.2020 CL (PQ) doğrusal G sapması"
                    );
                }
            }
        }
    }
}
