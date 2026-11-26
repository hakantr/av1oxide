//! İşlev profili (F3d / P27, PRF-04) — yalnız varsayılan kapalı `profil` özelliğiyle derlenir.
//!
//! Çözme hattındaki her DSP yuva çağrısı [`profil_olc!`](crate::dsp::profil_olc)
//! ile sarılıdır. Özellik **kapalıyken** makro çağrı ifadesini olduğu gibi
//! bırakır (tek bir belirteç bile eklenmez): üretilen kod değişmez; bu
//! modül derlenmez. Özellik **açıkken** her çağrının süresi
//! [`std::time::Instant`] ile ölçülür ve `(işlev, bit derinliği)` başına çağrı sayısı +
//! toplam nanosaniye olarak biriktirilir.
//!
//! - **Bit derinliği:** kare başında [`profil_bd!`](crate::dsp::profil_bd) ile
//!   kaydedilir (çözücü tek iş parçacığıdır; akışta bit derinliği sabittir).
//! - **Boş çağrı maliyeti:** `kalibre` işlevi aynı makroyu boş ifadeyle çalıştırır;
//!   ölçüm penceresine düşen pay (`ic_ns`) ve çağrı başına toplam ek yük
//!   (`toplam_ns`) ayrı ayrı döner. Net süre = ham − çağrı × `ic_ns`; kalan
//!   (yuva dışı) süre = (duvar − toplam çağrı × `toplam_ns`) − Σ net.
//! - **MSAC** derleme zamanı seçimlidir (yuva değildir; dav1d x86-64'te de
//!   öyledir). MSAC'ın ana tüketicisi olan katsayı sözdizimi
//!   (`decode::coef::decode_coefs`: sembol çözme + bağlam türetme) yuva
//!   olmayan ayrı bir satırdır (`sozdizimi.decode_coefs`); kip/hareket
//!   sözdizimi, refmvs aday arama ve kopyalar "kalan" satırındadır.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use crate::frame::BitDepth;

macro_rules! islevler {
    ($($ad:ident => $aile:literal, $alan:literal;)*) => {
        /// Profili tutulan yuva işlevi (aile × alan; alt tür indeksi birleşik).
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        pub enum Islev {
            $(#[doc = concat!("`", $aile, ".", $alan, "`")] $ad,)*
            /// Boş çağrı kalibrasyonu (tabloya girmez).
            Kalibrasyon,
        }

        /// `(işlev, aile, alan)` — [`crate::dsp::slot_keys`] adlarıyla
        /// (`sozdizimi` ailesi yuva değildir).
        pub const ISLEVLER: &[(Islev, &str, &str)] = &[$((Islev::$ad, $aile, $alan),)*];
    };
}

islevler! {
    ItxfmAdd => "itx", "itxfm_add";
    McMc => "mc", "mc";
    McMcScaled => "mc", "mc_scaled";
    McMct => "mc", "mct";
    McMctScaled => "mc", "mct_scaled";
    McAvg => "mc", "avg";
    McWAvg => "mc", "w_avg";
    McMask => "mc", "mask";
    McWMask => "mc", "w_mask";
    McBlend => "mc", "blend";
    McBlendV => "mc", "blend_v";
    McBlendH => "mc", "blend_h";
    McWarp8x8 => "mc", "warp8x8";
    McWarp8x8t => "mc", "warp8x8t";
    McEmuEdge => "mc", "emu_edge";
    Resize => "resize", "resize";
    IntraPred => "ipred", "intra_pred";
    CflAc => "ipred", "cfl_ac";
    CflPred => "ipred", "cfl_pred";
    PalPred => "ipred", "pal_pred";
    LoopFilterSb => "loopfilter", "loop_filter_sb";
    CdefDir => "cdef", "dir";
    CdefFb => "cdef", "fb";
    LrWiener => "looprestoration", "wiener";
    LrSgr => "looprestoration", "sgr";
    FgGrainY => "filmgrain", "generate_grain_y";
    FgGrainUv => "filmgrain", "generate_grain_uv";
    FgY => "filmgrain", "fgy_32x32xn";
    FgUv => "filmgrain", "fguv_32x32xn";
    PalIdxFinish => "pal", "pal_idx_finish";
    RefmvsLoadTmvs => "refmvs", "load_tmvs";
    RefmvsSaveTmvs => "refmvs", "save_tmvs";
    RefmvsSplatMv => "refmvs", "splat_mv";
    KatsayiCozme => "sozdizimi", "decode_coefs";
}

const N: usize = Islev::Kalibrasyon as usize + 1;

#[allow(clippy::declare_interior_mutable_const)]
const SIFIR: AtomicU64 = AtomicU64::new(0);
#[allow(clippy::declare_interior_mutable_const)]
const SIFIR3: [AtomicU64; 3] = [SIFIR; 3];
static SAYI: [[AtomicU64; 3]; N] = [SIFIR3; N];
static SURE: [[AtomicU64; 3]; N] = [SIFIR3; N];
static BD: AtomicU8 = AtomicU8::new(0);

/// Sonraki çağrıların bit derinliği (kare başında).
#[inline]
pub fn bit_derinligi(bd: BitDepth) {
    BD.store(
        match bd {
            BitDepth::Eight => 0,
            BitDepth::Ten => 1,
            BitDepth::Twelve => 2,
        },
        Relaxed,
    );
}

/// `t0`'dan bu yana geçen süreyi `islev` hanesine ekler.
#[inline]
pub fn kaydet(islev: Islev, t0: Instant) {
    let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let b = usize::from(BD.load(Relaxed));
    SAYI[islev as usize][b].fetch_add(1, Relaxed);
    SURE[islev as usize][b].fetch_add(ns, Relaxed);
}

/// Bütün sayaçları sıfırlar.
pub fn sifirla() {
    for i in 0..N {
        for b in 0..3 {
            SAYI[i][b].store(0, Relaxed);
            SURE[i][b].store(0, Relaxed);
        }
    }
}

/// Tek profil satırı: `(aile, alan, bit derinliği bits, çağrı, ham ns)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Satir {
    /// Yuva ailesi (`mc`, `itx`, …).
    pub aile: &'static str,
    /// Yuva alanı (`mc`, `avg`, `itxfm_add`, …).
    pub alan: &'static str,
    /// Bit derinliği (8, 10, 12).
    pub bd: u8,
    /// Çağrı sayısı.
    pub cagri: u64,
    /// Ölçüm penceresindeki ham toplam süre (ns).
    pub ham_ns: u64,
}

/// Sıfır olmayan satırlar ([`ISLEVLER`] sırasıyla).
#[must_use]
pub fn oku() -> Vec<Satir> {
    let mut out = Vec::new();
    for &(islev, aile, alan) in ISLEVLER {
        for (b, bits) in [8u8, 10, 12].into_iter().enumerate() {
            let cagri = SAYI[islev as usize][b].load(Relaxed);
            if cagri > 0 {
                out.push(Satir {
                    aile,
                    alan,
                    bd: bits,
                    cagri,
                    ham_ns: SURE[islev as usize][b].load(Relaxed),
                });
            }
        }
    }
    out
}

/// Boş çağrı maliyeti: `(ic_ns, toplam_ns)` çağrı başına.
///
/// `ic_ns` ölçüm penceresinin (iki `Instant` arası) boş ifadedeki ortalaması,
/// `toplam_ns` bir sarılı boş çağrının duvar saati maliyetidir (iki `Instant`
/// + kayıt). Kalibrasyon sayaçları tabloya girmez ve sonunda sıfırlanır.
#[must_use]
pub fn kalibre(n: u32) -> (f64, f64) {
    let k = Islev::Kalibrasyon as usize;
    for b in 0..3 {
        SAYI[k][b].store(0, Relaxed);
        SURE[k][b].store(0, Relaxed);
    }
    let duvar = Instant::now();
    for _ in 0..n {
        crate::dsp::profil_olc!(Kalibrasyon, core::hint::black_box(()));
    }
    let duvar_ns = duvar.elapsed().as_nanos() as f64;
    let b = usize::from(BD.load(Relaxed));
    let ic = SURE[k][b].load(Relaxed) as f64 / f64::from(n.max(1));
    SAYI[k][b].store(0, Relaxed);
    SURE[k][b].store(0, Relaxed);
    (ic, duvar_ns / f64::from(n.max(1)))
}

#[cfg(test)]
mod tests {
    use super::{ISLEVLER, Islev, kalibre, kaydet, oku};
    use std::collections::BTreeSet;

    /// Kapsam: `slot_keys()`'teki her çalışma zamanı yuvasının (aile, alan)
    /// çifti bir profil işlevine bağlıdır (MSAC derleme zamanıdır).
    #[test]
    fn her_yuva_alani_profilde() {
        let profil: BTreeSet<(&str, &str)> = ISLEVLER.iter().map(|&(_, a, f)| (a, f)).collect();
        let mut eksik = Vec::new();
        for key in crate::dsp::slot_keys() {
            if key.family == "msac" {
                continue;
            }
            let alan = key.field.split('[').next().unwrap_or_default();
            if !profil.contains(&(key.family, alan)) {
                eksik.push(format!("{}.{alan}", key.family));
            }
        }
        eksik.dedup();
        assert!(
            eksik.is_empty(),
            "profilde olmayan yuva alanları: {eksik:?}"
        );
    }

    /// Kayıt ve kalibrasyon. Sayaçlar süreç geneli olduğundan (paralel testler
    /// çözme yapabilir) yalnız artış ve kalibrasyonun tabloya girmediği sınanır.
    #[test]
    fn kayit_ve_kalibrasyon() {
        let once = |s: &[super::Satir]| {
            s.iter()
                .filter(|r| (r.aile, r.alan) == ("mc", "avg"))
                .map(|r| r.cagri)
                .sum::<u64>()
        };
        let a = once(&oku());
        kaydet(Islev::McAvg, std::time::Instant::now());
        assert!(once(&oku()) > a);
        let (ic, toplam) = kalibre(1000);
        assert!(ic >= 0.0 && toplam >= ic, "ic {ic} toplam {toplam}");
        assert!(
            oku()
                .iter()
                .all(|r| ISLEVLER.iter().any(|&(_, a, f)| (a, f) == (r.aile, r.alan))),
            "kalibrasyon tabloya girmez"
        );
    }
}
