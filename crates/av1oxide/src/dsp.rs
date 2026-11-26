//! DSP dispatch tablosu — F3c yuva yapısı (P26).
//!
//! ## Neden tablo?
//!
//! CEK-09 ve DSP-02: CPU dispatch **işlev düzeyindedir**, seviye düzeyinde
//! değildir. "AVX2 var" demek, AVX2'nin her işlemde kullanılacağı anlamına
//! gelmez; seçim `OS state + mimari + derlenmiş kod + bit derinliği + işlem
//! türü + mikro-mimari kısıtları` birlikte değerlendirilerek yapılır.
//!
//! ## Yuva anahtarı (ADR-0003 §2.1)
//!
//! Her yuva `(aile, işlev, bit derinliği, alt-tür)` ile adlandırılır ve dav1d
//! bağlam yapısının bir alanına karşılık gelir (`Dav1d*DSPContext`; ör.
//! `itxfm_add[TX_4X4][DCT_DCT]`, `mc[FILTER_2D_8TAP_REGULAR]`, `fb[0]`,
//! `loop_filter_sb[0][1]`). **Her yuvada her zaman bir giriş vardır**;
//! varsayılan giriş skaler gövdedir (DSP-01/6). `Option` yoktur.
//!
//! - Bit derinliğine bağlı aileler (itx, mc, resize, ipred, loopfilter, cdef,
//!   looprestoration, filmgrain) **her bit derinliği için ayrı tablo** taşır
//!   ([`BitDepthDsp`]); tablodaki her giriş o bit derinliğinin `const`
//!   monomorfudur (DOG-05). dav1d'nin 16bpc kaydı 10 ve 12 bit tablolarının
//!   ikisine karşılık gelir.
//! - Bit derinliğinden bağımsız aileler (pal, refmvs) dav1d'deki gibi tek
//!   tablodur (`dav1d_pal_dsp_init`, `dav1d_refmvs_dsp_init`).
//! - MSAC çalışma zamanı işaretçisi taşımaz: dav1d x86-64'te seçim derleme
//!   zamanındadır (`src/x86/msac.h:47-54`); yalnız `symbol_adapt16` için
//!   bağlam başına işaretçi vardır (`msac.h:57-67`). Bu port MSAC'ı
//!   [`MSAC_FUNCTIONS`] kaydıyla raporlar (seçim: derleme zamanı, skaler).
//!
//! ## Bit derinliği seçimi
//!
//! Çözme hattı bit derinliği tablosunu **kare başında bir kez** alır
//! ([`DspDispatch::for_bit_depth`]); blok, satır ve piksel döngülerinde bit
//! derinliği dalı ve çağrı başına `match bit_depth` yoktur. Çekirdek imzaları
//! bit derinliği parametresini taşımaya devam eder (ADR-0003 §2.2); monomorf
//! girişler bu parametreyi yalnız `debug_assert` ile yuvayla karşılaştırır.
//!
//! ## Değişmezlik (DSP-01/4)
//!
//! Tablo yalnız [`DspDispatch::select`] ile üretilir; bütün alanlar gizlidir ve
//! dışarıya yalnız paylaşılan referans (`&`) verilir. Kurulumdan sonra yazma
//! yolu (genel alan ya da `&mut self` yöntemi) yoktur:
//!
//! ```compile_fail,E0616
//! use av1oxide::cpu::CpuFlags;
//! use av1oxide::dsp::DspDispatch;
//! let mut dsp = DspDispatch::select(CpuFlags::detect());
//! dsp.refmvs = *DspDispatch::select(CpuFlags::NONE).refmvs(); // gizli alan
//! ```
//!
//! ```compile_fail,E0616
//! use av1oxide::cpu::CpuFlags;
//! use av1oxide::dsp::DspDispatch;
//! use av1oxide::frame::BitDepth;
//! let dsp = DspDispatch::select(CpuFlags::detect());
//! let mut t = *dsp.for_bit_depth(BitDepth::Eight);
//! t.mc = *t.mc(); // bit derinliği tablosunun alanları da gizli
//! ```
//!
//! ```compile_fail,E0594
//! use av1oxide::cpu::CpuFlags;
//! use av1oxide::dsp::DspDispatch;
//! use av1oxide::frame::BitDepth;
//! let dsp = DspDispatch::select(CpuFlags::detect());
//! let f = dsp.for_bit_depth(BitDepth::Ten).mc().mc[1];
//! dsp.for_bit_depth(BitDepth::Eight).mc().mc[0] = f; // `&` üzerinden yazma
//! ```
//!
//! ## `unsafe` politikası (GUV-02/GUV-04)
//!
//! Burada `forbid` değil **`deny`** vardır: `forbid` alt modüllerce
//! gevşetilemediği için, tek ve denetlenmiş `unsafe` sınırının
//! (`x86`, audit defterinde kayıtlı) altında `allow` ile açılabilmesi
//! gerekir. `dsp` ağacındaki **diğer** dosyalar `#![forbid(unsafe_code)]`
//! taşır; denetim testi (`audit::her_kaynak_dosyasi_politika_bildirir`)
//! kayıtsız bir dosyada `unsafe` kullanımını yakalar.

#![deny(unsafe_code)]

/// İşlev profili kancası (F3d / P27): `profil_olc!(İşlev, çağrı)`.
///
/// `profil` özelliği **kapalıyken** (varsayılan) çağrı ifadesinin kendisine
/// açılır; ilk argüman atılır, hiçbir kod eklenmez. Açıkken çağrının süresini
/// [`profil::kaydet`] ile biriktirir. `@` biçimi işlevi ifade olarak alır.
#[cfg(not(feature = "profil"))]
macro_rules! profil_olc {
    ($islev:ident, $e:expr) => {
        $e
    };
    (@ $islev:expr, $e:expr) => {
        $e
    };
}

/// İşlev profili kancası (F3d / P27) — `profil` açık: süre ölçümü.
#[cfg(feature = "profil")]
macro_rules! profil_olc {
    ($islev:ident, $e:expr) => {
        $crate::dsp::profil_olc!(@ $crate::dsp::profil::Islev::$islev, $e)
    };
    (@ $islev:expr, $e:expr) => {{
        let t0 = ::std::time::Instant::now();
        let r = $e;
        $crate::dsp::profil::kaydet($islev, t0);
        r
    }};
}
pub(crate) use profil_olc;

/// Profil bit derinliği kancası: kapalıyken boş.
#[cfg(not(feature = "profil"))]
macro_rules! profil_bd {
    ($bd:expr) => {};
}

/// Profil bit derinliği kancası — `profil` açık.
#[cfg(feature = "profil")]
macro_rules! profil_bd {
    ($bd:expr) => {
        $crate::dsp::profil::bit_derinligi($bd)
    };
}
pub(crate) use profil_bd;

pub mod cdef;
pub mod fg;
pub mod fg_tables;
pub mod ipred;
pub mod ipred_tables;
pub mod itx;
pub mod itx_tables;
pub mod lf;
pub mod lr;
pub mod lr_tables;
pub mod mc;
pub mod mc_tables;
pub mod pal;
/// İşlev profili (yalnız `profil` özelliğiyle; F3d).
#[cfg(feature = "profil")]
pub mod profil;
pub mod refmvs;
pub mod resize;
pub mod resize_tables;
pub mod scalar;

/// x86_64 ISA backend'i — üretim ağacındaki denetlenmiş `unsafe` bölgesi (GUV-02).
#[cfg(target_arch = "x86_64")]
pub mod x86;

/// `aarch64` NEON (`F5` / `P51`) ISA backend'i (`dsp/arm/mem.rs` dışında `%100 #![forbid(unsafe_code)]`).
pub mod arm;
pub use arm::ARM_ISA_CANDIDATES;

/// WebAssembly `simd128` ISA backend'i (`%100 #![forbid(unsafe_code)]`, `F6` / `P50` — `WP-10`).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::cpu::{
    Avx2Token, Avx512IclToken, CpuFlags, DotprodToken, NeonToken, Sse2Token, Sse41Token, Ssse3Token,
};
use crate::dsp::cdef::CdefDsp;
use crate::dsp::fg::FgDsp;
use crate::dsp::ipred::IntraPredDsp;
use crate::dsp::itx::{ItxDsp, TX_SIZES, TX_TYPES, TxType, is_supported};
use crate::dsp::lf::LoopFilterDsp;
use crate::dsp::lr::LrDsp;
use crate::dsp::mc::McDsp;
use crate::dsp::pal::PalDsp;
use crate::dsp::refmvs::RefmvsDsp;
use crate::dsp::resize::ResizeFn;
use crate::frame::BitDepth;

/// Tek bir bit derinliğinin DSP tablosu — bit derinliğine bağlı bütün aileler.
///
/// Her giriş o bit derinliğinin monomorfudur. Alanlar gizlidir; okuma
/// yöntemleri paylaşılan referans döndürür.
#[derive(Clone, Copy, Debug)]
pub struct BitDepthDsp {
    bit_depth: BitDepth,
    /// Tablonun kurulduğu skaler monomorfun `BPC`'si (P30: yuva kimliği işlev
    /// adresiyle değil, kurulum kaydıyla denetlenir).
    bpc: u32,
    itx: ItxDsp,
    mc: McDsp,
    resize: ResizeFn,
    ipred: IntraPredDsp,
    lf: LoopFilterDsp,
    cdef: CdefDsp,
    lr: LrDsp,
    fg: FgDsp,
}

impl BitDepthDsp {
    /// `BPC` bit derinliğinin skaler tablosu. `BPC` ile `bit_depth`'in uyumu
    /// derleme zamanında denetlenir ([`DspDispatch::SKALER`] bir `const`'tır).
    const fn scalar<const BPC: u32>(bit_depth: BitDepth) -> Self {
        assert!(
            matches!(
                (BPC, bit_depth),
                (8, BitDepth::Eight) | (10, BitDepth::Ten) | (12, BitDepth::Twelve)
            ),
            "BitDepthDsp::scalar: BPC ile bit derinliği uyuşmuyor"
        );
        Self {
            bit_depth,
            bpc: BPC,
            itx: ItxDsp::scalar::<BPC>(),
            mc: McDsp::scalar::<BPC>(),
            resize: resize::scalar::<BPC>(),
            ipred: IntraPredDsp::scalar::<BPC>(),
            lf: LoopFilterDsp::scalar::<BPC>(),
            cdef: CdefDsp::scalar::<BPC>(),
            lr: LrDsp::scalar::<BPC>(),
            fg: FgDsp::scalar::<BPC>(),
        }
    }

    /// Tablonun bit derinliği.
    #[must_use]
    pub const fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }

    /// Bütün ailelerin skaler girişlerinin monomorf `BPC`'si (kurulum kaydı;
    /// her aile tablosu aynı `BPC` ile `X::scalar::<BPC>()` kurulur).
    #[must_use]
    pub const fn monomorph_bpc(&self) -> u32 {
        self.bpc
    }

    /// itx ailesi (`Dav1dInvTxfmDSPContext`).
    #[must_use]
    pub const fn itx(&self) -> &ItxDsp {
        &self.itx
    }

    /// MC ailesi (`Dav1dMCDSPContext`, `resize` hariç).
    #[must_use]
    pub const fn mc(&self) -> &McDsp {
        &self.mc
    }

    /// Superres büyütme (`Dav1dMCDSPContext.resize`).
    #[must_use]
    pub const fn resize(&self) -> ResizeFn {
        self.resize
    }

    /// Intra tahmin ailesi (`Dav1dIntraPredDSPContext`).
    #[must_use]
    pub const fn ipred(&self) -> &IntraPredDsp {
        &self.ipred
    }

    /// Loop filter ailesi (`Dav1dLoopFilterDSPContext`).
    #[must_use]
    pub const fn lf(&self) -> &LoopFilterDsp {
        &self.lf
    }

    /// CDEF ailesi (`Dav1dCdefDSPContext`).
    #[must_use]
    pub const fn cdef(&self) -> &CdefDsp {
        &self.cdef
    }

    /// Loop restoration ailesi (`Dav1dLoopRestorationDSPContext`).
    #[must_use]
    pub const fn lr(&self) -> &LrDsp {
        &self.lr
    }

    /// Film grain ailesi (`Dav1dFilmGrainDSPContext`).
    #[must_use]
    pub const fn fg(&self) -> &FgDsp {
        &self.fg
    }
}

/// İşlev başına seçilmiş DSP çekirdekleri (DSP-01/3).
///
/// Bütün alanlar gizlidir; tablo yalnız [`DspDispatch::select`] ile üretilir
/// (modül belgesi, "Değişmezlik").
#[derive(Clone, Copy, Debug)]
pub struct DspDispatch {
    /// `[8 bit, 10 bit, 12 bit]`.
    tables: [BitDepthDsp; 3],
    /// Palet ailesi (bit derinliğinden bağımsız).
    pal: PalDsp,
    /// Refmvs ailesi (bit derinliğinden bağımsız).
    refmvs: RefmvsDsp,
    /// Yuvayı tutan ISA adaylarının bit kümesi ([`ISA_CANDIDATES`] indeksi;
    /// yuva başına en çok bir bit — üst düzey aday alt düzeyinkini siler).
    /// P33: aday sayısı 64'ü aştığı için kelime dizisi ([`KURULU_KELIME`]).
    installed: [u64; KURULU_KELIME],
    /// Kurulumda üretilen SSE2 yetki belirteci (P43).
    sse2: Option<Sse2Token>,
    /// Kurulumda üretilen SSSE3 yetki belirteci (P30).
    ssse3: Option<Ssse3Token>,
    /// Kurulumda üretilen SSE4.1 yetki belirteci (P30).
    sse41: Option<Sse41Token>,
    /// Kurulumda üretilen AVX2 yetki belirteci (F3b).
    avx2: Option<Avx2Token>,
    /// Kurulumda üretilen AVX512ICL yetki belirteci.
    avx512icl: Option<Avx512IclToken>,
    /// P36: bayraklarda [`CpuFlags::SLOW_GATHER`] var mı (AVX512ICL warp ve
    /// `resize` adayları kurulmaz; dav1d `mc_init_x86`).
    yavas_toplama: bool,
    /// P50 (`WP-10`): `wasm32` (`target_feature = "simd128"`) derlemesinde
    /// [`CpuFlags::WASM_SIMD128`] etkin mi.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    wasm_simd128: bool,
    /// P51 (`F5a`): `aarch64` derlemesinde kurulumda üretilen NEON yetki belirteci.
    #[cfg(target_arch = "aarch64")]
    neon: Option<NeonToken>,
    /// P51 (`F5a`): `aarch64` derlemesinde kurulumda üretilen DOTPROD yetki belirteci.
    #[cfg(target_arch = "aarch64")]
    dotprod: Option<DotprodToken>,
}

/// [`ISA_CANDIDATES`] kurulum bit kümesine (`KURULU_KELIME` kelime) sığar.
const _: () = assert!(X86_ISA_CANDIDATES.len() <= 64 * KURULU_KELIME);
const _: () = assert!(ARM_ISA_CANDIDATES.len() <= 64 * KURULU_KELIME);

/// ISA uygulama adının düzey sırası (dav1d `init_x86` sırası: SSE2 < SSSE3 <
/// SSE4.1 < AVX2 < AVX512ICL; `aarch64`: NEON = 1 < DOTPROD = 2; `wasm32`:
/// `simd128` = 1); kurulum alt düzeyden üste yürür, üst düzey aynı yuvanın
/// alt düzey girişinin üzerine yazar. Bilinmeyen ad 0.
#[must_use]
pub const fn isa_rank(implementation: &str) -> u8 {
    match implementation.as_bytes() {
        b"sse2" | b"neon" => 1,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        b"simd128" => 1,
        b"ssse3" | b"neon_dotprod" => 2,
        b"sse4" => 3,
        b"avx2" => 4,
        b"avx512icl" => 5,
        _ => 0,
    }
}

/// Kurulum sırası (P35): varsayılan adaylar `(düzey, indeks, yuva)` olarak,
/// düzey azalan ve aynı düzeyde indeks azalan sırada; `yuva` adayın yuvasının
/// temsilcisidir (aynı aile, alan ve bit derinliğindeki ilk adayın indeksi).
/// Süreçte bir kez hesaplanır ([`DspDispatch::select`] çözücü başına çağrılır).
///
/// `unnecessary_sort_by`: sıralama `(düzey, indeks)` azalan karşılaştırmayla
/// açıkça yazılı kalır; `sort_unstable_by_key(Reverse)` önerisi ikilinin
/// yerleşimini değiştiriyordu (P35 son hız bütçesi kapısı ve çözme ölçümü bu
/// makine koduyla koştu; `.text` karşılaştırması tur belgesinde).
#[allow(clippy::unnecessary_sort_by)]
fn kurulum_sirasi() -> &'static [(u8, u16, u16)] {
    static SIRA: crate::sync::OnceLock<alloc::vec::Vec<(u8, u16, u16)>> =
        crate::sync::OnceLock::new();
    SIRA.get_or_init(|| {
        let mut temsilci = indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        let mut sira = alloc::vec::Vec::new();
        for (i, c) in ISA_CANDIDATES.iter().enumerate() {
            let i = u16::try_from(i).expect("aday sayısı u16'ya sığar");
            let yuva = *temsilci
                .entry((c.family, c.field, c.bit_depth))
                .or_insert(i);
            if c.default {
                sira.push((isa_rank(c.implementation), i, yuva));
            }
        }
        sira.sort_unstable_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));
        sira
    })
}

/// Bir ISA adayının yuva girişi (kurulum ve çekirdek ölçümü,
/// [`DspDispatch::candidate_entry`]).
#[derive(Clone, Copy, Debug)]
pub enum IsaEntry {
    /// `cdef/dir`.
    CdefDir(cdef::CdefDirFn),
    /// `cdef/fb[i]`.
    CdefFb(usize, cdef::CdefFilterFn),
    /// `mc/mc[i]`.
    Mc(usize, mc::McPutFn),
    /// `mc/mct[i]` (P32).
    Mct(usize, mc::McPrepFn),
    /// `mc/avg` (P33).
    Avg(mc::AvgFn),
    /// `mc/w_avg` (P33).
    WAvg(mc::WAvgFn),
    /// `mc/mask` (P33).
    Mask(mc::MaskFn),
    /// `mc/w_mask[i]` (P33).
    WMask(usize, mc::WMaskFn),
    /// `mc/blend` (P33).
    Blend(mc::BlendFn),
    /// `mc/blend_v` (P33).
    BlendV(mc::BlendDirFn),
    /// `mc/blend_h` (P33).
    BlendH(mc::BlendDirFn),
    /// `mc/warp8x8` (P33).
    Warp(mc::Warp8x8Fn),
    /// `mc/warp8x8t` (P33).
    WarpT(mc::Warp8x8tFn),
    /// `mc/emu_edge` (P33).
    EmuEdge(mc::EmuEdgeFn),
    /// `mc/mc_scaled[i]` (P33).
    McScaled(usize, mc::McPutScaledFn),
    /// `mc/mct_scaled[i]` (P33).
    MctScaled(usize, mc::McPrepScaledFn),
    /// `resize/resize` (P33).
    Resize(resize::ResizeFn),
    /// `filmgrain/generate_grain_y` (P39).
    FgGrainY(fg::GenerateGrainYFn),
    /// `filmgrain/generate_grain_uv[i]` (P39; `i` = düzen dizini).
    FgGrainUv(usize, fg::GenerateGrainUvFn),
    /// `filmgrain/fgy_32x32xn` (P39).
    Fgy(fg::FgyFn),
    /// `filmgrain/fguv_32x32xn[i]` (P39; `i` = düzen dizini).
    Fguv(usize, fg::FguvFn),
    /// `itx/itxfm_add[tx][txtp]` (P43; `tx_idx`, `txtp_idx`).
    Itx(usize, usize, itx::ItxFn),
    /// `ipred/intra_pred[mode]` (P45).
    IpredAngular(usize, ipred::AngularPredFn),
    /// `ipred/cfl_ac[layout]` (P45).
    IpredCflAc(usize, ipred::CflAcFn),
    /// `ipred/cfl_pred[mode]` (P45).
    IpredCflPred(usize, ipred::CflPredFn),
    /// `ipred/pal_pred` (P45).
    IpredPalPred(ipred::PalPredFn),
    /// `pal/pal_idx_finish` (P45; bit derinliğinden bağımsız, `bit_depth == 0`).
    PalIdxFinish(pal::PalIdxFinishFn),
    /// `refmvs/load_tmvs` (P46; bit derinliğinden bağımsız, `bit_depth == 0`).
    RefmvsLoadTmvs(refmvs::LoadTmvsFn),
    /// `refmvs/save_tmvs` (P46; bit derinliğinden bağımsız, `bit_depth == 0`).
    RefmvsSaveTmvs(refmvs::SaveTmvsFn),
    /// `refmvs/splat_mv` (P46; bit derinliğinden bağımsız, `bit_depth == 0`).
    RefmvsSplatMv(refmvs::SplatMvFn),
    /// `loopfilter/loop_filter_sb[plane][dir]` (P46).
    LfSb(usize, usize, lf::LoopFilterSbFn),
    /// `looprestoration/wiener[i]` (P46).
    LrWiener(usize, lr::LrFilterFn),
    /// `looprestoration/sgr[i]` (P46).
    LrSgr(usize, lr::LrFilterFn),
    /// `msac/<işlev>` (P47; bit derinliğinden bağımsız, `bit_depth == 0`).
    Msac,
}

/// Kurulum bit kümesinin kelime sayısı (en çok `64·KURULU_KELIME` aday).
const KURULU_KELIME: usize = 32;

/// Bit derinliğinin tablo indeksi.
const fn bd_index(bit_depth: BitDepth) -> usize {
    match bit_depth {
        BitDepth::Eight => 0,
        BitDepth::Ten => 1,
        BitDepth::Twelve => 2,
    }
}

impl DspDispatch {
    /// Skaler başlangıç tablosu, derleme zamanında kurulur (tablo kurulumundaki
    /// `BPC` denetimi böylece derleme hatasıdır).
    const SKALER: Self = Self::scalar_only();

    /// Bütün yuvaları skaler tablo (yalnız kurulumun başlangıç noktası).
    const fn scalar_only() -> Self {
        Self {
            tables: [
                BitDepthDsp::scalar::<8>(BitDepth::Eight),
                BitDepthDsp::scalar::<10>(BitDepth::Ten),
                BitDepthDsp::scalar::<12>(BitDepth::Twelve),
            ],
            pal: PalDsp::SCALAR,
            refmvs: RefmvsDsp::SCALAR,
            installed: [0; KURULU_KELIME],
            sse2: None,
            ssse3: None,
            sse41: None,
            avx2: None,
            avx512icl: None,
            yavas_toplama: false,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            wasm_simd128: false,
            #[cfg(target_arch = "aarch64")]
            neon: None,
            #[cfg(target_arch = "aarch64")]
            dotprod: None,
        }
    }

    /// CPU bayraklarına göre tabloyu **bir kez** seçer (DSP-01/3).
    ///
    /// [`CpuFlags::detect`] çıktısı ya da maskelenmiş kümeler
    /// (`detected.difference(mask)`) buraya verilir.
    ///
    /// **F3b:** ISA yuvaları yetki belirteciyle kurulur. [`Avx2Token`] yalnız
    /// `flags` AVX2 içerdiğinde **ve** bu makinede AVX2 gerçekten tespit
    /// edildiğinde üretilir; AVX2 gövdelerinin işaretçileri yalnız belirteçle
    /// alınabilir (`x86::cdef::find_dir_fn` …).
    ///
    /// **F3c:** [`ISA_CANDIDATES`] içindeki varsayılan adaylar ilgili bit
    /// derinliği tablosunun yuvasına yazılır; diğer bütün yuvalar skalerdir.
    ///
    /// **P30:** bir yuvanın birden çok düzeyde adayı olabilir (SSSE3, SSE4.1,
    /// AVX2, AVX512ICL). Sonuç dav1d `init_x86` düzenidir: düzey sırasıyla
    /// ([`isa_rank`]) belirteci üretilen her varsayılan aday yuvaya yazılır ve
    /// aynı yuvanın alt düzey adayını siler. P35'ten beri kurulum aynı sonucu
    /// düzey azalan tek geçişle üretir (`install_isa`).
    #[must_use]
    pub fn select(flags: CpuFlags) -> Self {
        let mut table = Self::SKALER;
        table.sse2 = Sse2Token::new(flags);
        table.ssse3 = Ssse3Token::new(flags);
        table.sse41 = Sse41Token::new(flags);
        table.avx2 = Avx2Token::new(flags);
        table.avx512icl = Avx512IclToken::new(flags);
        table.yavas_toplama = flags.contains(CpuFlags::SLOW_GATHER);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            table.wasm_simd128 = flags.contains(CpuFlags::WASM_SIMD128)
                && CpuFlags::detect().contains(CpuFlags::WASM_SIMD128);
        }
        #[cfg(target_arch = "aarch64")]
        {
            table.neon = NeonToken::new(flags);
            table.dotprod = DotprodToken::new(flags);
        }
        table.install_isa();
        table
    }

    /// Varsayılan ISA adaylarını adayın bit derinliği tablosuna yazar: her
    /// yuvaya, belirteci üretilen ve bu derlemede gövdesi olan en yüksek
    /// düzeyli varsayılan aday (aynı düzeyde sonraki indeks) kurulur — düzey
    /// sırasıyla yürüyüp alt düzeyi silen dav1d `init_x86` düzeninin sonucu.
    ///
    /// P35: adaylar [`kurulum_sirasi`] ile düzey azalan sırada bir kez
    /// yürünür; kurulmuş yuva ve belirteci olmayan düzey atlanır. Önceki
    /// düzende her kurulan aday bütün aday listesini dizgi karşılaştırmasıyla
    /// tarıyordu (aday sayısının karesi; 359 adayda çözücü başına ~0,16 ms).
    fn install_isa(&mut self) {
        let mut dolu = [0u64; KURULU_KELIME];
        for &(rank, index, yuva) in kurulum_sirasi() {
            let (index, yuva) = (usize::from(index), usize::from(yuva));
            if !self.duzey_belirteci(rank) || dolu[yuva / 64] & (1u64 << (yuva % 64)) != 0 {
                continue;
            }
            let candidate = &ISA_CANDIDATES[index];
            let Some(entry) = self.candidate_entry(candidate) else {
                continue;
            };
            match entry {
                IsaEntry::PalIdxFinish(f) => self.pal.pal_idx_finish = f,
                IsaEntry::RefmvsLoadTmvs(f) => self.refmvs.load_tmvs = f,
                IsaEntry::RefmvsSaveTmvs(f) => self.refmvs.save_tmvs = f,
                IsaEntry::RefmvsSplatMv(f) => self.refmvs.splat_mv = f,
                IsaEntry::Msac => {}
                entry => {
                    let t = &mut self.tables[match candidate.bit_depth {
                        8 => 0,
                        10 => 1,
                        _ => 2,
                    }];
                    match entry {
                        IsaEntry::CdefDir(f) => t.cdef.dir = f,
                        IsaEntry::CdefFb(i, f) => t.cdef.fb[i] = f,
                        IsaEntry::Mc(i, f) => t.mc.mc[i] = f,
                        IsaEntry::Mct(i, f) => t.mc.mct[i] = f,
                        IsaEntry::Avg(f) => t.mc.avg = f,
                        IsaEntry::WAvg(f) => t.mc.w_avg = f,
                        IsaEntry::Mask(f) => t.mc.mask = f,
                        IsaEntry::WMask(i, f) => t.mc.w_mask[i] = f,
                        IsaEntry::Blend(f) => t.mc.blend = f,
                        IsaEntry::BlendV(f) => t.mc.blend_v = f,
                        IsaEntry::BlendH(f) => t.mc.blend_h = f,
                        IsaEntry::Warp(f) => t.mc.warp8x8 = f,
                        IsaEntry::WarpT(f) => t.mc.warp8x8t = f,
                        IsaEntry::EmuEdge(f) => t.mc.emu_edge = f,
                        IsaEntry::McScaled(i, f) => t.mc.mc_scaled[i] = f,
                        IsaEntry::MctScaled(i, f) => t.mc.mct_scaled[i] = f,
                        IsaEntry::Resize(f) => t.resize = f,
                        IsaEntry::FgGrainY(f) => t.fg.generate_grain_y = f,
                        IsaEntry::FgGrainUv(i, f) => t.fg.generate_grain_uv[i] = f,
                        IsaEntry::Fgy(f) => t.fg.fgy_32x32xn = f,
                        IsaEntry::Fguv(i, f) => t.fg.fguv_32x32xn[i] = f,
                        IsaEntry::Itx(tx_i, txtp_i, f) => t.itx.itxfm_add[tx_i][txtp_i] = f,
                        IsaEntry::IpredAngular(i, f) => t.ipred.intra_pred[i] = f,
                        IsaEntry::IpredCflAc(i, f) => t.ipred.cfl_ac[i] = f,
                        IsaEntry::IpredCflPred(i, f) => t.ipred.cfl_pred[i] = f,
                        IsaEntry::IpredPalPred(f) => t.ipred.pal_pred = f,
                        IsaEntry::LfSb(plane, dir, f) => t.lf.loop_filter_sb[plane][dir] = f,
                        IsaEntry::LrWiener(i, f) => t.lr.wiener[i] = f,
                        IsaEntry::LrSgr(i, f) => t.lr.sgr[i] = f,
                        IsaEntry::PalIdxFinish(_)
                        | IsaEntry::RefmvsLoadTmvs(_)
                        | IsaEntry::RefmvsSaveTmvs(_)
                        | IsaEntry::RefmvsSplatMv(_)
                        | IsaEntry::Msac => unreachable!(),
                    }
                }
            }
            dolu[yuva / 64] |= 1u64 << (yuva % 64);
            self.installed[index / 64] |= 1u64 << (index % 64);
        }
    }

    /// Düzeyin ([`isa_rank`]) belirteci bu kurulumda üretildi mi; bilinmeyen
    /// düzey kurulmaz.
    fn duzey_belirteci(&self, rank: u8) -> bool {
        match rank {
            #[cfg(all(
                not(target_arch = "aarch64"),
                not(all(target_arch = "wasm32", target_feature = "simd128"))
            ))]
            1 => self.sse2.is_some(),
            #[cfg(target_arch = "aarch64")]
            1 => self.neon.is_some(),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            1 => self.sse2.is_some() || self.wasm_simd128,
            #[cfg(not(target_arch = "aarch64"))]
            2 => self.ssse3.is_some(),
            #[cfg(target_arch = "aarch64")]
            2 => self.dotprod.is_some(),
            3 => self.sse41.is_some(),
            4 => self.avx2.is_some(),
            5 => self.avx512icl.is_some(),
            _ => false,
        }
    }

    /// Adayın yuva girişi — adayın düzeyinin belirteci bu kurulumda
    /// üretildiyse (varsayılan olup olmadığına bakılmaz; çekirdek ölçümü
    /// varsayılan olmayan gövdeyi de buradan alır). Belirteç yoksa ya da bu
    /// derlemede gövde yoksa `None`.
    #[must_use]
    pub fn candidate_entry(&self, candidate: &IsaCandidate) -> Option<IsaEntry> {
        if self.yavas_toplama && candidate.yavas_toplamada_kurulmaz() {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        {
            use x86::cdef as k;
            let fb = |f: cdef::CdefFilterFn| match candidate.field {
                "fb[0]" => Some(IsaEntry::CdefFb(0, f)),
                "fb[1]" => Some(IsaEntry::CdefFb(1, f)),
                "fb[2]" => Some(IsaEntry::CdefFb(2, f)),
                _ => None,
            };
            let hbd = candidate.bit_depth != 8;
            match (candidate.implementation, candidate.family, candidate.field) {
                ("avx2", "cdef", "dir") => self.avx2.map(|t| {
                    IsaEntry::CdefDir(if hbd {
                        k::find_dir16_fn(t)
                    } else {
                        k::find_dir_fn(t)
                    })
                }),
                ("avx2", "cdef", _) => self.avx2.and_then(|t| {
                    fb(if hbd {
                        k::filter16_fn(t)
                    } else {
                        k::filter_fn(t)
                    })
                }),
                ("ssse3", "cdef", "dir") => self.ssse3.map(|t| {
                    IsaEntry::CdefDir(if hbd {
                        x86::cdef_sse::find_dir16_ssse3_fn(t)
                    } else {
                        x86::cdef_sse::find_dir_ssse3_fn(t)
                    })
                }),
                ("ssse3", "cdef", _) => self.ssse3.and_then(|t| {
                    fb(if hbd {
                        x86::cdef_sse::filter16_ssse3_fn(t)
                    } else {
                        x86::cdef_sse::filter_ssse3_fn(t)
                    })
                }),
                ("sse4", "cdef", "dir") => self.sse41.map(|t| {
                    IsaEntry::CdefDir(if hbd {
                        x86::cdef_sse::find_dir16_sse41_fn(t)
                    } else {
                        x86::cdef_sse::find_dir_sse41_fn(t)
                    })
                }),
                ("sse4", "cdef", _) if !hbd => self
                    .sse41
                    .and_then(|t| fb(x86::cdef_sse::filter_sse41_fn(t))),
                ("avx512icl", "cdef", _) => self.avx512icl.and_then(|t| {
                    fb(if hbd {
                        x86::cdef_avx512::filter16_avx512_fn(t)
                    } else {
                        x86::cdef_avx512::filter_avx512_fn(t)
                    })
                }),
                // P36: AVX512ICL MC (`mc`/`mct`, bileşik, karışım, warp) ve
                // `resize`; 16 bpc gövdeleri 10 ve 12 bit yuvalarında ortak.
                // dav1d `mc_init_x86` gibi warp ve resize `SLOW_GATHER`'da
                // kurulmaz: aday girişi yukarıda verilmez
                // ([`IsaCandidate::yavas_toplamada_kurulmaz`]).
                ("avx512icl", "mc", field)
                    if field.starts_with("mc[") || field.starts_with("mct[") =>
                {
                    let (prep, f) = mc_slot_filter(field)?;
                    self.avx512icl.map(|t| match (prep, hbd) {
                        (false, false) => IsaEntry::Mc(f.index(), x86::mc_avx512::put_fn(t, f)),
                        (true, false) => IsaEntry::Mct(f.index(), x86::mc_avx512::prep_fn(t, f)),
                        (false, true) => IsaEntry::Mc(f.index(), x86::mc16_avx512::put16_fn(t, f)),
                        (true, true) => IsaEntry::Mct(f.index(), x86::mc16_avx512::prep16_fn(t, f)),
                    })
                }
                ("avx512icl", "mc", "avg") => self.avx512icl.map(|t| {
                    IsaEntry::Avg(if hbd {
                        x86::mc16_bilesik_avx512::avg_fn(t)
                    } else {
                        x86::mc_bilesik_avx512::avg_fn(t)
                    })
                }),
                ("avx512icl", "mc", "w_avg") => self.avx512icl.map(|t| {
                    IsaEntry::WAvg(if hbd {
                        x86::mc16_bilesik_avx512::w_avg_fn(t)
                    } else {
                        x86::mc_bilesik_avx512::w_avg_fn(t)
                    })
                }),
                ("avx512icl", "mc", "mask") => self.avx512icl.map(|t| {
                    IsaEntry::Mask(if hbd {
                        x86::mc16_bilesik_avx512::mask_fn(t)
                    } else {
                        x86::mc_bilesik_avx512::mask_fn(t)
                    })
                }),
                ("avx512icl", "mc", field) if field.starts_with("w_mask[") => {
                    let i = field
                        .strip_prefix("w_mask[")?
                        .strip_suffix(']')?
                        .parse::<usize>()
                        .ok()
                        .filter(|&i| i < mc::N_W_MASK)?;
                    self.avx512icl.map(|t| {
                        IsaEntry::WMask(
                            i,
                            if hbd {
                                x86::mc16_bilesik_avx512::w_mask_fn(t, i)
                            } else {
                                x86::mc_bilesik_avx512::w_mask_fn(t, i)
                            },
                        )
                    })
                }
                ("avx512icl", "mc", "blend") => self.avx512icl.map(|t| {
                    IsaEntry::Blend(if hbd {
                        x86::mc16_karisim_avx512::blend_fn(t)
                    } else {
                        x86::mc_karisim_avx512::blend_fn(t)
                    })
                }),
                ("avx512icl", "mc", "blend_v") => self.avx512icl.map(|t| {
                    IsaEntry::BlendV(if hbd {
                        x86::mc16_karisim_avx512::blend_v_fn(t)
                    } else {
                        x86::mc_karisim_avx512::blend_v_fn(t)
                    })
                }),
                ("avx512icl", "mc", "blend_h") => self.avx512icl.map(|t| {
                    IsaEntry::BlendH(if hbd {
                        x86::mc16_karisim_avx512::blend_h_fn(t)
                    } else {
                        x86::mc_karisim_avx512::blend_h_fn(t)
                    })
                }),
                ("avx512icl", "mc", "warp8x8") => self.avx512icl.map(|t| {
                    IsaEntry::Warp(if hbd {
                        x86::mc16_warp_avx512::warp8x8_fn(t)
                    } else {
                        x86::mc_warp_avx512::warp8x8_fn(t)
                    })
                }),
                ("avx512icl", "mc", "warp8x8t") => self.avx512icl.map(|t| {
                    IsaEntry::WarpT(if hbd {
                        x86::mc16_warp_avx512::warp8x8t_fn(t)
                    } else {
                        x86::mc_warp_avx512::warp8x8t_fn(t)
                    })
                }),
                ("avx512icl", "resize", "resize") => self.avx512icl.map(|t| {
                    IsaEntry::Resize(if hbd {
                        x86::buyutme16_avx512::resize_fn(t)
                    } else {
                        x86::buyutme_avx512::resize_fn(t)
                    })
                }),
                // P32: 8 bpc `mc[0..9]` / `mct[0..9]` (kopyasız AVX2 gövdeleri).
                // P33: 8 bpc bileşik `avg`/`w_avg`/`mask`/`w_mask[0..2]`.
                ("avx2", "mc", "avg") if !hbd => {
                    self.avx2.map(|t| IsaEntry::Avg(x86::mc_bilesik::avg_fn(t)))
                }
                ("avx2", "mc", "w_avg") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::WAvg(x86::mc_bilesik::w_avg_fn(t))),
                ("avx2", "mc", "mask") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::Mask(x86::mc_bilesik::mask_fn(t))),
                // P33: 8 bpc karışım `blend`/`blend_v`/`blend_h`.
                ("avx2", "mc", "blend") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::Blend(x86::mc_karisim::blend_fn(t))),
                ("avx2", "mc", "blend_v") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::BlendV(x86::mc_karisim::blend_v_fn(t))),
                ("avx2", "mc", "blend_h") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::BlendH(x86::mc_karisim::blend_h_fn(t))),
                // P33: 8 bpc ölçekli `mc_scaled[0..9]` / `mct_scaled[0..9]`.
                ("avx2", "mc", field)
                    if !hbd
                        && (field.starts_with("mc_scaled[")
                            || field.starts_with("mct_scaled[")) =>
                {
                    let prep = field.starts_with("mct_scaled[");
                    let ad = field
                        .strip_prefix(if prep { "mct_scaled[" } else { "mc_scaled[" })?
                        .strip_suffix(']')?;
                    let i = FILTER_2D_NAMES.iter().position(|n| *n == ad)?;
                    let f = mc::Filter2d::from_index(i)?;
                    self.avx2.map(|t| {
                        if prep {
                            IsaEntry::MctScaled(i, x86::mc_olcekli::prep_scaled_fn(t, f))
                        } else {
                            IsaEntry::McScaled(i, x86::mc_olcekli::put_scaled_fn(t, f))
                        }
                    })
                }
                // P33: 8 bpc superres büyütme.
                ("avx2", "resize", "resize") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::Resize(x86::buyutme::resize_fn(t))),
                // P33: 8 bpc `emu_edge`.
                ("avx2", "mc", "emu_edge") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::EmuEdge(x86::mc_kenar::emu_edge_fn(t))),
                // P33: 8 bpc `warp8x8`/`warp8x8t`.
                ("avx2", "mc", "warp8x8") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::Warp(x86::mc_warp::warp8x8_fn(t))),
                ("avx2", "mc", "warp8x8t") if !hbd => self
                    .avx2
                    .map(|t| IsaEntry::WarpT(x86::mc_warp::warp8x8t_fn(t))),
                ("avx2", "mc", field) if !hbd && field.starts_with("w_mask[") => {
                    let i = field
                        .strip_prefix("w_mask[")?
                        .strip_suffix(']')?
                        .parse::<usize>()
                        .ok()
                        .filter(|&i| i < mc::N_W_MASK)?;
                    self.avx2
                        .map(|t| IsaEntry::WMask(i, x86::mc_bilesik::w_mask_fn(t, i)))
                }
                // P34: 16 bpc ölçekli `mc_scaled[0..9]` / `mct_scaled[0..9]`.
                ("avx2", "mc", field)
                    if hbd
                        && (field.starts_with("mc_scaled[")
                            || field.starts_with("mct_scaled[")) =>
                {
                    let prep = field.starts_with("mct_scaled[");
                    let ad = field
                        .strip_prefix(if prep { "mct_scaled[" } else { "mc_scaled[" })?
                        .strip_suffix(']')?;
                    let i = FILTER_2D_NAMES.iter().position(|n| *n == ad)?;
                    let f = mc::Filter2d::from_index(i)?;
                    self.avx2.map(|t| {
                        if prep {
                            IsaEntry::MctScaled(i, x86::mc16_olcekli::prep_scaled_fn(t, f))
                        } else {
                            IsaEntry::McScaled(i, x86::mc16_olcekli::put_scaled_fn(t, f))
                        }
                    })
                }
                // P34: 16 bpc superres büyütme.
                ("avx2", "resize", "resize") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::Resize(x86::buyutme16::resize_fn(t))),
                // P34: 16 bpc `emu_edge`.
                ("avx2", "mc", "emu_edge") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::EmuEdge(x86::mc16_kenar::emu_edge_fn(t))),
                // P34: 16 bpc `warp8x8`/`warp8x8t`.
                ("avx2", "mc", "warp8x8") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::Warp(x86::mc16_warp::warp8x8_fn(t))),
                ("avx2", "mc", "warp8x8t") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::WarpT(x86::mc16_warp::warp8x8t_fn(t))),
                // P34: 16 bpc karışım `blend`/`blend_v`/`blend_h`.
                ("avx2", "mc", "blend") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::Blend(x86::mc16_karisim::blend_fn(t))),
                ("avx2", "mc", "blend_v") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::BlendV(x86::mc16_karisim::blend_v_fn(t))),
                ("avx2", "mc", "blend_h") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::BlendH(x86::mc16_karisim::blend_h_fn(t))),
                // P34: 16 bpc (10/12 bit) bileşik `avg`/`w_avg`/`mask`/`w_mask[0..2]`.
                ("avx2", "mc", "avg") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::Avg(x86::mc16_bilesik::avg_fn(t))),
                ("avx2", "mc", "w_avg") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::WAvg(x86::mc16_bilesik::w_avg_fn(t))),
                ("avx2", "mc", "mask") if hbd => self
                    .avx2
                    .map(|t| IsaEntry::Mask(x86::mc16_bilesik::mask_fn(t))),
                ("avx2", "mc", field) if hbd && field.starts_with("w_mask[") => {
                    let i = field
                        .strip_prefix("w_mask[")?
                        .strip_suffix(']')?
                        .parse::<usize>()
                        .ok()
                        .filter(|&i| i < mc::N_W_MASK)?;
                    self.avx2
                        .map(|t| IsaEntry::WMask(i, x86::mc16_bilesik::w_mask_fn(t, i)))
                }
                // P34: 16 bpc (10/12 bit) `mc[0..9]` / `mct[0..9]`; iki bit derinliği
                // aynı gövdeyi `bitdepth_max` ile paylaşır.
                ("avx2", "mc", field)
                    if hbd && (field.starts_with("mc[") || field.starts_with("mct[")) =>
                {
                    let (prep, f) = mc_slot_filter(field)?;
                    self.avx2.map(|t| {
                        if prep {
                            IsaEntry::Mct(f.index(), x86::mc16::prep16_fn(t, f))
                        } else {
                            IsaEntry::Mc(f.index(), x86::mc16::put16_fn(t, f))
                        }
                    })
                }
                // P35: 16 bpc (10/12 bit) SSSE3 `mc[0..9]` / `mct[0..9]`.
                ("ssse3", "mc", field)
                    if hbd && (field.starts_with("mc[") || field.starts_with("mct[")) =>
                {
                    let (prep, f) = mc_slot_filter(field)?;
                    self.ssse3.map(|t| {
                        if prep {
                            IsaEntry::Mct(f.index(), x86::mc16_ssse3::prep16_fn(t, f))
                        } else {
                            IsaEntry::Mc(f.index(), x86::mc16_ssse3::put16_fn(t, f))
                        }
                    })
                }
                // P35: 16 bpc (10/12 bit) SSSE3 bileşik `avg`/`w_avg`/`mask`/`w_mask[0..2]`.
                ("ssse3", "mc", "avg") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Avg(x86::mc16_bilesik_ssse3::avg_fn(t))),
                ("ssse3", "mc", "w_avg") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::WAvg(x86::mc16_bilesik_ssse3::w_avg_fn(t))),
                ("ssse3", "mc", "mask") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Mask(x86::mc16_bilesik_ssse3::mask_fn(t))),
                ("ssse3", "mc", field) if hbd && field.starts_with("w_mask[") => {
                    let i = field
                        .strip_prefix("w_mask[")?
                        .strip_suffix(']')?
                        .parse::<usize>()
                        .ok()
                        .filter(|&i| i < mc::N_W_MASK)?;
                    self.ssse3
                        .map(|t| IsaEntry::WMask(i, x86::mc16_bilesik_ssse3::w_mask_fn(t, i)))
                }
                // P35: 16 bpc (10/12 bit) SSSE3 karışım `blend`/`blend_v`/`blend_h`.
                ("ssse3", "mc", "blend") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Blend(x86::mc16_karisim_ssse3::blend_fn(t))),
                ("ssse3", "mc", "blend_v") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::BlendV(x86::mc16_karisim_ssse3::blend_v_fn(t))),
                ("ssse3", "mc", "blend_h") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::BlendH(x86::mc16_karisim_ssse3::blend_h_fn(t))),
                // P35: 16 bpc (10/12 bit) SSSE3 `warp8x8`/`warp8x8t`.
                ("ssse3", "mc", "warp8x8") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Warp(x86::mc16_warp_ssse3::warp8x8_fn(t))),
                ("ssse3", "mc", "warp8x8t") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::WarpT(x86::mc16_warp_ssse3::warp8x8t_fn(t))),
                // P35: 16 bpc (10/12 bit) SSSE3 `emu_edge`.
                ("ssse3", "mc", "emu_edge") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::EmuEdge(x86::mc16_kenar_ssse3::emu_edge_fn(t))),
                // P35: 16 bpc (10/12 bit) SSSE3 ölçekli `mc_scaled[0..9]` / `mct_scaled[0..9]`.
                ("ssse3", "mc", field)
                    if hbd
                        && (field.starts_with("mc_scaled[")
                            || field.starts_with("mct_scaled[")) =>
                {
                    let prep = field.starts_with("mct_scaled[");
                    let ad = field
                        .strip_prefix(if prep { "mct_scaled[" } else { "mc_scaled[" })?
                        .strip_suffix(']')?;
                    let i = FILTER_2D_NAMES.iter().position(|n| *n == ad)?;
                    let f = mc::Filter2d::from_index(i)?;
                    self.ssse3.map(|t| {
                        if prep {
                            IsaEntry::MctScaled(i, x86::mc16_olcekli_ssse3::prep_scaled_fn(t, f))
                        } else {
                            IsaEntry::McScaled(i, x86::mc16_olcekli_ssse3::put_scaled_fn(t, f))
                        }
                    })
                }
                // P35: 16 bpc (10/12 bit) SSSE3 superres büyütme.
                ("ssse3", "resize", "resize") if hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Resize(x86::buyutme16_ssse3::resize_fn(t))),
                // P35: 8 bpc SSSE3 `mc[0..9]` / `mct[0..9]`.
                ("ssse3", "mc", field)
                    if !hbd && (field.starts_with("mc[") || field.starts_with("mct[")) =>
                {
                    let (prep, f) = mc_slot_filter(field)?;
                    self.ssse3.map(|t| {
                        if prep {
                            IsaEntry::Mct(f.index(), x86::mc_ssse3::prep_fn(t, f))
                        } else {
                            IsaEntry::Mc(f.index(), x86::mc_ssse3::put_fn(t, f))
                        }
                    })
                }
                // P35: 8 bpc SSSE3 bileşik `avg`/`w_avg`/`mask`/`w_mask[0..2]`.
                ("ssse3", "mc", "avg") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Avg(x86::mc_bilesik_ssse3::avg_fn(t))),
                ("ssse3", "mc", "w_avg") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::WAvg(x86::mc_bilesik_ssse3::w_avg_fn(t))),
                ("ssse3", "mc", "mask") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Mask(x86::mc_bilesik_ssse3::mask_fn(t))),
                ("ssse3", "mc", field) if !hbd && field.starts_with("w_mask[") => {
                    let i = field
                        .strip_prefix("w_mask[")?
                        .strip_suffix(']')?
                        .parse::<usize>()
                        .ok()
                        .filter(|&i| i < mc::N_W_MASK)?;
                    self.ssse3
                        .map(|t| IsaEntry::WMask(i, x86::mc_bilesik_ssse3::w_mask_fn(t, i)))
                }
                // P35: 8 bpc SSSE3 karışım `blend`/`blend_v`/`blend_h`.
                ("ssse3", "mc", "blend") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Blend(x86::mc_karisim_ssse3::blend_fn(t))),
                ("ssse3", "mc", "blend_v") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::BlendV(x86::mc_karisim_ssse3::blend_v_fn(t))),
                ("ssse3", "mc", "blend_h") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::BlendH(x86::mc_karisim_ssse3::blend_h_fn(t))),
                // P35: 8 bpc SSSE3 ve SSE4.1 `warp8x8`/`warp8x8t`.
                ("ssse3", "mc", "warp8x8") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Warp(x86::mc_warp_ssse3::warp8x8_fn(t))),
                ("ssse3", "mc", "warp8x8t") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::WarpT(x86::mc_warp_ssse3::warp8x8t_fn(t))),
                ("sse4", "mc", "warp8x8") if !hbd => self
                    .sse41
                    .map(|t| IsaEntry::Warp(x86::mc_warp_sse41::warp8x8_fn(t))),
                ("sse4", "mc", "warp8x8t") if !hbd => self
                    .sse41
                    .map(|t| IsaEntry::WarpT(x86::mc_warp_sse41::warp8x8t_fn(t))),
                // P35: 8 bpc SSSE3 `emu_edge`.
                ("ssse3", "mc", "emu_edge") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::EmuEdge(x86::mc_kenar_ssse3::emu_edge_fn(t))),
                // P35: 8 bpc SSSE3 ölçekli `mc_scaled[0..9]` / `mct_scaled[0..9]`.
                ("ssse3", "mc", field)
                    if !hbd
                        && (field.starts_with("mc_scaled[")
                            || field.starts_with("mct_scaled[")) =>
                {
                    let prep = field.starts_with("mct_scaled[");
                    let ad = field
                        .strip_prefix(if prep { "mct_scaled[" } else { "mc_scaled[" })?
                        .strip_suffix(']')?;
                    let i = FILTER_2D_NAMES.iter().position(|n| *n == ad)?;
                    let f = mc::Filter2d::from_index(i)?;
                    self.ssse3.map(|t| {
                        if prep {
                            IsaEntry::MctScaled(i, x86::mc_olcekli_ssse3::prep_scaled_fn(t, f))
                        } else {
                            IsaEntry::McScaled(i, x86::mc_olcekli_ssse3::put_scaled_fn(t, f))
                        }
                    })
                }
                // P35: 8 bpc SSSE3 superres büyütme.
                ("ssse3", "resize", "resize") if !hbd => self
                    .ssse3
                    .map(|t| IsaEntry::Resize(x86::buyutme_ssse3::resize_fn(t))),
                ("avx2", "mc", field) if !hbd => {
                    let (prep, f) = mc_slot_filter(field)?;
                    self.avx2.map(|t| {
                        if prep {
                            IsaEntry::Mct(f.index(), x86::mc::prep_fn(t, f))
                        } else {
                            IsaEntry::Mc(f.index(), x86::mc::put_fn(t, f))
                        }
                    })
                }
                // P39 / P42: 8 ve 16 bpc (10/12 bit) AVX2 film grain (`fgy`/`fguv`
                // `SLOW_GATHER`'da kurulmaz: [`IsaCandidate::yavas_toplamada_kurulmaz`]).
                ("avx2", "filmgrain", "generate_grain_y") => self.avx2.map(|t| {
                    IsaEntry::FgGrainY(if hbd {
                        x86::fg16::generate_grain_y_fn(t)
                    } else {
                        x86::fg::generate_grain_y_fn(t)
                    })
                }),
                ("avx2", "filmgrain", field) if field.starts_with("generate_grain_uv[") => {
                    let duzen = fg_duzen(field, "generate_grain_uv")?;
                    self.avx2.map(|t| {
                        IsaEntry::FgGrainUv(
                            duzen,
                            if hbd {
                                x86::fg16::generate_grain_uv_fn(t, duzen)
                            } else {
                                x86::fg::generate_grain_uv_fn(t, duzen)
                            },
                        )
                    })
                }
                ("avx2", "filmgrain", "fgy_32x32xn") => self.avx2.map(|t| {
                    IsaEntry::Fgy(if hbd {
                        x86::fg16::fgy_fn(t)
                    } else {
                        x86::fg::fgy_fn(t)
                    })
                }),
                ("avx2", "filmgrain", field) => {
                    let duzen = fg_duzen(field, "fguv_32x32xn")?;
                    self.avx2.map(|t| {
                        IsaEntry::Fguv(
                            duzen,
                            if hbd {
                                x86::fg16::fguv_fn(t, duzen)
                            } else {
                                x86::fg::fguv_fn(t, duzen)
                            },
                        )
                    })
                }
                // P41: 8 bpc SSSE3 film grain (`SLOW_GATHER`'da `fgy`/`fguv` de kurulur:
                // dav1d `src/x86/filmgrain.h:48-72`).
                // P42: 16 bpc (10/12 bit) SSSE3 film grain (`x86::fg16_ssse3`).
                ("ssse3", "filmgrain", "generate_grain_y") => self.ssse3.map(|t| {
                    IsaEntry::FgGrainY(if hbd {
                        x86::fg16_ssse3::generate_grain_y_fn(t)
                    } else {
                        x86::fg_ssse3::generate_grain_y_fn(t)
                    })
                }),
                ("ssse3", "filmgrain", field) if field.starts_with("generate_grain_uv[") => {
                    let duzen = fg_duzen(field, "generate_grain_uv")?;
                    self.ssse3.map(|t| {
                        IsaEntry::FgGrainUv(
                            duzen,
                            if hbd {
                                x86::fg16_ssse3::generate_grain_uv_fn(t, duzen)
                            } else {
                                x86::fg_ssse3::generate_grain_uv_fn(t, duzen)
                            },
                        )
                    })
                }
                ("ssse3", "filmgrain", "fgy_32x32xn") => self.ssse3.map(|t| {
                    IsaEntry::Fgy(if hbd {
                        x86::fg16_ssse3::fgy_fn(t)
                    } else {
                        x86::fg_ssse3::fgy_fn(t)
                    })
                }),
                ("ssse3", "filmgrain", field) => {
                    let duzen = fg_duzen(field, "fguv_32x32xn")?;
                    self.ssse3.map(|t| {
                        IsaEntry::Fguv(
                            duzen,
                            if hbd {
                                x86::fg16_ssse3::fguv_fn(t, duzen)
                            } else {
                                x86::fg_ssse3::fguv_fn(t, duzen)
                            },
                        )
                    })
                }
                // P41 / P42: 8 ve 16 bpc (10/12 bit) AVX512ICL film grain (8 bpc `vpermb`
                // `SLOW_GATHER`'da da kurulur; 16 bpc `vpgatherdd` yalnız `!SLOW_GATHER`:
                // dav1d `src/x86/filmgrain.h:76`: `BITDEPTH == 8 || !(flags & SLOW_GATHER)`).
                ("avx512icl", "filmgrain", "fgy_32x32xn") => self.avx512icl.map(|t| {
                    IsaEntry::Fgy(if hbd {
                        x86::fg16_avx512::fgy_fn(t)
                    } else {
                        x86::fg_avx512::fgy_fn(t)
                    })
                }),
                ("avx512icl", "filmgrain", field) => {
                    let duzen = fg_duzen(field, "fguv_32x32xn")?;
                    self.avx512icl.map(|t| {
                        IsaEntry::Fguv(
                            duzen,
                            if hbd {
                                x86::fg16_avx512::fguv_fn(t, duzen)
                            } else {
                                x86::fg_avx512::fguv_fn(t, duzen)
                            },
                        )
                    })
                }
                // P43: Kayıpsız 4×4 WHT_WHT (SSE2 D02583/D02584, AVX2 D02581/D02582).
                ("sse2", "itx", "itxfm_add[TX_4X4][WHT_WHT]") => self.sse2.map(|t| {
                    IsaEntry::Itx(
                        itx::TxSize::Tx4x4.index(),
                        itx::TxType::WhtWht.index(),
                        if hbd {
                            x86::itx_wht::wht16_sse2_fn(t, candidate.bit_depth)
                        } else {
                            x86::itx_wht::wht8_sse2_fn(t)
                        },
                    )
                }),
                ("avx2", "itx", "itxfm_add[TX_4X4][WHT_WHT]") => self.avx2.map(|t| {
                    IsaEntry::Itx(
                        itx::TxSize::Tx4x4.index(),
                        itx::TxType::WhtWht.index(),
                        if hbd {
                            x86::itx_wht::wht16_avx2_fn(t, candidate.bit_depth)
                        } else {
                            x86::itx_wht::wht8_avx2_fn(t)
                        },
                    )
                }),
                // P43 Adım 2: 155 WHT-dışı 8 bpc SSSE3 ters dönüşüm kaydı (`itx_ssse3` + `itx_buyuk_ssse3`).
                ("ssse3", "itx", field) if !hbd && field != "itxfm_add[TX_4X4][WHT_WHT]" => {
                    let (tx, txtp) = itx_slot(field)?;
                    let t = self.ssse3?;
                    let f = x86::itx_ssse3::itx8_ssse3_fn(t, tx, txtp)
                        .or_else(|| x86::itx_buyuk_ssse3::itx8_buyuk_ssse3_fn(t, tx, txtp))?;
                    Some(IsaEntry::Itx(tx.index(), txtp.index(), f))
                }
                // P43 Adım 2: 155 WHT-dışı 8 bpc AVX2 ters dönüşüm kaydı (`itx` + `itx_buyuk`).
                ("avx2", "itx", field) if !hbd && field != "itxfm_add[TX_4X4][WHT_WHT]" => {
                    let (tx, txtp) = itx_slot(field)?;
                    let t = self.avx2?;
                    let f = x86::itx::itx8_avx2_fn(t, tx, txtp)
                        .or_else(|| x86::itx_buyuk::itx8_buyuk_avx2_fn(t, tx, txtp))?;
                    Some(IsaEntry::Itx(tx.index(), txtp.index(), f))
                }
                // P43 Adım 3: 155 WHT-dışı 8 bpc AVX512ICL ters dönüşüm kaydı (`itx_avx512` + `itx_buyuk_avx512`;
                // `SLOW_GATHER`'da da kurulur: dav1d `src/x86/itx.h:221-244`, 0 gather/scatter).
                ("avx512icl", "itx", field) if !hbd && field != "itxfm_add[TX_4X4][WHT_WHT]" => {
                    let (tx, txtp) = itx_slot(field)?;
                    let t = self.avx512icl?;
                    let f = x86::itx_avx512::itx8_avx512_fn(t, tx, txtp)
                        .or_else(|| x86::itx_buyuk_avx512::itx8_buyuk_avx512_fn(t, tx, txtp))?;
                    Some(IsaEntry::Itx(tx.index(), txtp.index(), f))
                }
                // P43 Adım 3: 155 WHT-dışı 10 bpc SSE4.1 ters dönüşüm kaydı (`itx16_sse4` + `itx16_buyuk_sse4`;
                // dav1d `src/x86/itx.h:178`: yalnız `bpc == 10`, 12 bpc skalerde kalır).
                ("sse4", "itx", field)
                    if candidate.bit_depth == 10 && field != "itxfm_add[TX_4X4][WHT_WHT]" =>
                {
                    let (tx, txtp) = itx_slot(field)?;
                    let t = self.sse41?;
                    let f = x86::itx16_sse4::itx10_sse4_fn(t, tx, txtp)
                        .or_else(|| x86::itx16_buyuk_sse4::itx10_buyuk_sse4_fn(t, tx, txtp))?;
                    Some(IsaEntry::Itx(tx.index(), txtp.index(), f))
                }
                // P44 Adım 1: 155 WHT-dışı 10 bpc AVX2 ters dönüşüm kaydı (`itx16_10` + `itx16_10_buyuk`;
                // dav1d `src/x86/itx.h:180-203`: yalnız `bpc == 10`).
                ("avx2", "itx", field)
                    if candidate.bit_depth == 10 && field != "itxfm_add[TX_4X4][WHT_WHT]" =>
                {
                    let (tx, txtp) = itx_slot(field)?;
                    let t = self.avx2?;
                    let f = x86::itx16_10::itx10_avx2_lookup(tx, txtp)
                        .or_else(|| x86::itx16_10_buyuk::itx10_buyuk_avx2_lookup(tx, txtp))?(
                        t
                    );
                    Some(IsaEntry::Itx(tx.index(), txtp.index(), f))
                }
                // P44 Adım 1 (Paket B): 147 WHT-dışı 12 bpc AVX2 ters dönüşüm kaydı (`itx16_12` + `itx16_12_buyuk`;
                // dav1d `src/x86/itx.h:249-263`: yalnız `bpc == 12`, en büyük 8 DCT_DCT skalerde kalır).
                ("avx2", "itx", field)
                    if candidate.bit_depth == 12 && field != "itxfm_add[TX_4X4][WHT_WHT]" =>
                {
                    let (tx, txtp) = itx_slot(field)?;
                    let t = self.avx2?;
                    let f = x86::itx16_12::itx12_avx2_lookup(t, tx, txtp)
                        .or_else(|| x86::itx16_12_buyuk::itx12_buyuk_avx2_lookup(t, tx, txtp))?;
                    Some(IsaEntry::Itx(tx.index(), txtp.index(), f))
                }
                // P44 Adım 2: 75 WHT-dışı 10 bpc AVX512ICL ters dönüşüm kaydı (`itx16_avx512` 60 + `itx16_buyuk_avx512` 15;
                // dav1d `src/x86/itx.h:289-304`: yalnız `bpc == 10` ve `w >= 8 && h >= 8`).
                ("avx512icl", "itx", field)
                    if candidate.bit_depth == 10 && field != "itxfm_add[TX_4X4][WHT_WHT]" =>
                {
                    let (tx, txtp) = itx_slot(field)?;
                    let t = self.avx512icl?;
                    let f = x86::itx16_avx512::itx10_avx512_fn(t, tx, txtp)
                        .or_else(|| x86::itx16_buyuk_avx512::itx10_buyuk_avx512_fn(t, tx, txtp))?;
                    Some(IsaEntry::Itx(tx.index(), txtp.index(), f))
                }
                // P45 Adım 1: Bit derinliğinden bağımsız `pal/pal_idx_finish` (`bit_depth == 0`;
                // dav1d `src/x86/pal.h:34-50`: SSSE3, AVX2 ve AVX512ICL; `SLOW_GATHER` engeli yok).
                ("ssse3", "pal", "pal_idx_finish") if candidate.bit_depth == 0 => self
                    .ssse3
                    .map(|t| IsaEntry::PalIdxFinish(x86::pal_ssse3::pal_idx_finish_ssse3_fn(t))),
                ("avx2", "pal", "pal_idx_finish") if candidate.bit_depth == 0 => self
                    .avx2
                    .map(|t| IsaEntry::PalIdxFinish(x86::pal::pal_idx_finish_avx2_fn(t))),
                ("avx512icl", "pal", "pal_idx_finish") if candidate.bit_depth == 0 => {
                    self.avx512icl.map(|t| {
                        IsaEntry::PalIdxFinish(x86::pal_avx512::pal_idx_finish_avx512icl_fn(t))
                    })
                }
                // P45 Adım 2: 8 ve 16 bpc (10/12 bit ortak) intra tahmin (`ipred`, 112 kayıt / 165 aday:
                // 8 bpc 22 SSSE3 + 22 AVX2 + 15 AVX512ICL = 59; 16 bpc 22 SSSE3 + 22 AVX2 + 9 AVX512ICL = 53 kayıt / 106 aday).
                ("ssse3", "ipred", "pal_pred") => self.ssse3.map(|t| {
                    IsaEntry::IpredPalPred(if hbd {
                        x86::ipred16_ssse3::pal_pred_fn(t)
                    } else {
                        x86::ipred_ssse3::pal_pred_fn(t)
                    })
                }),
                ("avx2", "ipred", "pal_pred") => self.avx2.map(|t| {
                    IsaEntry::IpredPalPred(if hbd {
                        x86::ipred16::pal_pred_fn(t)
                    } else {
                        x86::ipred::pal_pred_fn(t)
                    })
                }),
                ("avx512icl", "ipred", "pal_pred") => self.avx512icl.map(|t| {
                    IsaEntry::IpredPalPred(if hbd {
                        x86::ipred16_avx512::pal_pred_fn(t)
                    } else {
                        x86::ipred_avx512::pal_pred_fn(t)
                    })
                }),
                ("ssse3", "ipred", field) if field.starts_with("intra_pred[") => {
                    let mode = ipred_intra_slot(field)?;
                    self.ssse3.map(|t| {
                        IsaEntry::IpredAngular(
                            mode.index(),
                            if hbd {
                                x86::ipred16_ssse3::intra_pred_fn(t, mode)
                            } else {
                                x86::ipred_ssse3::intra_pred_fn(t, mode)
                            },
                        )
                    })
                }
                ("avx2", "ipred", field) if field.starts_with("intra_pred[") => {
                    let mode = ipred_intra_slot(field)?;
                    self.avx2.map(|t| {
                        IsaEntry::IpredAngular(
                            mode.index(),
                            if hbd {
                                x86::ipred16::intra_pred_fn(t, mode)
                            } else {
                                x86::ipred::intra_pred_fn(t, mode)
                            },
                        )
                    })
                }
                ("avx512icl", "ipred", field) if field.starts_with("intra_pred[") => {
                    let mode = ipred_intra_slot(field)?;
                    let t = self.avx512icl?;
                    let f = if hbd {
                        x86::ipred16_avx512::intra_pred_fn(t, mode)?
                    } else {
                        x86::ipred_avx512::intra_pred_fn(t, mode)
                    };
                    Some(IsaEntry::IpredAngular(mode.index(), f))
                }
                ("ssse3", "ipred", field) if field.starts_with("cfl_ac[") => {
                    let layout = fg_duzen(field, "cfl_ac")?;
                    let t = self.ssse3?;
                    let f = if hbd {
                        x86::ipred16_ssse3::cfl_ac_fn(t, layout)?
                    } else {
                        x86::ipred_ssse3::cfl_ac_fn(t, layout)?
                    };
                    Some(IsaEntry::IpredCflAc(layout, f))
                }
                ("avx2", "ipred", field) if field.starts_with("cfl_ac[") => {
                    let layout = fg_duzen(field, "cfl_ac")?;
                    let t = self.avx2?;
                    let f = if hbd {
                        x86::ipred16::cfl_ac_fn(t, layout)?
                    } else {
                        x86::ipred::cfl_ac_fn(t, layout)?
                    };
                    Some(IsaEntry::IpredCflAc(layout, f))
                }
                ("ssse3", "ipred", field) if field.starts_with("cfl_pred[") => {
                    let mode = ipred_cfl_pred_slot(field)?;
                    self.ssse3.map(|t| {
                        IsaEntry::IpredCflPred(
                            mode.index(),
                            if hbd {
                                x86::ipred16_ssse3::cfl_pred_fn(t, mode)
                            } else {
                                x86::ipred_ssse3::cfl_pred_fn(t, mode)
                            },
                        )
                    })
                }
                ("avx2", "ipred", field) if field.starts_with("cfl_pred[") => {
                    let mode = ipred_cfl_pred_slot(field)?;
                    self.avx2.map(|t| {
                        IsaEntry::IpredCflPred(
                            mode.index(),
                            if hbd {
                                x86::ipred16::cfl_pred_fn(t, mode)
                            } else {
                                x86::ipred::cfl_pred_fn(t, mode)
                            },
                        )
                    })
                }
                // P46 Adım 1: Bit derinliğinden bağımsız `refmvs` (`bit_depth == 0`, 14 kayıt / 7 aday:
                // `load_tmvs` SSE4.1, `save_tmvs` SSSE3/AVX2/AVX512ICL, `splat_mv` SSE2/AVX2/AVX512ICL).
                ("sse2", "refmvs", "splat_mv") if candidate.bit_depth == 0 => self
                    .sse2
                    .map(|t| IsaEntry::RefmvsSplatMv(x86::refmvs_sse2::splat_mv_sse2_fn(t))),
                ("ssse3", "refmvs", "save_tmvs") if candidate.bit_depth == 0 => self
                    .ssse3
                    .map(|t| IsaEntry::RefmvsSaveTmvs(x86::refmvs_ssse3::save_tmvs_ssse3_fn(t))),
                ("sse4", "refmvs", "load_tmvs") if candidate.bit_depth == 0 => self
                    .sse41
                    .map(|t| IsaEntry::RefmvsLoadTmvs(x86::refmvs_sse41::load_tmvs_sse41_fn(t))),
                ("avx2", "refmvs", "save_tmvs") if candidate.bit_depth == 0 => self
                    .avx2
                    .map(|t| IsaEntry::RefmvsSaveTmvs(x86::refmvs::save_tmvs_avx2_fn(t))),
                ("avx2", "refmvs", "splat_mv") if candidate.bit_depth == 0 => self
                    .avx2
                    .map(|t| IsaEntry::RefmvsSplatMv(x86::refmvs::splat_mv_avx2_fn(t))),
                ("avx512icl", "refmvs", "save_tmvs") if candidate.bit_depth == 0 => {
                    self.avx512icl.map(|t| {
                        IsaEntry::RefmvsSaveTmvs(x86::refmvs_avx512::save_tmvs_avx512icl_fn(t))
                    })
                }
                ("avx512icl", "refmvs", "splat_mv") if candidate.bit_depth == 0 => self
                    .avx512icl
                    .map(|t| IsaEntry::RefmvsSplatMv(x86::refmvs_avx512::splat_mv_avx512icl_fn(t))),
                // P46 Adım 2: 8 ve 16 bpc (`10/12` bit ortak) Deblocking Loop Filter (`loopfilter`,
                // 24 kayıt / 36 aday: `loop_filter_sb[0..2][0..2]` × {SSSE3, AVX2, AVX512ICL};
                // AVX512ICL `loop_filter_sb[0][0]` ve `loop_filter_sb[1][0]` `SLOW_GATHER`'da kurulmaz).
                ("ssse3", "loopfilter", field) => {
                    let (plane, dir) = lf_sb_slot(field)?;
                    let t = self.ssse3?;
                    let f = match candidate.bit_depth {
                        8 => x86::lf_ssse3::loop_filter_sb_fn(t, plane, dir)?,
                        10 => x86::lf16_ssse3::loop_filter_sb_fn::<10>(t, plane, dir)?,
                        12 => x86::lf16_ssse3::loop_filter_sb_fn::<12>(t, plane, dir)?,
                        _ => return None,
                    };
                    Some(IsaEntry::LfSb(plane, dir, f))
                }
                ("avx2", "loopfilter", field) => {
                    let (plane, dir) = lf_sb_slot(field)?;
                    let t = self.avx2?;
                    let f = match candidate.bit_depth {
                        8 => x86::lf::loop_filter_sb_fn(t, plane, dir)?,
                        10 => x86::lf16::loop_filter_sb_fn::<10>(t, plane, dir)?,
                        12 => x86::lf16::loop_filter_sb_fn::<12>(t, plane, dir)?,
                        _ => return None,
                    };
                    Some(IsaEntry::LfSb(plane, dir, f))
                }
                ("avx512icl", "loopfilter", field) => {
                    let (plane, dir) = lf_sb_slot(field)?;
                    let t = self.avx512icl?;
                    let f = match candidate.bit_depth {
                        8 => x86::lf_avx512::loop_filter_sb_fn(t, plane, dir)?,
                        10 => x86::lf16_avx512::loop_filter_sb_fn::<10>(t, plane, dir)?,
                        12 => x86::lf16_avx512::loop_filter_sb_fn::<12>(t, plane, dir)?,
                        _ => return None,
                    };
                    Some(IsaEntry::LfSb(plane, dir, f))
                }
                // P46 Adım 3: 8 ve 16 bpc (`10/12` bit Wiener, `10` bit SGR) Loop Restoration
                // (`looprestoration`, 31/31 kayıt = 38 aday: 8 bpc 2 SSE2 + 5 SSSE3 + 5 AVX2 + 5 AVX512ICL = 17;
                // 16 bpc 7 SSSE3 + 7 AVX2 + 7 AVX512ICL = 21 aday).
                ("sse2", "looprestoration", field) if candidate.bit_depth == 8 => {
                    let t = self.sse2?;
                    let tap = lr_wiener_slot(field)?;
                    Some(IsaEntry::LrWiener(
                        tap,
                        x86::lr_sse2::wiener_filter_fn(t, tap)?,
                    ))
                }
                ("ssse3", "looprestoration", field) => {
                    let t = self.ssse3?;
                    if let Some(tap) = lr_wiener_slot(field) {
                        let f = match candidate.bit_depth {
                            8 => x86::lr_ssse3::wiener_filter_fn(t, tap)?,
                            10 | 12 => x86::lr16_ssse3::wiener_16bpc_ssse3_fn(t, tap)?,
                            _ => return None,
                        };
                        Some(IsaEntry::LrWiener(tap, f))
                    } else {
                        let sgr = lr_sgr_slot(field)?;
                        let f = match candidate.bit_depth {
                            8 => x86::lr_ssse3::sgr_filter_fn(t, sgr)?,
                            10 => x86::lr16_ssse3::sgr_10bpc_ssse3_fn(t, sgr)?,
                            _ => return None,
                        };
                        Some(IsaEntry::LrSgr(sgr, f))
                    }
                }
                ("avx2", "looprestoration", field) => {
                    let t = self.avx2?;
                    if let Some(tap) = lr_wiener_slot(field) {
                        let f = match candidate.bit_depth {
                            8 => x86::lr::wiener_filter_fn(t, tap)?,
                            10 | 12 => x86::lr16::wiener_16bpc_avx2_fn(t, tap)?,
                            _ => return None,
                        };
                        Some(IsaEntry::LrWiener(tap, f))
                    } else {
                        let sgr = lr_sgr_slot(field)?;
                        let f = match candidate.bit_depth {
                            8 => x86::lr::sgr_filter_fn(t, sgr)?,
                            10 | 12 => x86::lr16::sgr_16bpc_avx2_fn(t, sgr)?,
                            _ => return None,
                        };
                        Some(IsaEntry::LrSgr(sgr, f))
                    }
                }
                ("avx512icl", "looprestoration", field) => {
                    let t = self.avx512icl?;
                    if let Some(tap) = lr_wiener_slot(field) {
                        let f = match candidate.bit_depth {
                            8 => x86::lr_avx512::wiener_filter_fn(t, tap)?,
                            10 | 12 => x86::lr16_avx512::wiener_16bpc_avx512icl_fn(t, tap)?,
                            _ => return None,
                        };
                        Some(IsaEntry::LrWiener(tap, f))
                    } else {
                        let sgr = lr_sgr_slot(field)?;
                        let f = match candidate.bit_depth {
                            8 => x86::lr_avx512::sgr_filter_fn(t, sgr)?,
                            10 => x86::lr16_avx512::sgr_10bpc_avx512icl_fn(t, sgr)?,
                            _ => return None,
                        };
                        Some(IsaEntry::LrSgr(sgr, f))
                    }
                }
                // P47: Bit derinliğinden bağımsız `msac` (`bit_depth == 0`, 8 kayıt / 8 aday:
                // 7 SSE2 + 1 AVX2 `decode_symbol_adapt16`).
                (
                    "sse2",
                    "msac",
                    "decode_symbol_adapt4"
                    | "decode_symbol_adapt8"
                    | "decode_symbol_adapt16"
                    | "decode_bool_adapt"
                    | "decode_bool_equi"
                    | "decode_bool"
                    | "decode_hi_tok",
                ) if candidate.bit_depth == 0 => self.sse2.map(|_| IsaEntry::Msac),
                ("avx2", "msac", "decode_symbol_adapt16") if candidate.bit_depth == 0 => {
                    self.avx2.map(|_| IsaEntry::Msac)
                }
                _ => None,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            arm::candidate_entry(candidate, self.neon?, self.dotprod)
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if self.wasm_simd128 {
                wasm::candidate_entry(candidate)
            } else {
                None
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        {
            let _ = candidate;
            None
        }
    }

    /// MSAC ISA düzeyi (`P47 / F4-MSAC`, `P54 / F5-MSAC`).
    #[must_use]
    pub const fn msac_isa(&self) -> crate::entropy::MsacIsa {
        #[cfg(target_arch = "x86_64")]
        {
            match (self.sse2, self.avx2) {
                (Some(t_sse2), Some(t_avx2)) => crate::entropy::MsacIsa::Avx2(t_sse2, t_avx2),
                (Some(t_sse2), None) => crate::entropy::MsacIsa::Sse2(t_sse2),
                (None, _) => crate::entropy::MsacIsa::Scalar,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match self.neon {
                Some(t_neon) => crate::entropy::MsacIsa::Neon(t_neon),
                None => crate::entropy::MsacIsa::Scalar,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            crate::entropy::MsacIsa::Scalar
        }
    }

    /// Bit derinliğinin tablosu — çözme hattı bunu **kare başında bir kez**
    /// alır (blok başına seçim yok, DOG-05).
    #[must_use]
    pub const fn for_bit_depth(&self, bit_depth: BitDepth) -> &BitDepthDsp {
        &self.tables[bd_index(bit_depth)]
    }

    /// Palet ailesi (`Dav1dPalDSPContext`, bit derinliğinden bağımsız).
    #[must_use]
    pub const fn pal(&self) -> &PalDsp {
        &self.pal
    }

    /// Refmvs ailesi (`Dav1dRefmvsDSPContext`, bit derinliğinden bağımsız).
    #[must_use]
    pub const fn refmvs(&self) -> &RefmvsDsp {
        &self.refmvs
    }

    /// Kurulumda üretilen AVX2 belirteci (`None`: AVX2 tespit edilmedi ya da
    /// maske kapattı). Belirteç yalnız [`DspDispatch::select`] ile üretilir;
    /// bu yöntem onu okur (ölçüm ve testler belirteçli girişleri buradan
    /// çağırır).
    #[must_use]
    pub const fn avx2_token(&self) -> Option<Avx2Token> {
        self.avx2
    }

    /// Kurulumda üretilen AVX512ICL belirteci.
    #[must_use]
    pub const fn avx512icl_token(&self) -> Option<Avx512IclToken> {
        self.avx512icl
    }

    /// Kurulumda üretilen SSE2 belirteci (P43).
    #[must_use]
    pub const fn sse2_token(&self) -> Option<Sse2Token> {
        self.sse2
    }

    /// Kurulumda üretilen SSSE3 belirteci (P30).
    #[must_use]
    pub const fn ssse3_token(&self) -> Option<Ssse3Token> {
        self.ssse3
    }

    /// Kurulumda üretilen SSE4.1 belirteci (P30).
    #[must_use]
    pub const fn sse41_token(&self) -> Option<Sse41Token> {
        self.sse41
    }

    /// Kurulumda üretilen `aarch64` NEON belirteci (`P51 / F5a`).
    #[must_use]
    pub fn neon_token(&self) -> Option<NeonToken> {
        #[cfg(target_arch = "aarch64")]
        {
            self.neon
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            NeonToken::new(CpuFlags::NONE)
        }
    }

    /// Kurulumda üretilen `aarch64` DOTPROD belirteci (`P51 / F5a`).
    #[must_use]
    pub fn dotprod_token(&self) -> Option<DotprodToken> {
        #[cfg(target_arch = "aarch64")]
        {
            self.dotprod
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            DotprodToken::new(CpuFlags::NONE)
        }
    }

    /// Aday yuvayı tutuyor mu (kuruldu ve üst düzey bir aday onu silmedi)?
    #[must_use]
    pub const fn is_installed(&self, candidate_index: usize) -> bool {
        candidate_index < ISA_CANDIDATES.len()
            && self.installed[candidate_index / 64] & (1u64 << (candidate_index % 64)) != 0
    }

    /// Yuva kimliği (P30): `bit_depth` tablosunun `family/field` yuvasını tutan
    /// uygulamanın adı (`"skaler"`, `"ssse3"`, `"sse4"`, `"avx2"`,
    /// `"avx512icl"`). Kurulumun seçim kaydından ([`Self::is_installed`]) okunur;
    /// işlev adresi karşılaştırılmaz (optimize derlemede aynı işlevin adresi kod
    /// birimine göre değişebilir).
    #[must_use]
    pub fn slot_implementation(
        &self,
        bit_depth: BitDepth,
        family: &str,
        field: &str,
    ) -> &'static str {
        let bd = match bit_depth {
            BitDepth::Eight => 8,
            BitDepth::Ten => 10,
            BitDepth::Twelve => 12,
        };
        ISA_CANDIDATES
            .iter()
            .enumerate()
            .find(|(i, c)| {
                c.family == family && c.field == field && c.bit_depth == bd && self.is_installed(*i)
            })
            .map_or("skaler", |(_, c)| c.implementation)
    }

    /// P45: Bit derinliğinden bağımsız (`bit_depth == 0`) yuva kimliği (`pal`, `refmvs`).
    #[must_use]
    pub fn independent_slot_implementation(&self, family: &str, field: &str) -> &'static str {
        ISA_CANDIDATES
            .iter()
            .enumerate()
            .find(|(i, c)| {
                c.family == family && c.field == field && c.bit_depth == 0 && self.is_installed(*i)
            })
            .map_or("skaler", |(_, c)| c.implementation)
    }

    /// Skaler olmayan (ISA) girişli yuva sayısı.
    #[must_use]
    pub const fn assigned_count(&self) -> usize {
        let mut n = 0;
        let mut k = 0;
        while k < KURULU_KELIME {
            n += self.installed[k].count_ones() as usize;
            k += 1;
        }
        n
    }

    /// Yuva durumları — raporlama, `docs/cpu-dispatch.json`, eşleme raporu ve
    /// test için **tek kaynak** (ADR-0003 §5.2).
    ///
    /// Sıra [`slot_keys`] ile aynıdır. Her kayıtta atanan uygulama (skaler ya
    /// da ISA adı) ve bu derlemedeki ISA adayı bulunur.
    #[must_use]
    pub fn slot_states(&self) -> Vec<SlotState> {
        slot_keys()
            .into_iter()
            .map(|key| {
                let candidates = candidate_indices(&key);
                let installed = candidates.iter().copied().find(|&i| self.is_installed(i));
                SlotState {
                    implementation: installed
                        .map_or("skaler", |i| ISA_CANDIDATES[i].implementation),
                    candidate: installed
                        .or_else(|| lowest_default(&candidates))
                        .map(|i| &ISA_CANDIDATES[i]),
                    candidates: candidates.iter().map(|&i| &ISA_CANDIDATES[i]).collect(),
                    key,
                }
            })
            .collect()
    }

    /// `flags` bir makinenin **gerçek** tespiti olsaydı [`DspDispatch::select`]'in
    /// kuracağı yuva durumları — makineden bağımsız raporlama
    /// (`docs/cpu-dispatch.json`) için. Varsayılan bir aday, gerektirdiği
    /// bayraklar `flags` içindeyse atanmış sayılır; bu makinenin tespiti
    /// `flags`'i kapsıyorsa sonuç `select(flags).slot_states()` ile aynıdır
    /// (test).
    #[must_use]
    pub fn planned_slot_states(flags: CpuFlags) -> Vec<SlotState> {
        let mut states = Self::scalar_only().slot_states();
        for state in &mut states {
            // En yüksek düzeydeki uygun varsayılan aday (adaylar düzey sırasında).
            if let Some(c) = state
                .candidates
                .iter()
                .rev()
                .find(|c| c.default && c.kurulabilir(flags))
            {
                state.implementation = c.implementation;
                state.candidate = Some(c);
            }
        }
        states
    }

    /// `aarch64` aday tablosu ([`ARM_ISA_CANDIDATES`]) üzerinden makineden
    /// bağımsız yuva durumları (`docs/cpu-dispatch-arm64.json` için; her
    /// mimaride deterministik aynı çıktıyı verir).
    #[must_use]
    pub fn planned_arm_slot_states(flags: CpuFlags) -> Vec<SlotState> {
        slot_keys()
            .into_iter()
            .map(|key| {
                let mut indices: Vec<usize> = (0..ARM_ISA_CANDIDATES.len())
                    .filter(|&i| {
                        let c = &ARM_ISA_CANDIDATES[i];
                        c.family == key.family
                            && c.field == key.field
                            && (Some(c.bit_depth) == key.bit_depth
                                || (key.bit_depth.is_none() && c.bit_depth == 0))
                    })
                    .collect();
                indices.sort_by_key(|&i| isa_rank(ARM_ISA_CANDIDATES[i].implementation));
                let candidates: Vec<&'static IsaCandidate> =
                    indices.iter().map(|&i| &ARM_ISA_CANDIDATES[i]).collect();
                let chosen = candidates
                    .iter()
                    .copied()
                    .rev()
                    .find(|c| c.default && c.kurulabilir(flags));
                let fallback = candidates
                    .iter()
                    .copied()
                    .find(|c| c.default)
                    .or_else(|| candidates.first().copied());
                SlotState {
                    implementation: chosen.map_or("skaler", |c| c.implementation),
                    candidate: chosen.or(fallback),
                    candidates,
                    key,
                }
            })
            .collect()
    }
}

impl PartialEq for DspDispatch {
    /// İki tabloyu **atama deseni** (kurulan ISA adayları) üzerinden
    /// karşılaştırır; işlev işaretçisi adresleri karşılaştırılmaz.
    fn eq(&self, other: &Self) -> bool {
        self.installed == other.installed
    }
}

impl Eq for DspDispatch {}

/// Yuvanın seçim biçimi.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Selection {
    /// Kurulumda seçilen işlev işaretçisi (tablodaki giriş).
    Runtime,
    /// Derleme zamanı seçimi; çalışma zamanı işaretçisi yok (MSAC).
    CompileTime,
}

/// Yuva anahtarı: `(aile, işlev, bit derinliği, alt-tür)`.
///
/// `field` dav1d bağlam alanının adıdır (`c->` olmadan); alt-tür dizin
/// kısmındadır (`itxfm_add[TX_4X4][DCT_DCT]`, `fb[1]`, `w_mask[2]`). `family`
/// dav1d dispatch matrisinin algoritma adıdır (`docs/dispatch-matrix.csv`
/// `algorithm`; büyütme için `resize`, MSAC için `msac`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotKey {
    /// Aile.
    pub family: &'static str,
    /// dav1d alanı (`c->` olmadan).
    pub field: String,
    /// Bit derinliği (`None`: bit derinliğinden bağımsız).
    pub bit_depth: Option<u8>,
    /// Seçim biçimi.
    pub selection: Selection,
}

impl SlotKey {
    /// `aile/alan@bd` biçiminde tekil kimlik (rapor ve eşleme dosyası).
    #[must_use]
    pub fn id(&self) -> String {
        match self.bit_depth {
            Some(bd) => format!("{}/{}@{bd}", self.family, self.field),
            None => format!("{}/{}", self.family, self.field),
        }
    }
}

/// [`DspDispatch::slot_states`] kaydı.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotState {
    /// Yuva anahtarı.
    pub key: SlotKey,
    /// Atanan uygulama: `"skaler"` ya da ISA adı (`"avx2"`).
    pub implementation: &'static str,
    /// Yuvayı tutan aday; yuva skalerse en düşük düzeydeki varsayılan aday
    /// (yoksa en düşük düzeydeki aday; `None`: bu derlemede ISA gövdesi yok).
    pub candidate: Option<&'static IsaCandidate>,
    /// Bu derlemedeki bütün ISA adayları, düzey sırasıyla (P30).
    pub candidates: Vec<&'static IsaCandidate>,
}

impl SlotState {
    /// Skaler olmayan bir gövde atanmış mı?
    #[must_use]
    pub fn is_assigned(&self) -> bool {
        self.implementation != "skaler"
    }
}

/// Bu derlemedeki ISA gövdesi adayı (yuva başına).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsaCandidate {
    /// Aile ([`SlotKey::family`]).
    pub family: &'static str,
    /// dav1d alanı ([`SlotKey::field`]).
    pub field: &'static str,
    /// Gövdenin bit derinliği.
    pub bit_depth: u8,
    /// Uygulama adı (`"avx2"`); dispatch matrisinin `tier` sütunuyla aynı.
    pub implementation: &'static str,
    /// [`DspDispatch::select`]'in bu adayı atamak için istediği bayraklar.
    pub requires: CpuFlags,
    /// Varsayılan atama (ölçüme bağlı karar, "ölç, inanma").
    pub default: bool,
    /// Karar ve dayanağı.
    pub decision: &'static str,
}

impl IsaCandidate {
    /// P36: `SLOW_GATHER` bayraklı işlemcide kurulmayan aday (dav1d
    /// `mc_init_x86`: AVX512ICL `warp8x8`, `warp8x8t`, `resize`).
    ///
    /// P39: AVX2 `fgy_32x32xn` ve `fguv_32x32xn[*]` (dav1d
    /// `film_grain_dsp_init_x86`: `generate_grain_*` her zaman, `fgy`/`fguv`
    /// yalnız `!SLOW_GATHER`).
    ///
    /// P41 / P42: 8 bpc AVX512ICL `fgy_32x32xn` ve `fguv_32x32xn[*]` `vpermb`
    /// kullandığı için `SLOW_GATHER`'da da kurulur; 16 bpc (`10`/`12` bpc)
    /// AVX512ICL `fgy`/`fguv` ise `vpgatherdd` kullandığından `SLOW_GATHER`'da
    /// kurulmaz (`filmgrain.h:76`:
    /// `BITDEPTH == 8 || !(flags & DAV1D_X86_CPU_FLAG_SLOW_GATHER)`).
    ///
    /// P46: AVX512ICL `loopfilter` `loop_filter_sb[0][0]` ve `loop_filter_sb[1][0]`
    /// (dikey kenarlar / yatay süzme `lpf_h_sb_y`, `lpf_h_sb_uv`) 8 ve 16 bpc'de
    /// `SLOW_GATHER`'da kurulmaz (`src/x86/loopfilter.h:64-67`).
    #[must_use]
    pub fn yavas_toplamada_kurulmaz(&self) -> bool {
        (self.implementation == "avx512icl"
            && matches!(
                (self.family, self.field),
                ("mc", "warp8x8") | ("mc", "warp8x8t") | ("resize", "resize")
            ))
            || (self.implementation == "avx2"
                && self.family == "filmgrain"
                && (self.field == "fgy_32x32xn" || self.field.starts_with("fguv_32x32xn[")))
            || (self.implementation == "avx512icl"
                && self.family == "filmgrain"
                && self.bit_depth != 8
                && (self.field == "fgy_32x32xn" || self.field.starts_with("fguv_32x32xn[")))
            || (self.implementation == "avx512icl"
                && self.family == "loopfilter"
                && (self.field == "loop_filter_sb[0][0]" || self.field == "loop_filter_sb[1][0]"))
    }

    /// Aday `flags` ile kurulabilir mi: istenen bayraklar var ve
    /// [`Self::yavas_toplamada_kurulmaz`] engeli yok.
    #[must_use]
    pub fn kurulabilir(&self, flags: CpuFlags) -> bool {
        flags.contains(self.requires)
            && !(flags.contains(CpuFlags::SLOW_GATHER) && self.yavas_toplamada_kurulmaz())
    }
}

/// P43 Adım 2: 8 bpc WHT-dışı SSSE3 ITX aday kurucusu (`X86_ISA_CANDIDATES` derleme zamanı sabiti).
const fn itx8_ssse3_aday(field: &'static str, decision: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "itx",
        field,
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision,
    }
}

/// P43 Adım 2: 8 bpc WHT-dışı AVX2 ITX aday kurucusu (`X86_ISA_CANDIDATES` derleme zamanı sabiti).
const fn itx8_avx2_aday(field: &'static str, decision: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "itx",
        field,
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision,
    }
}

/// P43 Adım 3: 8 bpc WHT-dışı AVX512ICL ITX aday kurucusu (`X86_ISA_CANDIDATES` derleme zamanı sabiti;
/// dav1d `src/x86/itx.h:221-244` gibi `SLOW_GATHER`'da da kurulur).
const fn itx8_avx512_aday(field: &'static str, decision: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "itx",
        field,
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision,
    }
}

/// P43 Adım 3: 10 bpc WHT-dışı SSE4.1 ITX aday kurucusu (`X86_ISA_CANDIDATES` derleme zamanı sabiti).
const fn itx10_sse4_aday(field: &'static str, decision: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "itx",
        field,
        bit_depth: 10,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: true,
        decision,
    }
}

/// P44 Adım 1: 10 bpc WHT-dışı AVX2 ITX aday kurucusu (`X86_ISA_CANDIDATES` derleme zamanı sabiti).
const fn itx10_avx2_aday(field: &'static str, decision: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "itx",
        field,
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision,
    }
}

/// P44 Adım 1 (Paket B): 12 bpc WHT-dışı AVX2 ITX aday kurucusu (`X86_ISA_CANDIDATES` derleme zamanı sabiti).
const fn itx12_avx2_aday(field: &'static str, decision: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "itx",
        field,
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision,
    }
}

/// P44 Adım 2: 10 bpc WHT-dışı (`w >= 8 && h >= 8`, 75 kayıt) AVX512ICL ITX aday kurucusu.
const fn itx10_avx512_aday(field: &'static str, decision: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "itx",
        field,
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision,
    }
}

const ITX8_SSSE3_KARAR_KUCUK: &str = "P43: itx-bench 8 bpc w, h <= 16 59 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,02–0,25×, Rust / dav1d SSSE3 eşli oran 0,65–2,11× (ortanca 1,29×) / 0,65–2,19× (ortanca 1,27×; 58/59 hücre ≤ 2,00×); 16 bit pmaddwd/pmulhrsw 1B XMM çekirdekler + DC-only hızlı yolu (rust_method: core::arch)";
const ITX8_SSSE3_KARAR_BUYUK: &str = "P43: itx-bench 8 bpc max(w, h) in {32, 64} 35 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,04–0,12×, Rust / dav1d SSSE3 eşli oran 0,56–1,69× (ortanca 1,22×) / 0,55–1,67× (ortanca 1,21×; 35/35 hücre ≤ 2,00×); DC-only + dav1d eob çeyrek/yarı-blok hızlı yolları + 16 bit XMM 1B çekirdekler (rust_method: core::arch)";
const ITX8_AVX2_KARAR_KUCUK: &str = "P43: itx-bench 8 bpc w, h <= 16 59 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,01–0,24×, Rust / dav1d AVX2 eşli oran 0,94–2,26× (ortanca 1,57×) / 0,90–2,34× (ortanca 1,58×); Split-YMM/YMM 16 bit vpmaddwd/vpmulhrsw 1B çekirdekler + DC-only hızlı yolu (rust_method: core::arch)";
const ITX8_AVX2_KARAR_BUYUK: &str = "P43: itx-bench 8 bpc max(w, h) in {32, 64} 35 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,02–0,11×, Rust / dav1d AVX2 eşli oran 0,86–2,46× (ortanca 1,46×) / 0,86–2,44× (ortanca 1,45×); DC-only + dav1d eob sekizde/çeyrek/yarı-blok hızlı yolları + Split-YMM/YMM 1B çekirdekler (rust_method: core::arch)";
const ITX8_AVX512_KARAR_KUCUK: &str = "P43: itx-bench 8 bpc w, h <= 16 59 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,01–0,26× (Rust AVX512ICL / AVX2 0,79–1,85× / 0,78–1,85×), Rust / dav1d AVX512ICL eşli oran 0,62–2,81× (ortanca 1,45×, 33/59 ≤ 1,50×, 52/59 ≤ 2,00×) / 0,67–2,80× (ortanca 1,43×); SLOW_GATHER'da da kurulur (0 gather/scatter), Split-ZMM/AVX512VL + vpdpwssd (VNNI) 1B çekirdekler + DC-only hızlı yolu (rust_method: core::arch)";
const ITX8_AVX512_KARAR_BUYUK: &str = "P43: itx-bench 8 bpc max(w, h) in {32, 64} 35 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,02–0,10× (Rust AVX512ICL / AVX2 0,69–1,19× / 0,67–1,18×), Rust / dav1d AVX512ICL eşli oran 0,59–2,50× (ortanca 1,49×, 18/35 ≤ 1,50×, 28/35 ≤ 2,00×) / 0,58–2,52× (ortanca 1,52×); SLOW_GATHER'da da kurulur (0 gather/scatter), 512 bit ZMM (32×i16) + vpdpwssd (VNNI) 1B çekirdekler + DC-only ve dav1d eob hızlı yolları (rust_method: core::arch)";
const ITX10_SSE4_KARAR_KUCUK: &str = "P43: itx16-bench 10 bpc w, h <= 16 59 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSE4.1 / skaler 0,02–0,24× / 0,02–0,25×, Rust / dav1d SSE4.1 eşli oran 0,90–2,18× (ortanca 1,24×, 58/59 ≤ 2,00×) / 0,91–2,20× (ortanca 1,24×, 58/59 ≤ 2,00×); dav1d itx.h:178 paritesiyle yalnız 10 bpc (12 bpc skalerde kalır), Pass 1 32 bit pmulld/pmaxsd/pminsd + packssdw col_clip + Pass 2 16 bit pmaddwd/pmulhrsw XMM 1B çekirdekler + DC-only hızlı yolu (rust_method: core::arch)";
const ITX10_SSE4_KARAR_BUYUK: &str = "P43: itx16-bench 10 bpc max(w, h) in {32, 64} 35 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSE4.1 / skaler 0,05–0,13×, Rust / dav1d SSE4.1 eşli oran 1,01–2,31× (ortanca 1,38×, 33/35 ≤ 2,00×) / 1,00–2,31× (ortanca 1,38×, 33/35 ≤ 2,00×); dav1d itx.h:178 paritesiyle yalnız 10 bpc (12 bpc skalerde kalır), DC-only + dav1d eob çeyrek/yarı-blok hızlı yolları + Pass 1 32 bit pmulld + packssdw + Pass 2 16 bit XMM 1B çekirdekler (rust_method: core::arch)";
const ITX10_AVX2_KARAR_KUCUK: &str = "P44: itx16-bench 10 bpc w, h <= 16 59 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,02–0,22× / 0,02–0,23×, Rust / dav1d AVX2 eşli oran 0,93–3,09× (ortanca 1,63×, 20/59 ≤ 1,50×, 47/59 ≤ 2,00×) / 0,98–3,09× (ortanca 1,63×, 20/59 ≤ 1,50×, 49/59 ≤ 2,00×); Pass 1 32 bit YMM/XMM vpmulld + clip_18b + vpackssdw col_clip + Pass 2 16 bit YMM/XMM vpmaddwd/vpmulhrsw 1B çekirdekler + DC-only hızlı yolu (rust_method: core::arch)";
const ITX10_AVX2_KARAR_BUYUK: &str = "P44: itx16-bench 10 bpc max(w, h) in {32, 64} 35 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,03–0,09×, Rust / dav1d AVX2 eşli oran 0,98–2,90× (ortanca 1,37×, 25/35 ≤ 1,50×, 29/35 ≤ 2,00×) / 0,99–2,91× (ortanca 1,40×, 23/35 ≤ 1,50×, 29/35 ≤ 2,00×); Pass 1 32 bit YMM vpmulld + clip_18b + vpackssdw col_clip + Pass 2 16 bit YMM/Split-YMM 1B çekirdekler + DC-only ve dav1d eob çeyrek/yarı-blok hızlı yolları (rust_method: core::arch)";

const ITX12_AVX2_KARAR_KUCUK: &str = "P44: itx16-bench 12 bpc w, h <= 16 59 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,03–0,23× (ortanca 0,09×), Rust / dav1d AVX2 eşli oran 0,86–5,44× (ortanca 2,44×, 13/59 ≤ 2,00×) / 0,93–5,51× (ortanca 2,42×, 13/59 ≤ 2,00×); iki geçişli 32 bit i32 (Pass 1 clip_20b 64 bit vpmuldq + Pass 2 clip_18b 32 bit vpmulld) XMM/YMM 1B çekirdekler + DC-only ve yarı-sıfır hızlı yolları (rust_method: core::arch)";
const ITX12_AVX2_KARAR_BUYUK: &str = "P44: itx16-bench 12 bpc max(w, h) == 32 11 hücre / 7 kayıt (RTX_8X32/RTX_32X8 DCT_DCT+IDTX ve RTX_16X32/RTX_32X16/TX_32X32 IDTX; dav1d src/x86/itx.h:249-263 paritesiyle en büyük 8 DCT_DCT 12 bpc'de skalerde kalır) (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,05–0,15×, Rust / dav1d AVX2 eşli oran 1,03–7,01× (ortanca 2,16×) / 1,02–7,10× (ortanca 2,16×); iki geçişli 32 bit i32 (Pass 1 clip_20b + Pass 2 clip_18b vpmulld) YMM çekirdekler + DC-only ve eob hızlı yolları (rust_method: core::arch)";
const ITX10_AVX512_KARAR_KUCUK: &str = "P44: itx16-bench 10 bpc 8 <= w, h <= 16 27 hücre / 60 kayıt (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / dav1d-c 0,05–0,35× (ortanca 0,09×, Rust AVX512ICL / AVX2 0,81–1,09×, ortanca 0,96×), Rust / dav1d AVX512ICL eşli oran 0,95–2,41× (ortanca 1,36×, 16/27 ≤ 1,50×, 24/27 ≤ 2,00×) / 0,97–2,36× (ortanca 1,38×, 16/27 ≤ 1,50×, 24/27 ≤ 2,00×); dav1d src/x86/itx.h:289-295 paritesiyle yalnız 10 bpc ve w, h >= 8; SLOW_GATHER'da da kurulur (0 gather/scatter), Pass 1 32 bit clip_18b + vpackssdw col_clip + Pass 2 16 bit vpdpwssd (VNNI) / vpmulhrsw çekirdekler + DC-only hızlı yolu (rust_method: core::arch)";
const ITX10_AVX512_KARAR_BUYUK: &str = "P44: itx16-bench 10 bpc max(w, h) in {32, 64} 35 hücre / 15 kayıt (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / dav1d-c 0,04–0,23× (ortanca 0,07×, Rust AVX512ICL / AVX2 0,81–1,13× / 0,83–1,14×, ortanca 0,97× / 0,98×), Rust / dav1d AVX512ICL eşli oran 0,92–3,19× (ortanca 1,51×, 17/35 ≤ 1,50×, 28/35 ≤ 2,00×) / 0,96–3,22× (ortanca 1,51×, 17/35 ≤ 1,50×, 28/35 ≤ 2,00×); dav1d src/x86/itx.h:292-304 paritesiyle yalnız 10 bpc; SLOW_GATHER'da da kurulur (0 gather/scatter), Pass 1 32 bit YMM clip_18b + vpackssdw col_clip + Pass 2 512 bit ZMM (z32) / YMM (x16) vpdpwssd (VNNI) 1B çekirdekler + DC-only ve dav1d eob hızlı yolları (rust_method: core::arch)";

const IPRED8_SSSE3_KARAR: &str = "varsayılan (P45 Adım 2, 8 bpc IPRED 22/22 SSSE3 — D01592..D01700): 128-bit SSSE3 `psadbw`/`pshufb`/`pmaddubsw`/`pmulhrsw`/`pavgb`/`psignw` (`src/x86/ipred_ssse3.asm`, 0 SSE4.1+/VEX/EVEX); iki CCD'de (çekirdek 6/14, 142 hücre) Rust SSSE3 / skaler 0,07–1,71× / 0,07–1,73× (ortanca 0,27× / 0,27×), Rust / dav1d SSSE3 eşli oran 0,95–8,33× (ortanca 2,31×) / 0,95–8,53× (ortanca 2,33×) (rust_method: core::arch)";
const IPRED8_AVX2_KARAR: &str = "varsayılan (P45 Adım 2, 8 bpc IPRED 22/22 AVX2 — D01590..D01696): 128/256-bit AVX2 `vpsadbw`/`vpshufb`/`vpmaddubsw`/`vpmulhrsw`/`vpavgb`/`vpsignw` (`src/x86/ipred.asm`, 0 EVEX); iki CCD'de (çekirdek 6/14, 142 hücre) Rust AVX2 / skaler 0,04–1,72× / 0,04–1,73× (ortanca 0,25× / 0,25×), Rust / dav1d AVX2 eşli oran 1,16–11,86× (ortanca 2,52×) / 1,17–12,57× (ortanca 2,55×) (rust_method: core::arch)";
const IPRED8_AVX512_KARAR: &str = "varsayılan (P45 Adım 2, 8 bpc IPRED 15/15 AVX512ICL — D01620..D01698): dav1d `src/x86/ipred.h:133-151` kapsam eşitliğiyle yalnız `intra_pred[14]` + `pal_pred` (`cfl_ac[3]` ve `cfl_pred[4]` 8-bpc'de `avx2`'de kalır; `SLOW_GATHER` engeli yok, 0 gather/scatter); iki CCD'de (çekirdek 6/14, 110 hücre) Rust AVX512ICL / skaler 0,04–1,58× / 0,04–1,62× (ortanca 0,22× / 0,22×), Rust AVX512ICL / AVX2 0,65–1,46× / 0,65–1,43× (ortanca 0,91× / 0,90×), Rust / dav1d AVX512ICL eşli oran ortanca 2,22× / 2,25× (rust_method: core::arch)";
const IPRED16_SSSE3_KARAR: &str = "varsayılan (P45 Adım 2, 16 bpc SSSE3 IPRED 22 kayıt / 44 yuva @10+@12 — D01593..D01701): 128-bit SSSE3 (`src/x86/ipred16_ssse3.asm`, 0 SSE4.1+/VEX/EVEX); iki CCD'de (çekirdek 6/14, 284 hücre) Rust SSSE3 / skaler 0,05–1,06× (ortanca 0,30× / 0,30×), Rust / dav1d SSSE3 eşli oran 0,49–10,93× (ortanca 2,95×) / 0,50–11,01× (ortanca 3,03×) (rust_method: core::arch)";
const IPRED16_AVX2_KARAR: &str = "varsayılan (P45 Adım 2, 16 bpc AVX2 IPRED 22 kayıt / 44 yuva @10+@12 — D01591..D01697): 128/256-bit AVX2 (`src/x86/ipred16.asm`, 0 EVEX); iki CCD'de (çekirdek 6/14, 284 hücre) Rust AVX2 / skaler 0,03–1,06× / 0,03–1,07× (ortanca 0,26× / 0,26×), Rust / dav1d AVX2 eşli oran 0,64–9,70× (ortanca 2,92×) / 0,62–9,83× (ortanca 2,91×) (rust_method: core::arch)";
const IPRED16_AVX512_KARAR: &str = "varsayılan (P45 Adım 2, 16 bpc AVX512ICL IPRED 9 kayıt / 18 yuva @10+@12 — D01631..D01699): dav1d `src/x86/ipred.h:142-151` kapsam eşitliğiyle yalnız Paeth, Smooth, SmoothH, SmoothV, Z1, Z2, Z3, Filter + pal_pred (`src/x86/ipred16_avx512.asm`; kalan 13 yuva 16-bpc'de `avx2`'de kalır; `SLOW_GATHER` engeli yok, 0 gather/scatter); iki CCD'de (çekirdek 6/14, 136 hücre) Rust AVX512ICL / skaler 0,02–1,24× / 0,02–1,27× (ortanca 0,28× / 0,28×), Rust AVX512ICL / AVX2 0,49–1,20× / 0,50–1,23× (ortanca 0,97× / 0,97×), Rust / dav1d AVX512ICL eşli oran ortanca 3,23× / 3,12× (rust_method: core::arch)";

const LF8_SSSE3_KARAR: &str = "varsayılan (P46 Adım 2, 8 bpc LF 4/4 SSSE3 — D02707/D02713/D02719/D02725): 128-bit SSSE3 `pabsw`/`pcmpgtw`/`pmaxsw`/`pminsw`/`packuswb` (`src/x86/loopfilter_sse.asm`, 0 SSE4.1+/VEX/EVEX); iki CCD'de (çekirdek 6/14, 28 hücre) Rust SSSE3 / skaler 0,75–0,95× / 0,73–0,94× (ortanca 0,84× / 0,83×), Rust / dav1d SSSE3 eşli oran 1,97–8,11× (ortanca 4,16×) / 1,92–8,09× (ortanca 4,00×) (rust_method: core::arch)";
const LF8_AVX2_KARAR: &str = "varsayılan (P46 Adım 2, 8 bpc LF 4/4 AVX2 — D02703/D02709/D02715/D02721): 256-bit AVX2 `vpabsw`/`vpcmpgtw`/`vpmaxsw`/`vpminsw`/`vpackuswb` (`src/x86/loopfilter_avx2.asm`, 0 EVEX); iki CCD'de (çekirdek 6/14, 28 hücre) Rust AVX2 / skaler 0,59–0,84× / 0,60–0,85× (ortanca 0,73× / 0,72×), Rust / dav1d AVX2 eşli oran 3,37–12,97× (ortanca 6,01×) / 3,32–12,70× (ortanca 6,00×) (rust_method: core::arch)";
const LF8_AVX512_KARAR: &str = "varsayılan (P46 Adım 2, 8 bpc LF 4/4 AVX512ICL — D02705/D02711/D02717/D02723): 512-bit AVX512ICL `vpabsw`/`vpcmpgtw`/`vpmaxsw`/`vpminsw`/`vpmovuswb` (`src/x86/loopfilter_avx512.asm`; 0 gather/scatter; `src/x86/loopfilter.h:61-65` paritesiyle `loop_filter_sb[0/1][0]` `SLOW_GATHER`'da kurulmaz); iki CCD'de (çekirdek 6/14, 28 hücre) Rust AVX512ICL / skaler 0,62–1,04× / 0,61–1,22× (ortanca 0,77× / 0,77×), Rust AVX512ICL / AVX2 `v_*` `mix`/`wd16` hücrelerinde 0,93–0,98× / 0,93–0,98×, Rust / dav1d AVX512ICL eşli oran 1,92–19,84× (ortanca 4,69×) / 2,05–20,44× (ortanca 4,73×) (rust_method: core::arch)";
const LF16_SSSE3_KARAR: &str = "varsayılan (P46 Adım 2, 16 bpc SSSE3 LF 4 kayıt / 8 yuva @10+@12 — D02708/D02714/D02720/D02726): 128-bit SSSE3 işaretli `i16`/`u16` (`src/x86/loopfilter16_sse.asm`, 0 SSE4.1+/VEX/EVEX); iki CCD'de (çekirdek 6/14, 56 hücre) Rust SSSE3 / skaler 0,35–0,70× (ortanca 0,45× / 0,44×), Rust / dav1d SSSE3 eşli oran 1,40–3,01× (ortanca 2,11×) / 1,38–2,96× (ortanca 2,10×) (rust_method: core::arch)";
const LF16_AVX2_KARAR: &str = "varsayılan (P46 Adım 2, 16 bpc AVX2 LF 4 kayıt / 8 yuva @10+@12 — D02704/D02710/D02716/D02722): 128/256-bit AVX2 (`src/x86/loopfilter16_avx2.asm`, 0 EVEX); iki CCD'de (çekirdek 6/14, 56 hücre) Rust AVX2 / skaler 0,24–0,77× / 0,23–0,63× (ortanca 0,35× / 0,34×), Rust / dav1d AVX2 eşli oran 1,84–5,75× (ortanca 3,13×) / 1,80–4,40× (ortanca 3,10×) (rust_method: core::arch)";
const LF16_AVX512_KARAR: &str = "varsayılan (P46 Adım 2, 16 bpc AVX512ICL LF 4 kayıt / 8 yuva @10+@12 — D02706/D02712/D02718/D02724): 128/256/512-bit AVX512ICL (`src/x86/loopfilter16_avx512.asm`; 0 gather/scatter; `src/x86/loopfilter.h:61-67` paritesiyle `loop_filter_sb[0/1][0]` `SLOW_GATHER`'da kurulmaz); iki CCD'de (çekirdek 6/14, 56 hücre) Rust AVX512ICL / skaler 0,14–0,58× (ortanca 0,33× / 0,33×), Rust AVX512ICL / AVX2 0,35–1,05× / 0,46–1,10× (ortanca 0,96× / 0,98×), Rust / dav1d AVX512ICL eşli oran 1,35–4,67× (ortanca 2,90×) / 1,31–4,66× (ortanca 2,92×) (rust_method: core::arch)";

const LR8_SSE2_KARAR: &str = "varsayılan (P46 Adım 3, 8 bpc LR Wiener 2/2 SSE2 — D02749/D02755): 128-bit SSE2 `pmaddwd`/`psrad`/`packssdw`/`packuswb` (`src/x86/looprestoration_sse.asm:257-579`, 0 SSE3+/VEX/EVEX); iki CCD'de (çekirdek 6/14, 12 hücre) Rust SSE2 / skaler 0,30–0,41× / 0,30–0,41× (ortanca 0,35× / 0,36×), Rust / dav1d SSE2 eşli oran 1,87–2,87× (ortanca 2,07×) / 1,82–2,85× (ortanca 2,11×) (rust_method: core::arch)";
const LR8_SSSE3_KARAR: &str = "varsayılan (P46 Adım 3, 8 bpc LR 5/5 SSSE3 — D02731/D02737/D02743/D02750/D02756): 128-bit SSSE3 `pmaddwd`/`psrad`/`packssdw`/`packuswb` Wiener 7/5 + SGRPROJ 5×5 / 3×3 / mix (`src/x86/looprestoration_sse.asm`, 0 SSE4.1+/VEX/EVEX); iki CCD'de (çekirdek 6/14, 30 hücre) Rust SSSE3 / skaler 0,29–0,85× / 0,30–0,86× (ortanca 0,60× / 0,60×), Rust / dav1d SSSE3 eşli oran 2,25–3,78× (ortanca 2,48×) / 2,26–4,09× (ortanca 2,48×) (rust_method: core::arch)";
const LR8_AVX2_KARAR: &str = "varsayılan (P46 Adım 3, 8 bpc LR 5/5 AVX2 — D02727/D02733/D02739/D02745/D02752): 256-bit AVX2 `vpmaddwd`/`vpmulld`/`vpackssdw`/`vpackuswb` Wiener 7/5 + SGRPROJ 5×5 / 3×3 / mix (`src/x86/looprestoration.asm`, 0 EVEX); iki CCD'de (çekirdek 6/14, 30 hücre) Rust AVX2 / skaler 0,17–0,44× / 0,17–0,42× (ortanca 0,30× / 0,30×), Rust / dav1d AVX2 eşli oran 2,18–4,56× (ortanca 2,66×) / 2,12–4,56× (ortanca 2,59×) (rust_method: core::arch)";
const LR8_AVX512_KARAR: &str = "varsayılan (P46 Adım 3, 8 bpc LR 4/4 kayıt + wiener[1]@8 eş-bağlama = 5/5 AVX512ICL — D02729/D02735/D02741/D02747): 512-bit AVX512ICL `vpmaddwd`/`vpmulld`/`vpermi2b` (`src/x86/looprestoration_avx512.asm`, `src/x86/looprestoration.h:81-86`; `SLOW_GATHER` engeli yok, 0 gather/scatter); iki CCD'de (çekirdek 6/14, 30 hücre) Rust AVX512ICL / skaler 0,18–0,52× / 0,18–0,53× (ortanca 0,34× / 0,33×), Rust AVX512ICL / AVX2 wiener7 w>=128 hücrelerinde 0,87–0,94× (0,89–1,23× / 0,87–1,32×), Rust / dav1d AVX512ICL eşli oran ortanca 3,28× / 3,36× (rust_method: core::arch)";
const LR16_SSSE3_KARAR: &str = "varsayılan (P46 Adım 3, 16 bpc SSSE3 LR 5 kayıt / 7 yuva: Wiener @10+@12 + SGR @10 — D02732/D02738/D02744/D02751/D02757): 128-bit SSSE3 (`src/x86/looprestoration16_sse.asm`, `src/x86/looprestoration.h:62`: 12 bpc SGR skalerde kalır; 0 SSE4.1+/VEX/EVEX); iki CCD'de (çekirdek 6/14, 42 hücre) Rust SSSE3 / skaler 0,15–0,68× / 0,16–0,69× (ortanca 0,21× / 0,21×), Rust / dav1d SSSE3 eşli oran 2,07–3,29× (ortanca 2,77×) / 2,10–3,34× (ortanca 2,77×) (rust_method: core::arch)";
const LR16_AVX2_KARAR: &str = "varsayılan (P46 Adım 3, 16 bpc AVX2 LR 5 kayıt / 7 yuva: Wiener @10+@12 + SGR @10 — D02728/D02734/D02740/D02746/D02753): 256-bit AVX2 (`src/x86/looprestoration16.asm`, `src/x86/looprestoration.h:73`: 12 bpc SGR skalerde kalır; 0 EVEX); iki CCD'de (çekirdek 6/14, 42 hücre) Rust AVX2 / skaler 0,06–0,31× / 0,05–0,30× (ortanca 0,10× / 0,10×), Rust / dav1d AVX2 eşli oran 1,74–3,05× (ortanca 2,35×) / 1,71–2,91× (ortanca 2,34×) (rust_method: core::arch)";
const LR16_AVX512_KARAR: &str = "varsayılan (P46 Adım 3, 16 bpc AVX512ICL LR 5 kayıt / 7 yuva: Wiener @10+@12 + SGR @10 — D02730/D02736/D02742/D02748/D02754): 512-bit AVX512ICL `vpmaddwd`/`vpmulld` (`src/x86/looprestoration16_avx512.asm`, `src/x86/looprestoration.h:88`: 12 bpc SGR skalerde kalır; `SLOW_GATHER` engeli yok, 0 gather/scatter); iki CCD'de (çekirdek 6/14, 42 hücre) Rust AVX512ICL / skaler 0,04–0,51× / 0,04–0,51× (ortanca 0,15× / 0,16×), Rust AVX512ICL / AVX2 wiener w>=128 hücrelerinde 0,78–0,88× (0,79–1,74× / 0,78–1,72×), Rust / dav1d AVX512ICL eşli oran ortanca 3,68× / 3,55× (rust_method: core::arch)";

const MSAC_SSE2_KARAR: &str = "varsayılan (P47 Adım 1, F4-MSAC 7/7 SSE2 — x86-msac-0001..0004, 0006..0008): 128-bit SSE2 `_mm_mulhi_epu16`/`_mm_subs_epu16`/`_mm_cmpeq_epi16`/`_mm_movemask_epi8` + paralel `_mm_avg_epu16`/`_mm_sra_epi16` CDF güncellemesi ve aşırı okumasız hızlı 64-bit kelime `ctx_refill_fast` (`src/x86/msac.asm`, `src/x86/msac.h:47-64`; 0 SSSE3+/VEX/EVEX) (rust_method: core::arch)";
const MSAC_AVX2_KARAR: &str = "varsayılan (P47 Adım 1, F4-MSAC 1/1 AVX2 — x86-msac-0005 `decode_symbol_adapt16_avx2`): 256-bit AVX2 `_mm256_mulhi_epu16`/`_mm256_max_epu16`/`_mm256_cmpeq_epi16`/`_mm256_movemask_epi8` + paralel `_mm256_avg_epu16`/`_mm256_sra_epi16` 16-şeritli CDF güncellemesi (`src/x86/msac.asm:616-671`, `src/x86/msac.h:66-68`; 0 EVEX) (rust_method: core::arch)";

const fn msac_aday(
    implementation: &'static str,
    requires: CpuFlags,
    field: &'static str,
    decision: &'static str,
) -> IsaCandidate {
    IsaCandidate {
        family: "msac",
        field,
        bit_depth: 0,
        implementation,
        requires,
        default: true,
        decision,
    }
}

const fn lr_aday(
    bit_depth: u8,
    implementation: &'static str,
    requires: CpuFlags,
    field: &'static str,
    decision: &'static str,
) -> IsaCandidate {
    IsaCandidate {
        family: "looprestoration",
        field,
        bit_depth,
        implementation,
        requires,
        default: true,
        decision,
    }
}

const fn lf_aday(
    bit_depth: u8,
    implementation: &'static str,
    requires: CpuFlags,
    field: &'static str,
    decision: &'static str,
) -> IsaCandidate {
    IsaCandidate {
        family: "loopfilter",
        field,
        bit_depth,
        implementation,
        requires,
        default: true,
        decision,
    }
}

const fn ipred8_ssse3_aday(field: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "ipred",
        field,
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: IPRED8_SSSE3_KARAR,
    }
}

const fn ipred8_avx2_aday(field: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "ipred",
        field,
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: IPRED8_AVX2_KARAR,
    }
}

const fn ipred8_avx512_aday(field: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "ipred",
        field,
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: IPRED8_AVX512_KARAR,
    }
}

const fn ipred16_ssse3_aday(bit_depth: u8, field: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "ipred",
        field,
        bit_depth,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: IPRED16_SSSE3_KARAR,
    }
}

const fn ipred16_avx2_aday(bit_depth: u8, field: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "ipred",
        field,
        bit_depth,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: IPRED16_AVX2_KARAR,
    }
}

const fn ipred16_avx512_aday(bit_depth: u8, field: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "ipred",
        field,
        bit_depth,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: IPRED16_AVX512_KARAR,
    }
}

/// x86_64 ISA adayları — veri tablosu (gövde değil; her hedefte derlenir).
///
/// Her aday dispatch matrisinde aynı alan, ISA ve bit derinliğinde bir kayda
/// sahip olmalıdır (eşleme testi). Ölçmeden varsayılan seçilmez (AVX512ICL
/// adayları: CDEF P30, MC P36). Eşleme dosyası (`docs/dispatch-slot-map.csv`) bu
/// tablodan üretilir, böylece denetimi hedeften bağımsızdır (P29); çalışma
/// zamanı seçimi [`ISA_CANDIDATES`]'ı kullanır.
pub const X86_ISA_CANDIDATES: &[IsaCandidate] = &[
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P28: cdef-bench (aynı girdi, 7 tur) dir AVX2 11,78 ns, skaler 74,60 ns, dav1d AVX2 8,93 ns (1,32×); çözme hattında",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P28: cdef-bench filter 8x8 AVX2 53,89 ns, skaler 4296,51 ns, dav1d AVX2 39,73 ns (1,36×)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P28: cdef-bench filter 4x8 AVX2 37,37 ns, skaler 2203,59 ns, dav1d AVX2 26,62 ns (1,40×)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P28: cdef-bench filter 4x4 AVX2 25,95 ns, skaler 1109,11 ns, dav1d AVX2 18,55 ns (1,40×)",
    },
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P29: cdef-bench (aynı girdi, 7 tur) 10 bit dir AVX2 12,55 ns, skaler 86,69 ns, dav1d AVX2 8,92 ns (1,41×); çözme hattı f4-cdef-16bpc-avx2.md §4",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P29: cdef-bench (aynı girdi, 7 tur) 10 bit filter 8x8 AVX2 64,93 ns, skaler 4243,96 ns, dav1d AVX2 39,89 ns (1,63×); çözme hattı f4-cdef-16bpc-avx2.md §4",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P29: cdef-bench (aynı girdi, 7 tur) 10 bit filter 4x8 AVX2 45,10 ns, skaler 2188,84 ns, dav1d AVX2 29,16 ns (1,55×); çözme hattı f4-cdef-16bpc-avx2.md §4",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P29: cdef-bench (aynı girdi, 7 tur) 10 bit filter 4x4 AVX2 31,85 ns, skaler 1110,61 ns, dav1d AVX2 20,60 ns (1,55×); çözme hattı f4-cdef-16bpc-avx2.md §4",
    },
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P29: cdef-bench (aynı girdi, 7 tur) 12 bit dir AVX2 12,57 ns, skaler 86,88 ns, dav1d AVX2 8,90 ns (1,41×); çözme hattı f4-cdef-16bpc-avx2.md §4",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P29: cdef-bench (aynı girdi, 7 tur) 12 bit filter 8x8 AVX2 64,22 ns, skaler 4384,56 ns, dav1d AVX2 39,90 ns (1,61×); çözme hattı f4-cdef-16bpc-avx2.md §4",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P29: cdef-bench (aynı girdi, 7 tur) 12 bit filter 4x8 AVX2 45,25 ns, skaler 2260,26 ns, dav1d AVX2 29,14 ns (1,55×); çözme hattı f4-cdef-16bpc-avx2.md §4",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P29: cdef-bench (aynı girdi, 7 tur) 12 bit filter 4x4 AVX2 31,85 ns, skaler 1145,14 ns, dav1d AVX2 20,68 ns (1,54×); çözme hattı f4-cdef-16bpc-avx2.md §4",
    },
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit dir 8x8: SSSE3 22,21/21,57 ns, skaler 74,68/73,21 ns, dav1d-ssse3 16,31/15,96 ns (1,36×/1,35×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 8x8: SSSE3 85,21/82,42 ns, skaler 4327,83/4222,71 ns, dav1d-ssse3 62,95/61,53 ns (1,35×/1,34×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 4x8: SSSE3 55,64/53,63 ns, skaler 2218,47/2166,11 ns, dav1d-ssse3 40,03/39,19 ns (1,39×/1,37×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 4x4: SSSE3 35,75/34,40 ns, skaler 1113,22/1090,90 ns, dav1d-ssse3 25,12/24,64 ns (1,42×/1,40×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 10 bit dir 8x8: SSSE3 22,53/21,76 ns, skaler 87,18/85,01 ns, dav1d-ssse3 16,60/16,16 ns (1,36×/1,35×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 10 bit filter 8x8: SSSE3 97,10/93,28 ns, skaler 4376,33/4266,54 ns, dav1d-ssse3 63,39/61,96 ns (1,53×/1,51×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 10 bit filter 4x8: SSSE3 63,91/60,84 ns, skaler 2249,56/2192,02 ns, dav1d-ssse3 39,70/38,56 ns (1,61×/1,58×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 10 bit filter 4x4: SSSE3 41,85/39,29 ns, skaler 1138,55/1108,76 ns, dav1d-ssse3 23,86/23,17 ns (1,75×/1,70×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 12 bit dir 8x8: SSSE3 22,56/22,32 ns, skaler 86,82/87,97 ns, dav1d-ssse3 16,60/16,65 ns (1,36×/1,34×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 12 bit filter 8x8: SSSE3 96,75/96,35 ns, skaler 4260,74/4376,88 ns, dav1d-ssse3 63,52/63,42 ns (1,52×/1,52×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 12 bit filter 4x8: SSSE3 63,25/63,26 ns, skaler 2195,65/2221,67 ns, dav1d-ssse3 39,74/39,68 ns (1,59×/1,59×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 12 bit filter 4x4: SSSE3 41,29/41,27 ns, skaler 1115,12/1131,57 ns, dav1d-ssse3 23,88/23,83 ns (1,73×/1,73×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 8,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit dir 8x8: SSE4 18,17/17,63 ns, SSSE3 22,21/21,57 ns (eşli oran 0,82/0,82), dav1d-sse4 12,85/12,61 ns (1,41×/1,40×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 8,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: false,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 8x8: SSE4 85,21/82,33 ns, SSSE3 85,21/82,42 ns (eşli oran 1,00/1,00), dav1d-sse4 59,53/57,95 ns (1,43×/1,42×); varsayılan değil: alt düzeyden hızlı değil (fark gürültü bandında), yuva SSSE3'te",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 8,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: false,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 4x8: SSE4 54,65/52,91 ns, SSSE3 55,64/53,63 ns (eşli oran 0,98/0,99), dav1d-sse4 38,38/37,53 ns (1,42×/1,41×); varsayılan değil: alt düzeyden hızlı değil (fark gürültü bandında), yuva SSSE3'te",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 8,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: false,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 4x4: SSE4 35,36/34,21 ns, SSSE3 35,75/34,40 ns (eşli oran 0,99/1,00), dav1d-sse4 24,35/23,84 ns (1,45×/1,43×); varsayılan değil: alt düzeyden hızlı değil (fark gürültü bandında), yuva SSSE3'te",
    },
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 10,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 10 bit dir 8x8: SSE4 18,78/18,15 ns, SSSE3 22,53/21,76 ns (eşli oran 0,83/0,83), dav1d-sse4 12,99/12,65 ns (1,45×/1,43×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "dir",
        bit_depth: 12,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 12 bit dir 8x8: SSE4 18,76/18,62 ns, SSSE3 22,56/22,32 ns (eşli oran 0,83/0,84), dav1d-sse4 12,98/13,41 ns (1,45×/1,39×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 8x8: AVX512ICL 36,90/34,97 ns, AVX2 53,27/50,77 ns (eşli oran 0,69/0,69), dav1d-avx512icl 23,12/22,63 ns (1,60×/1,55×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 4x8: AVX512ICL 26,62/25,77 ns, AVX2 40,08/38,14 ns (eşli oran 0,66/0,67), dav1d-avx512icl 18,10/17,74 ns (1,47×/1,45×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 8 bit filter 4x4: AVX512ICL 23,38/22,33 ns, AVX2 26,83/25,75 ns (eşli oran 0,87/0,87), dav1d-avx512icl 15,85/15,55 ns (1,48×/1,44×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 10 bit filter 8x8: AVX512ICL 46,38/44,16 ns, AVX2 61,85/59,05 ns (eşli oran 0,75/0,75), dav1d-avx512icl 31,16/30,12 ns (1,49×/1,47×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 10 bit filter 4x8: AVX512ICL 36,24/34,50 ns, AVX2 46,50/44,84 ns (eşli oran 0,78/0,78), dav1d-avx512icl 15,81/13,78 ns (2,29×/2,50×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: false,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 10 bit filter 4x4: AVX512ICL 34,00/32,57 ns, AVX2 32,80/31,38 ns (eşli oran 1,04/1,04), dav1d-avx512icl 11,05/8,66 ns (3,08×/3,76×); varsayılan değil: AVX2'den yavaş, yuva AVX2'de",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[0]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 12 bit filter 8x8: AVX512ICL 46,27/45,84 ns, AVX2 61,79/61,29 ns (eşli oran 0,75/0,75), dav1d-avx512icl 30,99/31,36 ns (1,49×/1,46×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[1]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 12 bit filter 4x8: AVX512ICL 36,19/37,08 ns, AVX2 45,92/46,33 ns (eşli oran 0,78/0,82), dav1d-avx512icl 15,48/14,58 ns (2,34×/2,54×); varsayılan (çözme hattı f4-cdef-kapisi.md)",
    },
    IsaCandidate {
        family: "cdef",
        field: "fb[2]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: false,
        decision: "P30: cdef-bench (aynı girdi, 9 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) 12 bit filter 4x4: AVX512ICL 33,96/35,18 ns, AVX2 32,20/32,37 ns (eşli oran 1,05/1,05), dav1d-avx512icl 10,99/9,45 ns (3,09×/3,72×); varsayılan değil: AVX2'den yavaş, yuva AVX2'de",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put regular 28 hücre, skalerden 0,80–52,8× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,77–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular` (aynı 6-tap gövde, katsayı farkı) ölçümü: put regular 28 hücre, skalerden 0,80–52,8× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,77–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put regular_sharp 28 hücre, skalerden 0,80–52,0× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,72–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular_sharp` (8-tap içeren aynı gövde, katsayı farkı) ölçümü: put regular_sharp 28 hücre, skalerden 0,80–52,0× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,72–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular_sharp` (8-tap içeren aynı gövde, katsayı farkı) ölçümü: put regular_sharp 28 hücre, skalerden 0,80–52,0× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,72–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular_sharp` (8-tap içeren aynı gövde, katsayı farkı) ölçümü: put regular_sharp 28 hücre, skalerden 0,80–52,0× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,72–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular` (aynı 6-tap gövde, katsayı farkı) ölçümü: put regular 28 hücre, skalerden 0,80–52,8× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,77–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular` (aynı 6-tap gövde, katsayı farkı) ölçümü: put regular 28 hücre, skalerden 0,80–52,8× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,77–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular_sharp` (8-tap içeren aynı gövde, katsayı farkı) ölçümü: put regular_sharp 28 hücre, skalerden 0,80–52,0× hızlı (kopya 128×32 skalerle eşit: bellek bandı), Rust/dav1d AVX2 eşli oran 0,72–3,79; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put bilinear 7 hücre, skalerden 3,91–20,4× hızlı, Rust/dav1d AVX2 eşli oran 1,35–3,30; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep regular 24 hücre, skalerden 1,94–33,5× hızlı, Rust/dav1d AVX2 eşli oran 0,77–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular` (aynı 6-tap gövde, katsayı farkı) ölçümü: prep regular 24 hücre, skalerden 1,94–33,5× hızlı, Rust/dav1d AVX2 eşli oran 0,77–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep regular_sharp 24 hücre, skalerden 1,94–33,9× hızlı, Rust/dav1d AVX2 eşli oran 0,70–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular_sharp` (8-tap içeren aynı gövde, katsayı farkı) ölçümü: prep regular_sharp 24 hücre, skalerden 1,94–33,9× hızlı, Rust/dav1d AVX2 eşli oran 0,70–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular_sharp` (8-tap içeren aynı gövde, katsayı farkı) ölçümü: prep regular_sharp 24 hücre, skalerden 1,94–33,9× hızlı, Rust/dav1d AVX2 eşli oran 0,70–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular_sharp` (8-tap içeren aynı gövde, katsayı farkı) ölçümü: prep regular_sharp 24 hücre, skalerden 1,94–33,9× hızlı, Rust/dav1d AVX2 eşli oran 0,70–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular` (aynı 6-tap gövde, katsayı farkı) ölçümü: prep regular 24 hücre, skalerden 1,94–33,5× hızlı, Rust/dav1d AVX2 eşli oran 0,77–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular` (aynı 6-tap gövde, katsayı farkı) ölçümü: prep regular 24 hücre, skalerden 1,94–33,5× hızlı, Rust/dav1d AVX2 eşli oran 0,77–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) eş yol `regular_sharp` (8-tap içeren aynı gövde, katsayı farkı) ölçümü: prep regular_sharp 24 hücre, skalerden 1,94–33,9× hızlı, Rust/dav1d AVX2 eşli oran 0,70–3,50; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P32: mc-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep bilinear 6 hücre, skalerden 8,01–24,0× hızlı, Rust/dav1d AVX2 eşli oran 1,18–2,93; çözme hattı tespit kipi ccvb −%9,9, test25 −%5,3, vq002 −%3,8 (skaler kip −%0,1…−%0,3), çıktı aynı; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) avg 6 hücre, skalerden 3,92–12,3× hızlı, Rust/dav1d AVX2 eşli oran 1,04–3,96; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_avg 6 hücre, skalerden 4,12–13,9× hızlı, Rust/dav1d AVX2 eşli oran 1,16–2,72; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mask 6 hücre, skalerden 3,19–11,3× hızlı, Rust/dav1d AVX2 eşli oran 1,27–3,17; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 4:4:4 5 hücre, skalerden 6,87–12,5× hızlı, Rust/dav1d AVX2 eşli oran 1,21–2,72; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 4:2:2 5 hücre, skalerden 8,20–137,7× hızlı, Rust/dav1d AVX2 eşli oran 1,26–2,48; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 4:2:0 5 hücre, skalerden 8,55–144,8× hızlı, Rust/dav1d AVX2 eşli oran 1,17–2,40; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend 4 hücre, skalerden 2,64–15,4× hızlı, Rust/dav1d AVX2 eşli oran 1,24–4,15; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend_v 5 hücre, skalerden 1,83–21,0× hızlı, Rust/dav1d AVX2 eşli oran 1,43–3,02; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend_h 7 hücre, skalerden 1,17–12,9× hızlı, Rust/dav1d AVX2 eşli oran 0,83–4,97; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) warp8x8 (öteleme + abcd) 2 hücre, skalerden 5,33–5,9× hızlı, Rust/dav1d AVX2 eşli oran 2,27–2,52; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) warp8x8t (öteleme + abcd) 2 hücre, skalerden 5,22–5,7× hızlı, Rust/dav1d AVX2 eşli oran 2,28–2,56; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "emu_edge",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) emu_edge (sol/sağ/üst/alt/köşe) 30 hücre, skalerden 1,15–5,9× hızlı, Rust/dav1d AVX2 eşli oran 1,50–3,34; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular (8-tap genel gövde; 6-tap tablolu eş birleşimler aynı kod) 21 hücre, skalerden 1,43–11,5× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular (8-tap genel gövde; 6-tap tablolu eş birleşimler aynı kod) 21 hücre, skalerden 1,43–11,5× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 21 hücre, skalerden 1,39–11,1× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,39; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 21 hücre, skalerden 1,39–11,1× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,39; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 21 hücre, skalerden 1,39–11,1× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,39; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 21 hücre, skalerden 1,39–11,1× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,39; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular (8-tap genel gövde; 6-tap tablolu eş birleşimler aynı kod) 21 hücre, skalerden 1,43–11,5× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular (8-tap genel gövde; 6-tap tablolu eş birleşimler aynı kod) 21 hücre, skalerden 1,43–11,5× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 21 hücre, skalerden 1,39–11,1× hızlı, Rust/dav1d AVX2 eşli oran 1,50–8,39; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled bilinear 21 hücre, skalerden 1,41–3,3× hızlı, Rust/dav1d AVX2 eşli oran 1,20–3,14; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular (8-tap genel gövde; 6-tap tablolu eş birleşimler aynı kod) 18 hücre, skalerden 2,14–9,8× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,22; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular (8-tap genel gövde; 6-tap tablolu eş birleşimler aynı kod) 18 hücre, skalerden 2,14–9,8× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,22; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 18 hücre, skalerden 2,29–9,9× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 18 hücre, skalerden 2,29–9,9× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 18 hücre, skalerden 2,29–9,9× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 18 hücre, skalerden 2,29–9,9× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular (8-tap genel gövde; 6-tap tablolu eş birleşimler aynı kod) 18 hücre, skalerden 2,14–9,8× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,22; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular (8-tap genel gövde; 6-tap tablolu eş birleşimler aynı kod) 18 hücre, skalerden 2,14–9,8× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,22; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp (8-tap genel gövde; sharp içeren eş birleşimler aynı kod) 18 hücre, skalerden 2,29–9,9× hızlı, Rust/dav1d AVX2 eşli oran 1,54–9,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled bilinear 18 hücre, skalerden 1,97–3,4× hızlı, Rust/dav1d AVX2 eşli oran 1,22–3,05; varsayılan",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P33: mc-kalan-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) resize (9/8, 12/8, 16/8 × dst_w 128/512/1920) 9 hücre, skalerden 4,94–9,0× hızlı, Rust/dav1d AVX2 eşli oran 1,43–2,44; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @10 63 hücre, skalerden 0,99–29,5× hızlı, Rust/dav1d AVX2 eşli oran 0,94–5,17 (kopya w=128: bellek bant genişliği sınırlı, skalerin memcpy'siyle başa baş); bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put bilin @10 7 hücre, skalerden 3,02–17,0× hızlı, Rust/dav1d AVX2 eşli oran 1,78–3,69; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @10 54 hücre, skalerden 3,00–32,6× hızlı, Rust/dav1d AVX2 eşli oran 0,98–5,17; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep bilin @10 6 hücre, skalerden 4,85–16,3× hızlı, Rust/dav1d AVX2 eşli oran 1,74–3,72; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) put bilin @10 7 hücre, skalerden 3,02–17,0× hızlı, Rust/dav1d AVX2 eşli oran 1,78–3,69; 12 bit aynı gövde (`bitdepth_max`'tan kaydırma), 12 bit 8-tap: put 8tap @12 6 hücre, skalerden 6,99–30,7× hızlı, Rust/dav1d AVX2 eşli oran 1,15–4,95; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; bu birleşim aynı genel gövde (yalnız katsayı tablosu farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) prep bilin @10 6 hücre, skalerden 4,85–16,3× hızlı, Rust/dav1d AVX2 eşli oran 1,74–3,72; 12 bit aynı gövde (`bitdepth_max`'tan kaydırma), 12 bit 8-tap: prep 8tap @12 6 hücre, skalerden 4,66–32,3× hızlı, Rust/dav1d AVX2 eşli oran 1,12–5,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) avg @10 6 hücre, skalerden 3,59–13,1× hızlı, Rust/dav1d AVX2 eşli oran 1,28–2,57; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_avg @10 6 hücre, skalerden 3,83–14,1× hızlı, Rust/dav1d AVX2 eşli oran 0,98–2,43; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mask @10 6 hücre, skalerden 4,40–11,4× hızlı, Rust/dav1d AVX2 eşli oran 1,03–2,58; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 444 @10 5 hücre, skalerden 8,16–10,9× hızlı, Rust/dav1d AVX2 eşli oran 0,93–2,40; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 422 @10 5 hücre, skalerden 7,23–82,8× hızlı, Rust/dav1d AVX2 eşli oran 1,11–2,23; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 420 @10 5 hücre, skalerden 6,86–83,2× hızlı, Rust/dav1d AVX2 eşli oran 1,16–2,31; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) avg @12 3 hücre, skalerden 3,69–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,67–2,18; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_avg @12 3 hücre, skalerden 3,92–14,1× hızlı, Rust/dav1d AVX2 eşli oran 1,25–1,89; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mask @12 3 hücre, skalerden 4,22–11,9× hızlı, Rust/dav1d AVX2 eşli oran 1,29–2,23; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 444 @12 3 hücre, skalerden 7,83–10,8× hızlı, Rust/dav1d AVX2 eşli oran 1,12–2,38; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 422 @12 3 hücre, skalerden 7,15–81,1× hızlı, Rust/dav1d AVX2 eşli oran 1,19–2,20; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) w_mask 420 @12 3 hücre, skalerden 7,05–81,3× hızlı, Rust/dav1d AVX2 eşli oran 1,22–2,30; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend @10 4 hücre, skalerden 2,54–16,7× hızlı, Rust/dav1d AVX2 eşli oran 1,45–3,67; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend_v @10 5 hücre, skalerden 1,21–15,7× hızlı, Rust/dav1d AVX2 eşli oran 1,53–3,42; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend_h @10 7 hücre, skalerden 1,06–15,8× hızlı, Rust/dav1d AVX2 eşli oran 1,33–4,29; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend @10 4 hücre, skalerden 2,54–16,7× hızlı, Rust/dav1d AVX2 eşli oran 1,45–3,67; 12 bit: gövde bit derinliğinden bağımsız; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend_v @10 5 hücre, skalerden 1,21–15,7× hızlı, Rust/dav1d AVX2 eşli oran 1,53–3,42; 12 bit: gövde bit derinliğinden bağımsız; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) blend_h @10 7 hücre, skalerden 1,06–15,8× hızlı, Rust/dav1d AVX2 eşli oran 1,33–4,29; 12 bit: gövde bit derinliğinden bağımsız; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) warp8x8 @10 2 hücre, skalerden 5,67–5,8× hızlı, Rust/dav1d AVX2 eşli oran 2,31–2,53; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) warp8x8t @10 2 hücre, skalerden 5,64–5,8× hızlı, Rust/dav1d AVX2 eşli oran 2,30–2,53; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) warp8x8 @12 2 hücre, skalerden 5,76–5,8× hızlı, Rust/dav1d AVX2 eşli oran 2,42–2,50; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) warp8x8t @10 2 hücre, skalerden 5,64–5,8× hızlı, Rust/dav1d AVX2 eşli oran 2,30–2,53; 12 bit: aynı gövde (`bitdepth_max`'tan yuvarlama), 12 bit warp8x8: warp8x8 @12 2 hücre, skalerden 5,76–5,8× hızlı, Rust/dav1d AVX2 eşli oran 2,42–2,50; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "emu_edge",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) emu_edge @10 30 hücre, skalerden 1,21–10,1× hızlı, Rust/dav1d AVX2 eşli oran 1,17–3,71; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "emu_edge",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) emu_edge @10 30 hücre, skalerden 1,21–10,1× hızlı, Rust/dav1d AVX2 eşli oran 1,17–3,71; 12 bit: gövde bit derinliğinden bağımsız; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular @10 21 hücre, skalerden 2,30–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–6,12; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular @10 21 hücre, skalerden 2,30–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–6,12; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; bu birleşim sharp ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; bu birleşim sharp ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; bu birleşim sharp ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular @10 21 hücre, skalerden 2,30–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–6,12; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular @10 21 hücre, skalerden 2,30–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–6,12; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; bu birleşim sharp ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled bilin @10 21 hücre, skalerden 1,57–6,1× hızlı, Rust/dav1d AVX2 eşli oran 0,78–2,71; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular @10 18 hücre, skalerden 4,26–13,2× hızlı, Rust/dav1d AVX2 eşli oran 1,47–5,99; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular @10 18 hücre, skalerden 4,26–13,2× hızlı, Rust/dav1d AVX2 eşli oran 1,47–5,99; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; bu birleşim sharp ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; bu birleşim sharp ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; bu birleşim sharp ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular @10 18 hücre, skalerden 4,26–13,2× hızlı, Rust/dav1d AVX2 eşli oran 1,47–5,99; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular @10 18 hücre, skalerden 4,26–13,2× hızlı, Rust/dav1d AVX2 eşli oran 1,47–5,99; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; bu birleşim sharp ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled bilin @10 18 hücre, skalerden 1,76–5,1× hızlı, Rust/dav1d AVX2 eşli oran 0,83–2,76; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; 12 bit aynı gövde, 12 bit regular: mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; 12 bit aynı gövde, 12 bit regular: mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; 12 bit aynı gövde, 12 bit regular: mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; 12 bit aynı gövde, 12 bit regular: mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled sharp @10 21 hücre, skalerden 2,33–13,6× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,80; 12 bit aynı gövde, 12 bit regular: mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mc_scaled bilin @10 21 hücre, skalerden 1,57–6,1× hızlı, Rust/dav1d AVX2 eşli oran 0,78–2,71; 12 bit aynı gövde, 12 bit regular: mc_scaled regular @12 9 hücre, skalerden 4,00–13,5× hızlı, Rust/dav1d AVX2 eşli oran 1,33–5,29; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; 12 bit aynı gövde, 12 bit regular: mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; 12 bit aynı gövde, 12 bit regular: mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; 12 bit aynı gövde, 12 bit regular: mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; 12 bit aynı gövde, 12 bit regular: mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; bu birleşim regular ile aynı genel gövde (yalnız katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled sharp @10 18 hücre, skalerden 4,37–13,0× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,93; 12 bit aynı gövde, 12 bit regular: mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) mct_scaled bilin @10 18 hücre, skalerden 1,76–5,1× hızlı, Rust/dav1d AVX2 eşli oran 0,83–2,76; 12 bit aynı gövde, 12 bit regular: mct_scaled regular @12 9 hücre, skalerden 4,14–12,4× hızlı, Rust/dav1d AVX2 eşli oran 1,46–5,94; varsayılan",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) resize @10 9 hücre, skalerden 4,76–10,2× hızlı, Rust/dav1d AVX2 eşli oran 1,50–3,03; varsayılan",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P34: mc16-bench (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) resize @12 9 hücre, skalerden 4,77–10,2× hızlı, Rust/dav1d AVX2 eşli oran 1,50–3,03; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put regular (+ kopya) 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–30,61× hızlı, Rust/dav1d SSSE3 eşli oran 1,06–3,34; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put regular (+ kopya) 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–30,61× hızlı, Rust/dav1d SSSE3 eşli oran 1,06–3,34; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put regular_sharp (+ kopya) 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–29,73× hızlı, Rust/dav1d SSSE3 eşli oran 0,62–3,34; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put regular_sharp (+ kopya) 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–29,73× hızlı, Rust/dav1d SSSE3 eşli oran 0,62–3,34; eş yol `regular_sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put regular_sharp (+ kopya) 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–29,73× hızlı, Rust/dav1d SSSE3 eşli oran 0,62–3,34; eş yol `regular_sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put sharp (+ kopya) 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–28,39× hızlı, Rust/dav1d SSSE3 eşli oran 0,67–3,64; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put regular (+ kopya) 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–30,61× hızlı, Rust/dav1d SSSE3 eşli oran 1,06–3,34; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put regular (+ kopya) 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–30,61× hızlı, Rust/dav1d SSSE3 eşli oran 1,06–3,34; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put regular_sharp (+ kopya) 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–29,73× hızlı, Rust/dav1d SSSE3 eşli oran 0,62–3,34; eş yol `regular_sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench put bilin (+ kopya) 28 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,86–16,74× hızlı, Rust/dav1d SSSE3 eşli oran 1,22–3,34; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep regular (+ kopya) 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,82× hızlı, Rust/dav1d SSSE3 eşli oran 1,04–2,76; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep regular (+ kopya) 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,82× hızlı, Rust/dav1d SSSE3 eşli oran 1,04–2,76; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep regular_sharp (+ kopya) 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,02× hızlı, Rust/dav1d SSSE3 eşli oran 0,62–2,76; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep regular_sharp (+ kopya) 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,02× hızlı, Rust/dav1d SSSE3 eşli oran 0,62–2,76; eş yol `regular_sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep regular_sharp (+ kopya) 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,02× hızlı, Rust/dav1d SSSE3 eşli oran 0,62–2,76; eş yol `regular_sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep sharp (+ kopya) 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,47× hızlı, Rust/dav1d SSSE3 eşli oran 0,68–2,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep regular (+ kopya) 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,82× hızlı, Rust/dav1d SSSE3 eşli oran 1,04–2,76; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep regular (+ kopya) 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,82× hızlı, Rust/dav1d SSSE3 eşli oran 1,04–2,76; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep regular_sharp (+ kopya) 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–23,02× hızlı, Rust/dav1d SSSE3 eşli oran 0,62–2,76; eş yol `regular_sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-bench prep bilin (+ kopya) 24 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,79–16,17× hızlı, Rust/dav1d SSSE3 eşli oran 1,07–2,76; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,87× hızlı, Rust/dav1d SSSE3 eşli oran 0,95–3,18; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,87× hızlı, Rust/dav1d SSSE3 eşli oran 0,95–3,18; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular_sharp (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,86× hızlı, Rust/dav1d SSSE3 eşli oran 0,79–5,78; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular_sharp (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,86× hızlı, Rust/dav1d SSSE3 eşli oran 0,79–5,78; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular_sharp (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,86× hızlı, Rust/dav1d SSSE3 eşli oran 0,79–5,78; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put sharp (+ kopya) @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–20,62× hızlı, Rust/dav1d SSSE3 eşli oran 0,90–3,11; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,87× hızlı, Rust/dav1d SSSE3 eşli oran 0,95–3,18; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,87× hızlı, Rust/dav1d SSSE3 eşli oran 0,95–3,18; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular_sharp (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,86× hızlı, Rust/dav1d SSSE3 eşli oran 0,79–5,78; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put bilin (+ kopya) @10 28 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–10,04× hızlı, Rust/dav1d SSSE3 eşli oran 1,12–3,16; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,81× hızlı, Rust/dav1d SSSE3 eşli oran 1,01–3,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,81× hızlı, Rust/dav1d SSSE3 eşli oran 1,01–3,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular_sharp (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,92× hızlı, Rust/dav1d SSSE3 eşli oran 0,87–4,74; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular_sharp (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,92× hızlı, Rust/dav1d SSSE3 eşli oran 0,87–4,74; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular_sharp (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,92× hızlı, Rust/dav1d SSSE3 eşli oran 0,87–4,74; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep sharp (+ kopya) @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–21,15× hızlı, Rust/dav1d SSSE3 eşli oran 0,93–3,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,81× hızlı, Rust/dav1d SSSE3 eşli oran 1,01–3,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,81× hızlı, Rust/dav1d SSSE3 eşli oran 1,01–3,07; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular_sharp (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,92× hızlı, Rust/dav1d SSSE3 eşli oran 0,87–4,74; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep bilin (+ kopya) @10 24 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–9,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,26–3,28; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular (+ kopya) @12 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 8,87–22,49× hızlı, Rust/dav1d SSSE3 eşli oran 0,97–2,88; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular (+ kopya) @12 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 8,87–22,49× hızlı, Rust/dav1d SSSE3 eşli oran 0,97–2,88; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular_sharp (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,86× hızlı, Rust/dav1d SSSE3 eşli oran 0,79–5,78; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular_sharp (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,86× hızlı, Rust/dav1d SSSE3 eşli oran 0,79–5,78; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular_sharp (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,86× hızlı, Rust/dav1d SSSE3 eşli oran 0,79–5,78; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put sharp (+ kopya) @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–20,62× hızlı, Rust/dav1d SSSE3 eşli oran 0,90–3,11; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular (+ kopya) @12 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 8,87–22,49× hızlı, Rust/dav1d SSSE3 eşli oran 0,97–2,88; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular (+ kopya) @12 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 8,87–22,49× hızlı, Rust/dav1d SSSE3 eşli oran 0,97–2,88; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put regular_sharp (+ kopya) @10 56 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–22,86× hızlı, Rust/dav1d SSSE3 eşli oran 0,79–5,78; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench put bilin (+ kopya) @10 28 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 0,90–10,04× hızlı, Rust/dav1d SSSE3 eşli oran 1,12–3,16; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular (+ kopya) @12 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,56–23,71× hızlı, Rust/dav1d SSSE3 eşli oran 1,03–2,84; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular (+ kopya) @12 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,56–23,71× hızlı, Rust/dav1d SSSE3 eşli oran 1,03–2,84; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular_sharp (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,92× hızlı, Rust/dav1d SSSE3 eşli oran 0,87–4,74; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular_sharp (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,92× hızlı, Rust/dav1d SSSE3 eşli oran 0,87–4,74; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular_sharp (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,92× hızlı, Rust/dav1d SSSE3 eşli oran 0,87–4,74; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep sharp (+ kopya) @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–21,15× hızlı, Rust/dav1d SSSE3 eşli oran 0,93–3,07; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular (+ kopya) @12 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,56–23,71× hızlı, Rust/dav1d SSSE3 eşli oran 1,03–2,84; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular (+ kopya) @12 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,56–23,71× hızlı, Rust/dav1d SSSE3 eşli oran 1,03–2,84; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep regular_sharp (+ kopya) @10 48 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–24,92× hızlı, Rust/dav1d SSSE3 eşli oran 0,87–4,74; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench prep bilin (+ kopya) @10 24 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,07–9,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,26–3,28; 12 bit: gövde bit derinliğinden bağımsız, 10 bit ölçümü; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench avg 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,55–9,60× hızlı, Rust/dav1d SSSE3 eşli oran 1,56–3,80; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench w_avg 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,41–8,20× hızlı, Rust/dav1d SSSE3 eşli oran 1,43–4,12; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mask 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,63–7,76× hızlı, Rust/dav1d SSSE3 eşli oran 1,30–3,06; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench w_mask 444 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,51–8,99× hızlı, Rust/dav1d SSSE3 eşli oran 1,11–1,95; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench w_mask 422 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 7,91–75,96× hızlı, Rust/dav1d SSSE3 eşli oran 1,18–1,84; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench w_mask 420 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 7,85–80,60× hızlı, Rust/dav1d SSSE3 eşli oran 1,12–1,83; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench avg @10 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,74–8,01× hızlı, Rust/dav1d SSSE3 eşli oran 1,29–2,41; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench w_avg @10 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,87–8,71× hızlı, Rust/dav1d SSSE3 eşli oran 1,04–2,38; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mask @10 12 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,51–7,73× hızlı, Rust/dav1d SSSE3 eşli oran 1,09–2,69; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench w_mask 444 @10 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,55–7,61× hızlı, Rust/dav1d SSSE3 eşli oran 1,20–1,71; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench w_mask 422 @10 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,77–41,50× hızlı, Rust/dav1d SSSE3 eşli oran 1,13–1,57; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench w_mask 420 @10 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,78–43,65× hızlı, Rust/dav1d SSSE3 eşli oran 1,14–1,59; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench avg @12 6 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,77–7,96× hızlı, Rust/dav1d SSSE3 eşli oran 1,43–2,06; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench w_avg @12 6 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,90–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–2,10; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mask @12 6 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,70–7,47× hızlı, Rust/dav1d SSSE3 eşli oran 1,22–2,37; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench w_mask 444 @12 6 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,60–7,65× hızlı, Rust/dav1d SSSE3 eşli oran 1,19–1,71; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench w_mask 422 @12 6 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,81–40,73× hızlı, Rust/dav1d SSSE3 eşli oran 1,13–1,58; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench w_mask 420 @12 6 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,83–42,18× hızlı, Rust/dav1d SSSE3 eşli oran 1,13–1,59; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench blend 8 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,67–10,89× hızlı, Rust/dav1d SSSE3 eşli oran 1,27–4,01; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench blend_v 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,92–20,59× hızlı, Rust/dav1d SSSE3 eşli oran 1,42–3,13; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench blend_h 14 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,16–10,25× hızlı, Rust/dav1d SSSE3 eşli oran 1,10–4,67; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench blend @10 8 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,43–10,19× hızlı, Rust/dav1d SSSE3 eşli oran 1,77–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench blend_v @10 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,22–11,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,88–4,48; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench blend_h @10 14 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,08–9,75× hızlı, Rust/dav1d SSSE3 eşli oran 1,64–5,18; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench blend @10 8 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,43–10,19× hızlı, Rust/dav1d SSSE3 eşli oran 1,77–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench blend_v @10 10 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,22–11,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,88–4,48; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench blend_h @10 14 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,08–9,75× hızlı, Rust/dav1d SSSE3 eşli oran 1,64–5,18; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench warp8x8 4 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,77–4,87× hızlı, Rust/dav1d SSSE3 eşli oran 1,93–1,95; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench warp8x8t 4 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,72–4,86× hızlı, Rust/dav1d SSSE3 eşli oran 1,87–1,94; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 8,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: true,
        decision: "P35: mc-kalan-bench SSE4.1 warp8x8 4 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,91–5,14× hızlı, Rust/dav1d SSE4.1 eşli oran 1,87–1,94; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 8,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: true,
        decision: "P35: mc-kalan-bench SSE4.1 warp8x8t 4 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,85–5,07× hızlı, Rust/dav1d SSE4.1 eşli oran 1,87–1,92; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench warp8x8 @10 4 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,93–5,07× hızlı, Rust/dav1d SSSE3 eşli oran 1,62–1,67; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench warp8x8t @10 4 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,88–4,96× hızlı, Rust/dav1d SSSE3 eşli oran 1,63–1,68; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench warp8x8 @12 4 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,99–5,09× hızlı, Rust/dav1d SSSE3 eşli oran 1,61–1,67; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench warp8x8t @10 4 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,88–4,96× hızlı, Rust/dav1d SSSE3 eşli oran 1,63–1,68; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "emu_edge",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench emu_edge 60 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,19–5,02× hızlı, Rust/dav1d SSSE3 eşli oran 1,41–3,49; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "emu_edge",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench emu_edge @10 60 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,44–6,84× hızlı, Rust/dav1d SSSE3 eşli oran 1,05–3,37; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "emu_edge",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench emu_edge @10 60 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,44–6,84× hızlı, Rust/dav1d SSSE3 eşli oran 1,05–3,37; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled regular 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,34–9,37× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,67; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled regular 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,34–9,37× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,67; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled sharp 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,26–9,56× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,72; eş yol `sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled sharp 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,26–9,56× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,72; eş yol `sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled sharp 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,26–9,56× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,72; eş yol `sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled sharp 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,26–9,56× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,72; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled regular 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,34–9,37× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,67; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled regular 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,34–9,37× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,67; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled sharp 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,26–9,56× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,72; eş yol `sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mc_scaled bilin 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,39–3,32× hızlı, Rust/dav1d SSSE3 eşli oran 0,76–2,30; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled regular 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,52–8,00× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,65; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled regular 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,52–8,00× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,65; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled sharp 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,61–8,50× hızlı, Rust/dav1d SSSE3 eşli oran 1,17–3,69; eş yol `sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled sharp 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,61–8,50× hızlı, Rust/dav1d SSSE3 eşli oran 1,17–3,69; eş yol `sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled sharp 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,61–8,50× hızlı, Rust/dav1d SSSE3 eşli oran 1,17–3,69; eş yol `sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled sharp 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,61–8,50× hızlı, Rust/dav1d SSSE3 eşli oran 1,17–3,69; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled regular 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,52–8,00× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,65; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled regular 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,52–8,00× hızlı, Rust/dav1d SSSE3 eşli oran 1,16–3,65; eş yol `regular` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled sharp 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 3,61–8,50× hızlı, Rust/dav1d SSSE3 eşli oran 1,17–3,69; eş yol `sharp` (aynı gövde, katsayı farkı); varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench mct_scaled bilin 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,86–3,39× hızlı, Rust/dav1d SSSE3 eşli oran 0,78–2,26; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled regular @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,78–7,70× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,83; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled regular @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,78–7,70× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,83; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled regular @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,78–7,70× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,83; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled regular @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,78–7,70× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,83; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled bilin @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,56–3,74× hızlı, Rust/dav1d SSSE3 eşli oran 0,69–1,74; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled regular @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,01–8,91× hızlı, Rust/dav1d SSSE3 eşli oran 1,24–3,20; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled regular @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,01–8,91× hızlı, Rust/dav1d SSSE3 eşli oran 1,24–3,20; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled regular @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,01–8,91× hızlı, Rust/dav1d SSSE3 eşli oran 1,24–3,20; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled regular @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,01–8,91× hızlı, Rust/dav1d SSSE3 eşli oran 1,24–3,20; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled bilin @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,63–4,12× hızlı, Rust/dav1d SSSE3 eşli oran 0,59–1,73; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled regular @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,33–8,00× hızlı, Rust/dav1d SSSE3 eşli oran 1,38–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled regular @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,33–8,00× hızlı, Rust/dav1d SSSE3 eşli oran 1,38–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled regular @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,33–8,00× hızlı, Rust/dav1d SSSE3 eşli oran 1,38–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled regular @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,33–8,00× hızlı, Rust/dav1d SSSE3 eşli oran 1,38–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled sharp @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 2,76–8,03× hızlı, Rust/dav1d SSSE3 eşli oran 1,39–3,79; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mc_scaled bilin @10 42 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,56–3,74× hızlı, Rust/dav1d SSSE3 eşli oran 0,69–1,74; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled regular @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,83–8,54× hızlı, Rust/dav1d SSSE3 eşli oran 1,26–3,25; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled regular @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,83–8,54× hızlı, Rust/dav1d SSSE3 eşli oran 1,26–3,25; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled regular @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,83–8,54× hızlı, Rust/dav1d SSSE3 eşli oran 1,26–3,25; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled regular @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,83–8,54× hızlı, Rust/dav1d SSSE3 eşli oran 1,26–3,25; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled sharp @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 5,22–8,57× hızlı, Rust/dav1d SSSE3 eşli oran 1,23–3,21; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mct_scaled[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench mct_scaled bilin @10 36 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 1,63–4,12× hızlı, Rust/dav1d SSSE3 eşli oran 0,59–1,73; varsayılan",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc-kalan-bench resize 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 4,73–8,20× hızlı, Rust/dav1d SSSE3 eşli oran 1,08–1,75; varsayılan",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench resize @10 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 6,05–9,71× hızlı, Rust/dav1d SSSE3 eşli oran 0,91–1,41; varsayılan",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P35: mc16-bench resize @12 18 hücre (aynı girdi, 5 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10), skalerden 6,04–9,71× hızlı, Rust/dav1d SSSE3 eşli oran 0,90–1,42; varsayılan",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 28 hücre: geometrik ortalama 0,888/0,870, ortanca 0,997/0,973, en büyük 1,034/1,048; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 28 hücre: geometrik ortalama 0,888/0,870, ortanca 0,997/0,973, en büyük 1,034/1,048; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre: geometrik ortalama 0,848/0,839, ortanca 0,986/0,990, en büyük 1,040/1,020; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre: geometrik ortalama 0,848/0,839, ortanca 0,986/0,990, en büyük 1,040/1,020; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre: geometrik ortalama 0,848/0,839, ortanca 0,986/0,990, en büyük 1,040/1,020; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 14 hücre: geometrik ortalama 0,872/0,858, ortanca 1,001/0,995, en büyük 1,037/1,031; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 28 hücre: geometrik ortalama 0,888/0,870, ortanca 0,997/0,973, en büyük 1,034/1,048; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 28 hücre: geometrik ortalama 0,888/0,870, ortanca 0,997/0,973, en büyük 1,034/1,048; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre: geometrik ortalama 0,848/0,839, ortanca 0,986/0,990, en büyük 1,040/1,020; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 7 hücre: geometrik ortalama 0,864/0,874, ortanca 0,994/0,998, en büyük 1,005/1,012; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 24 hücre: geometrik ortalama 0,855/0,861, ortanca 0,950/0,961, en büyük 1,031/1,069; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 24 hücre: geometrik ortalama 0,855/0,861, ortanca 0,950/0,961, en büyük 1,031/1,069; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre: geometrik ortalama 0,818/0,816, ortanca 0,899/0,905, en büyük 1,022/1,023; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre: geometrik ortalama 0,818/0,816, ortanca 0,899/0,905, en büyük 1,022/1,023; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre: geometrik ortalama 0,818/0,816, ortanca 0,899/0,905, en büyük 1,022/1,023; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 12 hücre: geometrik ortalama 0,839/0,826, ortanca 0,921/0,899, en büyük 1,022/1,012; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 24 hücre: geometrik ortalama 0,855/0,861, ortanca 0,950/0,961, en büyük 1,031/1,069; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 24 hücre: geometrik ortalama 0,855/0,861, ortanca 0,950/0,961, en büyük 1,031/1,069; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre: geometrik ortalama 0,818/0,816, ortanca 0,899/0,905, en büyük 1,022/1,023; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,834/0,846, ortanca 0,856/0,863, en büyük 1,016/1,010; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,887/0,960, ortanca 0,922/0,972, en büyük 1,015/1,016; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,888/0,895, ortanca 0,934/0,921, en büyük 1,022/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,879/0,886, ortanca 0,908/0,890, en büyük 1,029/1,037; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre: geometrik ortalama 0,854/0,834, ortanca 0,846/0,785, en büyük 1,010/1,018; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre: geometrik ortalama 0,831/0,791, ortanca 0,805/0,704, en büyük 1,004/1,004; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre: geometrik ortalama 0,862/0,812, ortanca 0,866/0,761, en büyük 1,009/1,031; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 4 hücre: geometrik ortalama 0,931/0,919, ortanca 0,992/1,001, en büyük 0,999/1,013; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre: geometrik ortalama 0,928/0,887, ortanca 0,988/1,002, en büyük 0,996/1,004; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 7 hücre: geometrik ortalama 0,908/0,856, ortanca 0,993/1,002, en büyük 1,016/1,009; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 2 hücre: geometrik ortalama 0,722/0,761, ortanca 0,722/0,761, en büyük 0,722/0,761; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 2 hücre: geometrik ortalama 0,713/0,758, ortanca 0,713/0,758, en büyük 0,715/0,763; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 9 hücre: geometrik ortalama 0,329/0,309, ortanca 0,327/0,298, en büyük 0,400/0,402; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 28 hücre: geometrik ortalama 0,913/0,894, ortanca 1,004/0,999, en büyük 1,071/1,060; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 28 hücre: geometrik ortalama 0,913/0,894, ortanca 1,004/0,999, en büyük 1,071/1,060; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre: geometrik ortalama 0,885/0,860, ortanca 1,007/1,001, en büyük 1,032/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre: geometrik ortalama 0,885/0,860, ortanca 1,007/1,001, en büyük 1,032/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre: geometrik ortalama 0,885/0,860, ortanca 1,007/1,001, en büyük 1,032/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 14 hücre: geometrik ortalama 0,894/0,866, ortanca 0,993/1,000, en büyük 1,043/1,038; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 28 hücre: geometrik ortalama 0,913/0,894, ortanca 1,004/0,999, en büyük 1,071/1,060; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 28 hücre: geometrik ortalama 0,913/0,894, ortanca 1,004/0,999, en büyük 1,071/1,060; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre: geometrik ortalama 0,885/0,860, ortanca 1,007/1,001, en büyük 1,032/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 7 hücre: geometrik ortalama 0,914/0,856, ortanca 1,001/0,998, en büyük 1,029/1,012; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 24 hücre: geometrik ortalama 0,893/0,838, ortanca 0,954/0,937, en büyük 1,051/1,002; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 24 hücre: geometrik ortalama 0,893/0,838, ortanca 0,954/0,937, en büyük 1,051/1,002; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre: geometrik ortalama 0,871/0,821, ortanca 0,919/0,888, en büyük 1,047/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre: geometrik ortalama 0,871/0,821, ortanca 0,919/0,888, en büyük 1,047/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre: geometrik ortalama 0,871/0,821, ortanca 0,919/0,888, en büyük 1,047/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 12 hücre: geometrik ortalama 0,855/0,832, ortanca 0,907/0,890, en büyük 1,026/0,998; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 24 hücre: geometrik ortalama 0,893/0,838, ortanca 0,954/0,937, en büyük 1,051/1,002; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 24 hücre: geometrik ortalama 0,893/0,838, ortanca 0,954/0,937, en büyük 1,051/1,002; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre: geometrik ortalama 0,871/0,821, ortanca 0,919/0,888, en büyük 1,047/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,910/0,869, ortanca 0,914/0,901, en büyük 1,013/1,004; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,823/0,888, ortanca 0,908/1,004, en büyük 1,046/1,020; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,823/0,888, ortanca 0,908/1,004, en büyük 1,046/1,020; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,885/0,860, ortanca 1,007/1,001, en büyük 1,032/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,885/0,860, ortanca 1,007/1,001, en büyük 1,032/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,885/0,860, ortanca 1,007/1,001, en büyük 1,032/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 14 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,894/0,866, ortanca 0,993/1,000, en büyük 1,043/1,038; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,823/0,888, ortanca 0,908/1,004, en büyük 1,046/1,020; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,823/0,888, ortanca 0,908/1,004, en büyük 1,046/1,020; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 21 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,885/0,860, ortanca 1,007/1,001, en büyük 1,032/1,047; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mc[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 7 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,914/0,856, ortanca 1,001/0,998, en büyük 1,029/1,012; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,832/0,888, ortanca 0,937/0,997, en büyük 1,014/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,832/0,888, ortanca 0,937/0,997, en büyük 1,014/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_REGULAR_SHARP]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,871/0,821, ortanca 0,919/0,888, en büyük 1,047/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_REGULAR]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,871/0,821, ortanca 0,919/0,888, en büyük 1,047/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,871/0,821, ortanca 0,919/0,888, en büyük 1,047/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SHARP]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 12 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,855/0,832, ortanca 0,907/0,890, en büyük 1,026/0,998; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,832/0,888, ortanca 0,937/0,997, en büyük 1,014/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,832/0,888, ortanca 0,937/0,997, en büyük 1,014/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_8TAP_SMOOTH_SHARP]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 18 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,871/0,821, ortanca 0,919/0,888, en büyük 1,047/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mct[FILTER_2D_BILINEAR]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,910/0,869, ortanca 0,914/0,901, en büyük 1,013/1,004; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,832/0,793, ortanca 0,824/0,799, en büyük 1,018/1,021; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,836/0,791, ortanca 0,835/0,788, en büyük 1,012/1,015; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 6 hücre: geometrik ortalama 0,824/0,799, ortanca 0,760/0,740, en büyük 1,011/1,017; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre: geometrik ortalama 0,792/0,721, ortanca 0,711/0,612, en büyük 0,994/0,995; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre: geometrik ortalama 0,740/0,695, ortanca 0,619/0,585, en büyük 0,997/1,005; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre: geometrik ortalama 0,731/0,702, ortanca 0,610/0,641, en büyük 1,003/1,009; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "avg",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 3 hücre: geometrik ortalama 0,804/0,800, ortanca 0,792/0,833, en büyük 1,029/1,013; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_avg",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 3 hücre: geometrik ortalama 0,823/0,808, ortanca 0,824/0,845, en büyük 1,057/1,036; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "mask",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 3 hücre: geometrik ortalama 0,792/0,812, ortanca 0,752/0,767, en büyük 1,022/1,000; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[0]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 3 hücre: geometrik ortalama 0,854/0,827, ortanca 0,984/0,993, en büyük 1,018/1,009; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[1]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 3 hücre: geometrik ortalama 0,847/0,795, ortanca 1,011/0,994, en büyük 1,016/0,997; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "w_mask[2]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 3 hücre: geometrik ortalama 0,855/0,802, ortanca 1,010/0,997, en büyük 1,017/1,008; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 4 hücre: geometrik ortalama 0,874/0,870, ortanca 0,887/0,901, en büyük 1,000/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre: geometrik ortalama 0,898/0,894, ortanca 0,994/0,997, en büyük 1,000/1,004; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 7 hücre: geometrik ortalama 0,899/0,886, ortanca 0,879/0,887, en büyük 1,000/1,002; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 4 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,874/0,870, ortanca 0,887/0,901, en büyük 1,000/1,003; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_v",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 5 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,898/0,894, ortanca 0,994/0,997, en büyük 1,000/1,004; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "blend_h",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 7 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,899/0,886, ortanca 0,879/0,887, en büyük 1,000/1,002; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 2 hücre: geometrik ortalama 0,715/0,718, ortanca 0,715/0,718, en büyük 0,716/0,720; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 2 hücre: geometrik ortalama 0,717/0,718, ortanca 0,717/0,718, en büyük 0,717/0,721; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 2 hücre: geometrik ortalama 0,725/0,712, ortanca 0,725/0,712, en büyük 0,726/0,712; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "mc",
        field: "warp8x8t",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 2 hücre (12 bit hücresi yok: aynı 16 bpc gövdesinin 10 bit hücreleri): geometrik ortalama 0,717/0,718, ortanca 0,717/0,718, en büyük 0,717/0,721; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 9 hücre: geometrik ortalama 0,362/0,354, ortanca 0,365/0,357, en büyük 0,388/0,381; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "resize",
        field: "resize",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P36: hücre ölçümü (aynı girdi, 7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 9 hücre: geometrik ortalama 0,362/0,352, ortanca 0,363/0,356, en büyük 0,393/0,378; varsayılan (çözme hattı f4-mc-avx512.md §10)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_y",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P39: hücre ölçümünde ve çözme hattında skalerden hızlı (docs/baseline/f4-fg-8bpc-avx2.md §3, §6); LFSR dörder adım + AR üst satırları AVX2 (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P39: hücre ölçümünde ve çözme hattında skalerden hızlı (docs/baseline/f4-fg-8bpc-avx2.md §3, §6); LFSR dörder adım + AR üst satırları AVX2 (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P39: hücre ölçümünde ve çözme hattında skalerden hızlı (docs/baseline/f4-fg-8bpc-avx2.md §3, §6); LFSR dörder adım + AR üst satırları AVX2 (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P39: hücre ölçümünde ve çözme hattında skalerden hızlı (docs/baseline/f4-fg-8bpc-avx2.md §3, §6); LFSR dörder adım + AR üst satırları AVX2 (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P39: hücre ölçümünde ve çözme hattında skalerden hızlı (docs/baseline/f4-fg-8bpc-avx2.md §3, §6); ölçekleme LUT araması vpgatherdd (rust_method: core::arch); SLOW_GATHER'da kurulmaz",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P39: hücre ölçümünde ve çözme hattında skalerden hızlı (docs/baseline/f4-fg-8bpc-avx2.md §3, §6); ölçekleme LUT araması vpgatherdd (rust_method: core::arch); SLOW_GATHER'da kurulmaz",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P39: hücre ölçümünde ve çözme hattında skalerden hızlı (docs/baseline/f4-fg-8bpc-avx2.md §3, §6); ölçekleme LUT araması vpgatherdd (rust_method: core::arch); SLOW_GATHER'da kurulmaz",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P39: hücre ölçümünde ve çözme hattında skalerden hızlı (docs/baseline/f4-fg-8bpc-avx2.md §3, §6); ölçekleme LUT araması vpgatherdd (rust_method: core::arch); SLOW_GATHER'da kurulmaz",
    },
    // P41: 8 bpc SSSE3 film grain (8 kayıt; dav1d `film_grain_dsp_init_x86`
    // `src/x86/filmgrain.h:48-72`: `SLOW_GATHER` etkin ve AVX512ICL kapalıyken
    // `fgy`/`fguv` bu SSSE3 adaylarına düşer).
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_y",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P41: hücre ölçümünde skalerden ve dav1d SSSE3'ten hızlı; LFSR dörder adım + AR üst satırları pmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P41: hücre ölçümünde skalerden ve dav1d SSSE3'ten hızlı; LFSR dörder adım + AR üst satırları pmaddwd + phaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P41: hücre ölçümünde skalerden ve dav1d SSSE3'ten hızlı; LFSR dörder adım + AR üst satırları pmaddwd + phaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P41: hücre ölçümünde skalerden ve dav1d SSSE3'ten hızlı; LFSR dörder adım + AR üst satırları pmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P41: hücre ölçümünde skalerden ve dav1d SSSE3'ten hızlı; ölçekleme &[u8; 256] doğrudan okuma + pmulhrsw (rust_method: core::arch); SLOW_GATHER'da AVX2 yerine bu aday kalır",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P41: hücre ölçümünde skalerden ve dav1d SSSE3'ten hızlı; pmaddubsw + pavgw + &[u8; 256] ölçekleme (rust_method: core::arch); SLOW_GATHER'da AVX2 yerine bu aday kalır",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P41: hücre ölçümünde skalerden ve dav1d SSSE3'ten hızlı; pmaddubsw + pavgw + &[u8; 256] ölçekleme (rust_method: core::arch); SLOW_GATHER'da AVX2 yerine bu aday kalır",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 8,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P41: hücre ölçümünde skalerden ve dav1d SSSE3'ten hızlı; pmaddubsw + &[u8; 256] ölçekleme (rust_method: core::arch); SLOW_GATHER'da AVX2 yerine bu aday kalır",
    },
    // P41 — 8 bpc AVX512ICL film grain (4 kayıt: fgy_32x32xn + fguv_32x32xn[420/422/444]).
    // vpermi2b/vpermt2b tabanlı; SLOW_GATHER bayrağı açıkken de kurulur (filmgrain.h:76).
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P41: fg-bench 10 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 0,18–0,22×, Rust / dav1d AVX512ICL eşli oran 1,15–1,30× / 1,14–1,34× (≤ 1,50×); 64 bayt ZMM + vpermi2b/vpermt2b ölçekleme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P41: fg-bench 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 0,27–0,35×, Rust / dav1d AVX512ICL eşli oran 1,27–1,49× / 1,24–1,49× (≤ 1,50×); 4 satır ZMM + vpermi2b/vpermt2b luma ve ölçekleme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P41: fg-bench 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 0,27–0,34×, Rust / dav1d AVX512ICL eşli oran 1,15–1,27× / 1,16–1,27× (≤ 1,50×); 4 satır ZMM + vpermi2b/vpermt2b luma ve ölçekleme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 8,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P41: fg-bench 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / AVX2 0,22–0,24×, Rust / dav1d AVX512ICL eşli oran 1,19–1,43× / 1,21–1,42× (≤ 1,50×); 2 satır ZMM + vpermi2b/vpermt2b ölçekleme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    // P42 — 16 bpc (10 ve 12 bit) SSSE3 film grain (8 kayıt × 2 bit derinliği = 16 yuva;
    // dav1d `film_grain_dsp_init_x86` `src/x86/filmgrain.h:48-72`).
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_y",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,24–0,47×, Rust / dav1d SSSE3 eşli oran 0,57–1,86× / 0,57–1,87× (≤ 2,00×); LFSR dörder adım + AR üst satırları pmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,16–0,30×, Rust / dav1d SSSE3 eşli oran 0,79–1,79× / 0,79–1,78× (≤ 2,00×); LFSR dörder adım + AR üst satırları pmaddwd + phaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,16–0,30×, Rust / dav1d SSSE3 eşli oran 0,75–1,76× / 0,75–1,77× (≤ 2,00×); LFSR dörder adım + AR üst satırları pmaddwd + phaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,13–0,28×, Rust / dav1d SSSE3 eşli oran 0,64–1,76× / 0,64–1,77× (≤ 2,00×); LFSR dörder adım + AR üst satırları pmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 10 bpc 10 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,21–0,27×, Rust / dav1d SSSE3 eşli oran 1,34–1,53× / 1,34–1,55× (≤ 2,00×); &[u8; 1024] ölçekleme + pmulhrsw + 32 bit pmaddwd bindirme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,09–0,10×, Rust / dav1d SSSE3 eşli oran 1,34–1,79× / 1,33–1,79× (≤ 2,00×); phaddw + pavgw + &[u8; 1024] ölçekleme + 32 bit pmaddwd bindirme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,08–0,09×, Rust / dav1d SSSE3 eşli oran 1,36–1,63× / 1,35–1,66× (≤ 2,00×); phaddw + pavgw + &[u8; 1024] ölçekleme + 32 bit pmaddwd bindirme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 10,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,08–0,08×, Rust / dav1d SSSE3 eşli oran 1,26–1,39× / 1,30–1,38× (≤ 2,00×); &[u8; 1024] ölçekleme + pmulhrsw + 32 bit pmaddwd bindirme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_y",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,24–0,52×, Rust / dav1d SSSE3 eşli oran 0,57–1,86× / 0,57–1,86× (≤ 2,00×); LFSR dörder adım (shift==0 sinyali dahil) + AR üst satırları pmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,16–0,30×, Rust / dav1d SSSE3 eşli oran 0,79–1,78× / 0,79–1,79× (≤ 2,00×); LFSR dörder adım (shift==0 sinyali dahil) + AR üst satırları pmaddwd + phaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,16–0,29×, Rust / dav1d SSSE3 eşli oran 0,76–1,76× / 0,76–1,74× (≤ 2,00×); LFSR dörder adım (shift==0 sinyali dahil) + AR üst satırları pmaddwd + phaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,13–0,28×, Rust / dav1d SSSE3 eşli oran 0,65–1,77× / 0,64–1,77× (≤ 2,00×); LFSR dörder adım (shift==0 sinyali dahil) + AR üst satırları pmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 12 bpc 10 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,22–0,27×, Rust / dav1d SSSE3 eşli oran 1,36–1,55× / 1,35–1,52× (≤ 2,00×); &[u8; 4096] ölçekleme + pmulhrsw + 32 bit pmaddwd bindirme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,08–0,10×, Rust / dav1d SSSE3 eşli oran 1,35–1,67× / 1,35–1,67× (≤ 2,00×); phaddw + pavgw + &[u8; 4096] ölçekleme + 32 bit pmaddwd bindirme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,08–0,09×, Rust / dav1d SSSE3 eşli oran 1,32–1,59× / 1,31–1,59× (≤ 2,00×); phaddw + pavgw + &[u8; 4096] ölçekleme + 32 bit pmaddwd bindirme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 12,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust SSSE3 / skaler 0,08–0,09×, Rust / dav1d SSSE3 eşli oran 1,27–1,44× / 1,29–1,45× (≤ 2,00×); &[u8; 4096] ölçekleme + pmulhrsw + 32 bit pmaddwd bindirme (rust_method: core::arch); SLOW_GATHER'da da kurulur",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_y",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,18–0,34×, Rust / dav1d AVX2 eşli oran 0,59–0,88× / 0,59–0,87× (≤ 1,50×); LFSR dörder adım + 16 şerit AR vpmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,11–0,14×, Rust / dav1d AVX2 eşli oran 0,64–1,05× / 0,64–1,04× (≤ 1,50×); LFSR dörder adım + 16 şerit AR vpmaddwd + vphaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,10–0,14×, Rust / dav1d AVX2 eşli oran 0,61–1,03× / 0,61–1,03× (≤ 1,50×); LFSR dörder adım + 16 şerit AR vpmaddwd + vphaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,10–0,12×, Rust / dav1d AVX2 eşli oran 0,59–0,89× / 0,58–0,89× (≤ 1,50×); LFSR dörder adım + 16 şerit AR vpmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 10 bpc 10 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,21–0,25×, Rust / dav1d AVX2 eşli oran 1,21–1,36× / 1,20–1,32× (≤ 1,50×); &[u8; 1024] sınır güvenli 4 bayt hizalı vpgatherdd + vpmulhrsw + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,08–0,09×, Rust / dav1d AVX2 eşli oran 1,21–1,30× / 1,20–1,30× (≤ 1,50×); vphaddw + vpavgw + &[u8; 1024] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,07–0,08×, Rust / dav1d AVX2 eşli oran 1,11–1,31× / 1,13–1,31× (≤ 1,50×); vphaddw + vpavgw + &[u8; 1024] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,07–0,09×, Rust / dav1d AVX2 eşli oran 1,14–1,46× / 1,17–1,49× (≤ 1,50×); &[u8; 1024] sınır güvenli vpgatherdd + vpmulhrsw + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_y",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,18–0,34×, Rust / dav1d AVX2 eşli oran 0,60–0,87× / 0,60–0,87× (≤ 1,50×); LFSR dörder adım (shift==0 dahil) + 16 şerit AR vpmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,11–0,15×, Rust / dav1d AVX2 eşli oran 0,64–1,05× / 0,64–1,05× (≤ 1,50×); LFSR dörder adım (shift==0 dahil) + 16 şerit AR vpmaddwd + vphaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,11–0,14×, Rust / dav1d AVX2 eşli oran 0,62–1,02× / 0,62–1,01× (≤ 1,50×); LFSR dörder adım (shift==0 dahil) + 16 şerit AR vpmaddwd + vphaddw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,10–0,13×, Rust / dav1d AVX2 eşli oran 0,59–0,90× / 0,59–0,90× (≤ 1,50×); LFSR dörder adım (shift==0 dahil) + 16 şerit AR vpmaddwd (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 12 bpc 10 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,21–0,25×, Rust / dav1d AVX2 eşli oran 1,22–1,33× / 1,22–1,34× (≤ 1,50×); &[u8; 4096] sınır güvenli 4 bayt hizalı vpgatherdd + vpmulhrsw + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,08–0,09×, Rust / dav1d AVX2 eşli oran 1,16–1,35× / 1,15–1,34× (≤ 1,50×); vphaddw + vpavgw + &[u8; 4096] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,07–0,08×, Rust / dav1d AVX2 eşli oran 1,11–1,29× / 1,10–1,26× (≤ 1,50×); vphaddw + vpavgw + &[u8; 4096] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX2 / skaler 0,07–0,08×, Rust / dav1d AVX2 eşli oran 1,11–1,37× / 1,14–1,39× (≤ 1,50×); &[u8; 4096] sınır güvenli vpgatherdd + vpmulhrsw + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    // P42 — 16 bpc (10 ve 12 bit) AVX512ICL film grain (4 kayıt × 2 bit derinliği = 8 yuva:
    // fgy_32x32xn + fguv_32x32xn[420/422/444]; dav1d `film_grain_dsp_init_x86` `src/x86/filmgrain.h:74-80`).
    // 512 bit vpgatherdd tabanlı olduğu için SLOW_GATHER bayrağı açıkken kurulmaz (`!SLOW_GATHER`).
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P42: fg16-bench 10 bpc 10 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,24–0,29×, Rust / dav1d AVX512ICL eşli oran 1,30–1,39× / 1,29–1,36× (≤ 1,50×); 64 bayt ZMM + &[u8; 1024] sınır güvenli 4 bayt hizalı vpgatherdd + vpternlogd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,08–0,09×, Rust / dav1d AVX512ICL eşli oran 1,29–1,42× / 1,32–1,44× (≤ 1,50×); vpmaddwd + vpermt2w + vpavgw + &[u8; 1024] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,08–0,09×, Rust / dav1d AVX512ICL eşli oran 1,30–1,48× / 1,30–1,53× (≤ 1,65×); vpmaddwd + vpermt2w + &[u8; 1024] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 10,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P42: fg16-bench 10 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,07–0,08× (Rust AVX512ICL / AVX2 0,92–1,06× / 0,97–1,04×), Rust / dav1d AVX512ICL eşli oran 1,10–1,34× / 1,13–1,38× (≤ 1,50×); 64 bayt ZMM + &[u8; 1024] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fgy_32x32xn",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P42: fg16-bench 12 bpc 10 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,26–0,30×, Rust / dav1d AVX512ICL eşli oran 1,27–1,40× / 1,32–1,39× (≤ 1,50×); 64 bayt ZMM + &[u8; 4096] sınır güvenli 4 bayt hizalı vpgatherdd + vpternlogd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,08–0,09×, Rust / dav1d AVX512ICL eşli oran 1,33–1,43× / 1,33–1,46× (≤ 1,50×); vpmaddwd + vpermt2w + vpavgw + &[u8; 4096] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,08–0,10×, Rust / dav1d AVX512ICL eşli oran 1,34–1,55× / 1,37–1,54× (≤ 1,65×); vpmaddwd + vpermt2w + &[u8; 4096] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    IsaCandidate {
        family: "filmgrain",
        field: "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]",
        bit_depth: 12,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "P42: fg16-bench 12 bpc 4 hücre (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) Rust AVX512ICL / skaler 0,07–0,08× (Rust AVX512ICL / AVX2 1,01–1,15× / 0,99–1,17×), Rust / dav1d AVX512ICL eşli oran 1,22–1,40× / 1,24–1,40× (≤ 1,50×); 64 bayt ZMM + &[u8; 4096] sınır güvenli vpgatherdd + 32 bit vpmaddwd bindirme (rust_method: core::arch); yalnız !SLOW_GATHER",
    },
    // P43 Adım 1 — Kayıpsız 4×4 WHT_WHT (SSE2 D02583/D02584 ve AVX2 D02581/D02582;
    // 4 kayıt = 6 yuva adayı: 8, 10 ve 12 bit).
    IsaCandidate {
        family: "itx",
        field: "itxfm_add[TX_4X4][WHT_WHT]",
        bit_depth: 8,
        implementation: "sse2",
        requires: CpuFlags::SSE2,
        default: true,
        decision: "P43: itx-bench 8 bpc wht_wht 4x4 (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) SSE2 4,68/4,68 ns, skaler 34,44/34,79 ns (0,14×/0,13×), dav1d-sse2 3,23/3,23 ns (eşli oran 1,45×/1,45×); paketlenmiş 16 bit IWHT4_1D + doyurmalı packuswb (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "itx",
        field: "itxfm_add[TX_4X4][WHT_WHT]",
        bit_depth: 10,
        implementation: "sse2",
        requires: CpuFlags::SSE2,
        default: true,
        decision: "P43: itx16-bench 10 bpc wht_wht 4x4 (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) SSE2 5,08/5,06 ns, skaler 34,49/34,74 ns (0,15×/0,15×), dav1d-sse2 3,46/3,43 ns (eşli oran 1,47×/1,48×); 32 bit IWHT4_1D + pminsw/pmaxsw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "itx",
        field: "itxfm_add[TX_4X4][WHT_WHT]",
        bit_depth: 12,
        implementation: "sse2",
        requires: CpuFlags::SSE2,
        default: true,
        decision: "P43: itx16-bench 12 bpc wht_wht 4x4 (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) SSE2 5,13/5,13 ns, skaler 34,94/34,64 ns (0,15×/0,15×), dav1d-sse2 3,23/3,21 ns (eşli oran 1,59×/1,60×); 32 bit IWHT4_1D + pminsw/pmaxsw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "itx",
        field: "itxfm_add[TX_4X4][WHT_WHT]",
        bit_depth: 8,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P43: itx-bench 8 bpc wht_wht 4x4 (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) AVX2 4,51/4,49 ns, skaler 34,44/34,79 ns (0,13×/0,13×), dav1d-avx2 3,21/3,16 ns (eşli oran 1,41×/1,41×); paketlenmiş 16 bit IWHT4_1D + vpblendd + pmovzxbw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "itx",
        field: "itxfm_add[TX_4X4][WHT_WHT]",
        bit_depth: 10,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P43: itx16-bench 10 bpc wht_wht 4x4 (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) AVX2 4,99/4,99 ns, skaler 34,49/34,74 ns (0,14×/0,14×), dav1d-avx2 4,08/4,06 ns (eşli oran 1,23×/1,23×); 256 bit vpermq IWHT4_1D + pminsw/pmaxsw (rust_method: core::arch)",
    },
    IsaCandidate {
        family: "itx",
        field: "itxfm_add[TX_4X4][WHT_WHT]",
        bit_depth: 12,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "P43: itx16-bench 12 bpc wht_wht 4x4 (7 dönüşümlü tur, ccd0 çekirdek 6 / ccd1 çekirdek 10) AVX2 5,08/5,04 ns, skaler 34,94/34,64 ns (0,15×/0,15×), dav1d-avx2 3,93/3,93 ns (eşli oran 1,29×/1,28×); 256 bit vpermq IWHT4_1D + pminsw/pmaxsw (rust_method: core::arch)",
    },
    // P43 Adım 2 — 155 WHT-dışı 8 bpc SSSE3 ters dönüşüm kaydı (`itx_ssse3` 140 + `itx_buyuk_ssse3` 15).
    // TX_4X4 (16)
    itx8_ssse3_aday("itxfm_add[TX_4X4][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[TX_4X4][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[TX_4X4][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][V_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][H_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][V_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_4X4][H_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    // RTX_4X8 (16)
    itx8_ssse3_aday("itxfm_add[RTX_4X8][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[RTX_4X8][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][V_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][H_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][V_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X8][H_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    // RTX_8X4 (16)
    itx8_ssse3_aday("itxfm_add[RTX_8X4][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[RTX_8X4][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][V_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][H_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][V_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X4][H_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    // TX_8X8 (16)
    itx8_ssse3_aday("itxfm_add[TX_8X8][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[TX_8X8][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[TX_8X8][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][V_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][H_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][V_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_8X8][H_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    // RTX_4X16 (16)
    itx8_ssse3_aday("itxfm_add[RTX_4X16][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[RTX_4X16][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][V_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][H_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][V_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_4X16][H_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    // RTX_16X4 (16)
    itx8_ssse3_aday("itxfm_add[RTX_16X4][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[RTX_16X4][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][V_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][H_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][V_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X4][H_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    // RTX_8X16 (16)
    itx8_ssse3_aday("itxfm_add[RTX_8X16][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[RTX_8X16][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][V_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][H_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][V_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X16][H_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    // RTX_16X8 (16)
    itx8_ssse3_aday("itxfm_add[RTX_16X8][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[RTX_16X8][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][V_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][H_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][V_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X8][H_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    // TX_16X16 (12)
    itx8_ssse3_aday("itxfm_add[TX_16X16][DCT_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][ADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][DCT_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][ADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][FLIPADST_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][DCT_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday(
        "itxfm_add[TX_16X16][FLIPADST_FLIPADST]",
        ITX8_SSSE3_KARAR_KUCUK,
    ),
    itx8_ssse3_aday("itxfm_add[TX_16X16][ADST_FLIPADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][FLIPADST_ADST]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][IDTX]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][V_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    itx8_ssse3_aday("itxfm_add[TX_16X16][H_DCT]", ITX8_SSSE3_KARAR_KUCUK),
    // Büyük bloklar (`itx_buyuk_ssse3`, 15)
    itx8_ssse3_aday("itxfm_add[RTX_8X32][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_8X32][IDTX]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_32X8][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_32X8][IDTX]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X32][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X32][IDTX]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_32X16][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_32X16][IDTX]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[TX_32X32][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[TX_32X32][IDTX]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_16X64][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_64X16][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_32X64][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[RTX_64X32][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    itx8_ssse3_aday("itxfm_add[TX_64X64][DCT_DCT]", ITX8_SSSE3_KARAR_BUYUK),
    // P43 Adım 2 — 155 WHT-dışı 8 bpc AVX2 ters dönüşüm kaydı (`itx` 140 + `itx_buyuk` 15).
    // TX_4X4 (16)
    itx8_avx2_aday("itxfm_add[TX_4X4][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[TX_4X4][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[TX_4X4][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][V_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][H_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][V_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_4X4][H_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    // RTX_4X8 (16)
    itx8_avx2_aday("itxfm_add[RTX_4X8][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[RTX_4X8][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[RTX_4X8][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][V_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][H_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][V_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X8][H_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    // RTX_8X4 (16)
    itx8_avx2_aday("itxfm_add[RTX_8X4][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[RTX_8X4][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[RTX_8X4][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][V_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][H_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][V_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X4][H_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    // TX_8X8 (16)
    itx8_avx2_aday("itxfm_add[TX_8X8][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[TX_8X8][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[TX_8X8][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][V_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][H_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][V_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_8X8][H_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    // RTX_4X16 (16)
    itx8_avx2_aday("itxfm_add[RTX_4X16][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[RTX_4X16][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[RTX_4X16][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][V_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][H_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][V_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_4X16][H_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    // RTX_16X4 (16)
    itx8_avx2_aday("itxfm_add[RTX_16X4][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[RTX_16X4][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[RTX_16X4][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][V_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][H_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][V_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X4][H_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    // RTX_8X16 (16)
    itx8_avx2_aday("itxfm_add[RTX_8X16][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[RTX_8X16][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[RTX_8X16][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][V_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][H_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][V_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_8X16][H_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    // RTX_16X8 (16)
    itx8_avx2_aday("itxfm_add[RTX_16X8][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[RTX_16X8][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[RTX_16X8][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][V_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][H_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][V_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[RTX_16X8][H_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    // TX_16X16 (12)
    itx8_avx2_aday("itxfm_add[TX_16X16][DCT_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][ADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][DCT_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][ADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][FLIPADST_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][DCT_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday(
        "itxfm_add[TX_16X16][FLIPADST_FLIPADST]",
        ITX8_AVX2_KARAR_KUCUK,
    ),
    itx8_avx2_aday("itxfm_add[TX_16X16][ADST_FLIPADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][FLIPADST_ADST]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][IDTX]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][V_DCT]", ITX8_AVX2_KARAR_KUCUK),
    itx8_avx2_aday("itxfm_add[TX_16X16][H_DCT]", ITX8_AVX2_KARAR_KUCUK),
    // Büyük bloklar (`itx_buyuk`, 15)
    itx8_avx2_aday("itxfm_add[RTX_8X32][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_8X32][IDTX]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_32X8][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_32X8][IDTX]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_16X32][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_16X32][IDTX]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_32X16][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_32X16][IDTX]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[TX_32X32][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[TX_32X32][IDTX]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_16X64][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_64X16][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_32X64][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[RTX_64X32][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    itx8_avx2_aday("itxfm_add[TX_64X64][DCT_DCT]", ITX8_AVX2_KARAR_BUYUK),
    // P43 Adım 3 — 155 WHT-dışı 8 bpc AVX512ICL ters dönüşüm kaydı (`itx_avx512` 140 + `itx_buyuk_avx512` 15).
    // TX_4X4 (16)
    itx8_avx512_aday("itxfm_add[TX_4X4][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[TX_4X4][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[TX_4X4][ADST_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][FLIPADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][V_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][H_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][V_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_4X4][H_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    // RTX_4X8 (16)
    itx8_avx512_aday("itxfm_add[RTX_4X8][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[RTX_4X8][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[RTX_4X8][ADST_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][FLIPADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][V_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][H_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][V_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X8][H_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    // RTX_8X4 (16)
    itx8_avx512_aday("itxfm_add[RTX_8X4][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[RTX_8X4][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[RTX_8X4][ADST_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][FLIPADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][V_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][H_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][V_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X4][H_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    // TX_8X8 (16)
    itx8_avx512_aday("itxfm_add[TX_8X8][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[TX_8X8][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[TX_8X8][ADST_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][FLIPADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][V_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][H_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][V_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_8X8][H_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    // RTX_4X16 (16)
    itx8_avx512_aday("itxfm_add[RTX_4X16][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[RTX_4X16][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[RTX_4X16][ADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[RTX_4X16][FLIPADST_ADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[RTX_4X16][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][V_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][H_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][V_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_4X16][H_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    // RTX_16X4 (16)
    itx8_avx512_aday("itxfm_add[RTX_16X4][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[RTX_16X4][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[RTX_16X4][ADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[RTX_16X4][FLIPADST_ADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[RTX_16X4][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][V_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][H_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][V_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X4][H_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    // RTX_8X16 (16)
    itx8_avx512_aday("itxfm_add[RTX_8X16][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[RTX_8X16][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[RTX_8X16][ADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[RTX_8X16][FLIPADST_ADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[RTX_8X16][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][V_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][H_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][V_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_8X16][H_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    // RTX_16X8 (16)
    itx8_avx512_aday("itxfm_add[RTX_16X8][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[RTX_16X8][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[RTX_16X8][ADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[RTX_16X8][FLIPADST_ADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[RTX_16X8][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][V_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][H_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][V_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[RTX_16X8][H_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    // TX_16X16 (12)
    itx8_avx512_aday("itxfm_add[TX_16X16][DCT_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_16X16][ADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_16X16][DCT_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_16X16][ADST_ADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_16X16][FLIPADST_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_16X16][DCT_FLIPADST]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday(
        "itxfm_add[TX_16X16][FLIPADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[TX_16X16][ADST_FLIPADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday(
        "itxfm_add[TX_16X16][FLIPADST_ADST]",
        ITX8_AVX512_KARAR_KUCUK,
    ),
    itx8_avx512_aday("itxfm_add[TX_16X16][IDTX]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_16X16][V_DCT]", ITX8_AVX512_KARAR_KUCUK),
    itx8_avx512_aday("itxfm_add[TX_16X16][H_DCT]", ITX8_AVX512_KARAR_KUCUK),
    // Büyük bloklar (`itx_buyuk_avx512`, 15)
    itx8_avx512_aday("itxfm_add[RTX_8X32][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_8X32][IDTX]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_32X8][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_32X8][IDTX]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_16X32][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_16X32][IDTX]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_32X16][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_32X16][IDTX]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[TX_32X32][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[TX_32X32][IDTX]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_16X64][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_64X16][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_32X64][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[RTX_64X32][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    itx8_avx512_aday("itxfm_add[TX_64X64][DCT_DCT]", ITX8_AVX512_KARAR_BUYUK),
    // =========================================================================
    // P43 Adım 3: 10 bpc WHT-dışı SSE4.1 ters dönüşüm adayları (155 kayıt:
    // 140 küçük blok `itx16_sse4` + 15 büyük blok `itx16_buyuk_sse4`;
    // dav1d `src/x86/itx.h:178` paritesi: 12 bpc skalerde kalır).
    // =========================================================================
    // TX_4X4 (16)
    itx10_sse4_aday("itxfm_add[TX_4X4][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[TX_4X4][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[TX_4X4][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][V_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][H_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][V_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_4X4][H_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    // RTX_4X8 (16)
    itx10_sse4_aday("itxfm_add[RTX_4X8][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[RTX_4X8][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[RTX_4X8][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][V_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][H_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][V_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X8][H_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    // RTX_8X4 (16)
    itx10_sse4_aday("itxfm_add[RTX_8X4][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[RTX_8X4][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[RTX_8X4][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][V_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][H_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][V_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X4][H_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    // TX_8X8 (16)
    itx10_sse4_aday("itxfm_add[TX_8X8][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[TX_8X8][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[TX_8X8][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][V_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][H_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][V_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_8X8][H_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    // RTX_4X16 (16)
    itx10_sse4_aday("itxfm_add[RTX_4X16][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[RTX_4X16][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[RTX_4X16][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][V_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][H_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][V_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_4X16][H_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    // RTX_16X4 (16)
    itx10_sse4_aday("itxfm_add[RTX_16X4][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[RTX_16X4][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[RTX_16X4][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][V_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][H_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][V_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X4][H_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    // RTX_8X16 (16)
    itx10_sse4_aday("itxfm_add[RTX_8X16][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[RTX_8X16][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[RTX_8X16][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][V_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][H_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][V_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_8X16][H_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    // RTX_16X8 (16)
    itx10_sse4_aday("itxfm_add[RTX_16X8][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[RTX_16X8][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[RTX_16X8][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][V_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][H_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][V_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[RTX_16X8][H_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    // TX_16X16 (12)
    itx10_sse4_aday("itxfm_add[TX_16X16][DCT_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][ADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][DCT_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][ADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][FLIPADST_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][DCT_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday(
        "itxfm_add[TX_16X16][FLIPADST_FLIPADST]",
        ITX10_SSE4_KARAR_KUCUK,
    ),
    itx10_sse4_aday("itxfm_add[TX_16X16][ADST_FLIPADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][FLIPADST_ADST]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][IDTX]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][V_DCT]", ITX10_SSE4_KARAR_KUCUK),
    itx10_sse4_aday("itxfm_add[TX_16X16][H_DCT]", ITX10_SSE4_KARAR_KUCUK),
    // Büyük bloklar (`itx16_buyuk_sse4`, 15)
    itx10_sse4_aday("itxfm_add[RTX_8X32][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_8X32][IDTX]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_32X8][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_32X8][IDTX]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_16X32][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_16X32][IDTX]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_32X16][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_32X16][IDTX]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[TX_32X32][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[TX_32X32][IDTX]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_16X64][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_64X16][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_32X64][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[RTX_64X32][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    itx10_sse4_aday("itxfm_add[TX_64X64][DCT_DCT]", ITX10_SSE4_KARAR_BUYUK),
    // =========================================================================
    // P44 Adım 1: 10 bpc WHT-dışı AVX2 ters dönüşüm adayları (155 kayıt:
    // 140 küçük blok `itx16_10` + 15 büyük blok `itx16_10_buyuk`).
    // =========================================================================
    // TX_4X4 (16)
    itx10_avx2_aday("itxfm_add[TX_4X4][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[TX_4X4][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[TX_4X4][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][V_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][H_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][V_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_4X4][H_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    // RTX_4X8 (16)
    itx10_avx2_aday("itxfm_add[RTX_4X8][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[RTX_4X8][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[RTX_4X8][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][V_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][H_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][V_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X8][H_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    // RTX_8X4 (16)
    itx10_avx2_aday("itxfm_add[RTX_8X4][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[RTX_8X4][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[RTX_8X4][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][V_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][H_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][V_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X4][H_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    // TX_8X8 (16)
    itx10_avx2_aday("itxfm_add[TX_8X8][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[TX_8X8][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[TX_8X8][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][V_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][H_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][V_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_8X8][H_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    // RTX_4X16 (16)
    itx10_avx2_aday("itxfm_add[RTX_4X16][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[RTX_4X16][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[RTX_4X16][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][V_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][H_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][V_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_4X16][H_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    // RTX_16X4 (16)
    itx10_avx2_aday("itxfm_add[RTX_16X4][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[RTX_16X4][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[RTX_16X4][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][V_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][H_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][V_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X4][H_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    // RTX_8X16 (16)
    itx10_avx2_aday("itxfm_add[RTX_8X16][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[RTX_8X16][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[RTX_8X16][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][V_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][H_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][V_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_8X16][H_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    // RTX_16X8 (16)
    itx10_avx2_aday("itxfm_add[RTX_16X8][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[RTX_16X8][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[RTX_16X8][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][V_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][H_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][V_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[RTX_16X8][H_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    // TX_16X16 (12)
    itx10_avx2_aday("itxfm_add[TX_16X16][DCT_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][ADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][DCT_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][ADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][FLIPADST_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][DCT_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday(
        "itxfm_add[TX_16X16][FLIPADST_FLIPADST]",
        ITX10_AVX2_KARAR_KUCUK,
    ),
    itx10_avx2_aday("itxfm_add[TX_16X16][ADST_FLIPADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][FLIPADST_ADST]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][IDTX]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][V_DCT]", ITX10_AVX2_KARAR_KUCUK),
    itx10_avx2_aday("itxfm_add[TX_16X16][H_DCT]", ITX10_AVX2_KARAR_KUCUK),
    // Büyük bloklar (`itx16_10_buyuk`, 15)
    itx10_avx2_aday("itxfm_add[RTX_8X32][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_8X32][IDTX]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_32X8][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_32X8][IDTX]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_16X32][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_16X32][IDTX]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_32X16][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_32X16][IDTX]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[TX_32X32][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[TX_32X32][IDTX]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_16X64][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_64X16][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_32X64][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[RTX_64X32][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    itx10_avx2_aday("itxfm_add[TX_64X64][DCT_DCT]", ITX10_AVX2_KARAR_BUYUK),
    // =========================================================================
    // P44 Adım 1 (Paket B): 12 bpc WHT-dışı AVX2 ters dönüşüm adayları (147 kayıt:
    // 140 küçük blok `itx16_12` + 7 büyük blok `itx16_12_buyuk`;
    // dav1d `src/x86/itx.h:249-263` paritesi: en büyük 8 DCT_DCT 12 bpc'de skalerde kalır).
    // =========================================================================
    // TX_4X4 (16)
    itx12_avx2_aday("itxfm_add[TX_4X4][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[TX_4X4][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[TX_4X4][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][V_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][H_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][V_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_4X4][H_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    // RTX_4X8 (16)
    itx12_avx2_aday("itxfm_add[RTX_4X8][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[RTX_4X8][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[RTX_4X8][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][V_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][H_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][V_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X8][H_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    // RTX_8X4 (16)
    itx12_avx2_aday("itxfm_add[RTX_8X4][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[RTX_8X4][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[RTX_8X4][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][V_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][H_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][V_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X4][H_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    // TX_8X8 (16)
    itx12_avx2_aday("itxfm_add[TX_8X8][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[TX_8X8][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[TX_8X8][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][V_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][H_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][V_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_8X8][H_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    // RTX_4X16 (16)
    itx12_avx2_aday("itxfm_add[RTX_4X16][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[RTX_4X16][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[RTX_4X16][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][V_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][H_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][V_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_4X16][H_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    // RTX_16X4 (16)
    itx12_avx2_aday("itxfm_add[RTX_16X4][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[RTX_16X4][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[RTX_16X4][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][V_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][H_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][V_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X4][H_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    // RTX_8X16 (16)
    itx12_avx2_aday("itxfm_add[RTX_8X16][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[RTX_8X16][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[RTX_8X16][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][V_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][H_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][V_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_8X16][H_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    // RTX_16X8 (16)
    itx12_avx2_aday("itxfm_add[RTX_16X8][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[RTX_16X8][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[RTX_16X8][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][V_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][H_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][V_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[RTX_16X8][H_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    // TX_16X16 (12)
    itx12_avx2_aday("itxfm_add[TX_16X16][DCT_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][ADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][DCT_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][ADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][FLIPADST_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][DCT_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday(
        "itxfm_add[TX_16X16][FLIPADST_FLIPADST]",
        ITX12_AVX2_KARAR_KUCUK,
    ),
    itx12_avx2_aday("itxfm_add[TX_16X16][ADST_FLIPADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][FLIPADST_ADST]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][IDTX]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][V_DCT]", ITX12_AVX2_KARAR_KUCUK),
    itx12_avx2_aday("itxfm_add[TX_16X16][H_DCT]", ITX12_AVX2_KARAR_KUCUK),
    // Büyük bloklar (`itx16_12_buyuk`, 15: 5 IDTX + 10 DCT_DCT)
    itx12_avx2_aday("itxfm_add[RTX_8X32][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_8X32][IDTX]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_32X8][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_32X8][IDTX]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_16X32][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_16X32][IDTX]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_32X16][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_32X16][IDTX]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[TX_32X32][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[TX_32X32][IDTX]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_16X64][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_32X64][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_64X16][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[RTX_64X32][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    itx12_avx2_aday("itxfm_add[TX_64X64][DCT_DCT]", ITX12_AVX2_KARAR_BUYUK),
    // =========================================================================
    // P44 Adım 2: 10 bpc WHT-dışı (`w >= 8 && h >= 8`) AVX512ICL ters dönüşüm adayları (75 kayıt:
    // 60 küçük blok `itx16_avx512` + 15 büyük blok `itx16_buyuk_avx512`;
    // dav1d `src/x86/itx.h:289-304` paritesi: yalnız `bpc == 10` ve `w, h >= 8`).
    // =========================================================================
    // TX_8X8 (16)
    itx10_avx512_aday("itxfm_add[TX_8X8][DCT_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][ADST_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][DCT_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][ADST_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][FLIPADST_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][DCT_FLIPADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday(
        "itxfm_add[TX_8X8][FLIPADST_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday("itxfm_add[TX_8X8][ADST_FLIPADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][FLIPADST_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][IDTX]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][V_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][H_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][V_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][H_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][V_FLIPADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_8X8][H_FLIPADST]", ITX10_AVX512_KARAR_KUCUK),
    // RTX_8X16 (16)
    itx10_avx512_aday("itxfm_add[RTX_8X16][DCT_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][ADST_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][DCT_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][ADST_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday(
        "itxfm_add[RTX_8X16][FLIPADST_DCT]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[RTX_8X16][DCT_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[RTX_8X16][FLIPADST_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[RTX_8X16][ADST_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[RTX_8X16][FLIPADST_ADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday("itxfm_add[RTX_8X16][IDTX]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][V_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][H_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][V_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][H_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][V_FLIPADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_8X16][H_FLIPADST]", ITX10_AVX512_KARAR_KUCUK),
    // RTX_16X8 (16)
    itx10_avx512_aday("itxfm_add[RTX_16X8][DCT_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][ADST_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][DCT_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][ADST_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday(
        "itxfm_add[RTX_16X8][FLIPADST_DCT]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[RTX_16X8][DCT_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[RTX_16X8][FLIPADST_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[RTX_16X8][ADST_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[RTX_16X8][FLIPADST_ADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday("itxfm_add[RTX_16X8][IDTX]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][V_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][H_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][V_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][H_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][V_FLIPADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[RTX_16X8][H_FLIPADST]", ITX10_AVX512_KARAR_KUCUK),
    // TX_16X16 (12)
    itx10_avx512_aday("itxfm_add[TX_16X16][DCT_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_16X16][ADST_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_16X16][DCT_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_16X16][ADST_ADST]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday(
        "itxfm_add[TX_16X16][FLIPADST_DCT]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[TX_16X16][DCT_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[TX_16X16][FLIPADST_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[TX_16X16][ADST_FLIPADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday(
        "itxfm_add[TX_16X16][FLIPADST_ADST]",
        ITX10_AVX512_KARAR_KUCUK,
    ),
    itx10_avx512_aday("itxfm_add[TX_16X16][IDTX]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_16X16][V_DCT]", ITX10_AVX512_KARAR_KUCUK),
    itx10_avx512_aday("itxfm_add[TX_16X16][H_DCT]", ITX10_AVX512_KARAR_KUCUK),
    // Büyük bloklar (`itx16_buyuk_avx512`, 15)
    itx10_avx512_aday("itxfm_add[RTX_8X32][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_8X32][IDTX]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_32X8][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_32X8][IDTX]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_16X32][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_16X32][IDTX]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_32X16][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_32X16][IDTX]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[TX_32X32][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[TX_32X32][IDTX]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_16X64][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_64X16][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_32X64][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[RTX_64X32][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    itx10_avx512_aday("itxfm_add[TX_64X64][DCT_DCT]", ITX10_AVX512_KARAR_BUYUK),
    // -------------------------------------------------------------------------
    // P45 Adım 1: Bit derinliğinden bağımsız `pal/pal_idx_finish` (6 matris kaydı:
    // D03040/D03041 SSSE3, D03036/D03037 AVX2, D03038/D03039 AVX512ICL; `bit_depth == 0`).
    // -------------------------------------------------------------------------
    IsaCandidate {
        family: "pal",
        field: "pal_idx_finish",
        bit_depth: 0,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "varsayılan (P45 Adım 1, 6/6 F4-PAL — D03040/D03041): 128-bit SSSE3 `pmaddubsw` + `packuswb` + `pshufb` (`src/x86/pal.asm:62-272`); iki CCD'de (çekirdek 6/14) Rust SSSE3 / dav1d SSSE3 ortanca 1,27× / 1,28× (bw≥32'de 0,88×–1,26×), Rust skalere göre 0,04×–0,36× (2,8–26× hızlanma)",
    },
    IsaCandidate {
        family: "pal",
        field: "pal_idx_finish",
        bit_depth: 0,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "varsayılan (P45 Adım 1, 6/6 F4-PAL — D03036/D03037): 128/256-bit AVX2 `vpmaddubsw` + `vpackuswb` + `vpermq` + `vpshufb` (`src/x86/pal.asm:275-470`); iki CCD'de (çekirdek 6/14) Rust AVX2 / dav1d AVX2 bw≥32'de 0,89×–1,41× (9 hücre ortancası 1,69× / 1,67×), Rust SSSE3'e göre bw≥32'de 0,70×–0,73×",
    },
    IsaCandidate {
        family: "pal",
        field: "pal_idx_finish",
        bit_depth: 0,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "varsayılan (P45 Adım 1, 6/6 F4-PAL — D03038/D03039): 128/512-bit AVX512ICL `vpmaddubsw` + `vpmovwb`/`vpermt2b` + `vpermb` (`src/x86/pal.asm:471-640`; `SLOW_GATHER` engeli yok, 0 gather/scatter); iki CCD'de (çekirdek 6/14) Rust AVX512ICL / dav1d AVX512ICL bw≥32'de 1,41×–1,67× (9 hücre ortancası 1,67× / 1,65×), Rust AVX2'ye göre bw≥16'da 0,69×–0,97×",
    },
    // -------------------------------------------------------------------------
    // P45 Adım 2: 8 ve 16 bpc (10/12 bit ortak) intra tahmin (`ipred`, 112 kayıt / 165 aday:
    // 8 bpc 22 SSSE3 + 22 AVX2 + 15 AVX512ICL = 59; 16 bpc 22 SSSE3 + 22 AVX2 + 9 AVX512ICL = 53 kayıt / 106 aday).
    // -------------------------------------------------------------------------
    ipred8_ssse3_aday("intra_pred[DC_PRED]"),
    ipred8_ssse3_aday("intra_pred[VERT_PRED]"),
    ipred8_ssse3_aday("intra_pred[HOR_PRED]"),
    ipred8_ssse3_aday("intra_pred[LEFT_DC_PRED]"),
    ipred8_ssse3_aday("intra_pred[TOP_DC_PRED]"),
    ipred8_ssse3_aday("intra_pred[DC_128_PRED]"),
    ipred8_ssse3_aday("intra_pred[Z1_PRED]"),
    ipred8_ssse3_aday("intra_pred[Z2_PRED]"),
    ipred8_ssse3_aday("intra_pred[Z3_PRED]"),
    ipred8_ssse3_aday("intra_pred[SMOOTH_PRED]"),
    ipred8_ssse3_aday("intra_pred[SMOOTH_V_PRED]"),
    ipred8_ssse3_aday("intra_pred[SMOOTH_H_PRED]"),
    ipred8_ssse3_aday("intra_pred[PAETH_PRED]"),
    ipred8_ssse3_aday("intra_pred[FILTER_PRED]"),
    ipred8_ssse3_aday("cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]"),
    ipred8_ssse3_aday("cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]"),
    ipred8_ssse3_aday("cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]"),
    ipred8_ssse3_aday("cfl_pred[DC_PRED]"),
    ipred8_ssse3_aday("cfl_pred[LEFT_DC_PRED]"),
    ipred8_ssse3_aday("cfl_pred[TOP_DC_PRED]"),
    ipred8_ssse3_aday("cfl_pred[DC_128_PRED]"),
    ipred8_ssse3_aday("pal_pred"),
    ipred8_avx2_aday("intra_pred[DC_PRED]"),
    ipred8_avx2_aday("intra_pred[VERT_PRED]"),
    ipred8_avx2_aday("intra_pred[HOR_PRED]"),
    ipred8_avx2_aday("intra_pred[LEFT_DC_PRED]"),
    ipred8_avx2_aday("intra_pred[TOP_DC_PRED]"),
    ipred8_avx2_aday("intra_pred[DC_128_PRED]"),
    ipred8_avx2_aday("intra_pred[Z1_PRED]"),
    ipred8_avx2_aday("intra_pred[Z2_PRED]"),
    ipred8_avx2_aday("intra_pred[Z3_PRED]"),
    ipred8_avx2_aday("intra_pred[SMOOTH_PRED]"),
    ipred8_avx2_aday("intra_pred[SMOOTH_V_PRED]"),
    ipred8_avx2_aday("intra_pred[SMOOTH_H_PRED]"),
    ipred8_avx2_aday("intra_pred[PAETH_PRED]"),
    ipred8_avx2_aday("intra_pred[FILTER_PRED]"),
    ipred8_avx2_aday("cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]"),
    ipred8_avx2_aday("cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]"),
    ipred8_avx2_aday("cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]"),
    ipred8_avx2_aday("cfl_pred[DC_PRED]"),
    ipred8_avx2_aday("cfl_pred[LEFT_DC_PRED]"),
    ipred8_avx2_aday("cfl_pred[TOP_DC_PRED]"),
    ipred8_avx2_aday("cfl_pred[DC_128_PRED]"),
    ipred8_avx2_aday("pal_pred"),
    ipred8_avx512_aday("intra_pred[DC_PRED]"),
    ipred8_avx512_aday("intra_pred[VERT_PRED]"),
    ipred8_avx512_aday("intra_pred[HOR_PRED]"),
    ipred8_avx512_aday("intra_pred[LEFT_DC_PRED]"),
    ipred8_avx512_aday("intra_pred[TOP_DC_PRED]"),
    ipred8_avx512_aday("intra_pred[DC_128_PRED]"),
    ipred8_avx512_aday("intra_pred[Z1_PRED]"),
    ipred8_avx512_aday("intra_pred[Z2_PRED]"),
    ipred8_avx512_aday("intra_pred[Z3_PRED]"),
    ipred8_avx512_aday("intra_pred[SMOOTH_PRED]"),
    ipred8_avx512_aday("intra_pred[SMOOTH_V_PRED]"),
    ipred8_avx512_aday("intra_pred[SMOOTH_H_PRED]"),
    ipred8_avx512_aday("intra_pred[PAETH_PRED]"),
    ipred8_avx512_aday("intra_pred[FILTER_PRED]"),
    ipred8_avx512_aday("pal_pred"),
    ipred16_ssse3_aday(10, "intra_pred[DC_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[VERT_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[HOR_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[LEFT_DC_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[TOP_DC_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[DC_128_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[Z1_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[Z2_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[Z3_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[SMOOTH_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[SMOOTH_V_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[SMOOTH_H_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[PAETH_PRED]"),
    ipred16_ssse3_aday(10, "intra_pred[FILTER_PRED]"),
    ipred16_ssse3_aday(10, "cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]"),
    ipred16_ssse3_aday(10, "cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]"),
    ipred16_ssse3_aday(10, "cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]"),
    ipred16_ssse3_aday(10, "cfl_pred[DC_PRED]"),
    ipred16_ssse3_aday(10, "cfl_pred[LEFT_DC_PRED]"),
    ipred16_ssse3_aday(10, "cfl_pred[TOP_DC_PRED]"),
    ipred16_ssse3_aday(10, "cfl_pred[DC_128_PRED]"),
    ipred16_ssse3_aday(10, "pal_pred"),
    ipred16_ssse3_aday(12, "intra_pred[DC_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[VERT_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[HOR_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[LEFT_DC_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[TOP_DC_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[DC_128_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[Z1_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[Z2_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[Z3_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[SMOOTH_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[SMOOTH_V_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[SMOOTH_H_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[PAETH_PRED]"),
    ipred16_ssse3_aday(12, "intra_pred[FILTER_PRED]"),
    ipred16_ssse3_aday(12, "cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]"),
    ipred16_ssse3_aday(12, "cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]"),
    ipred16_ssse3_aday(12, "cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]"),
    ipred16_ssse3_aday(12, "cfl_pred[DC_PRED]"),
    ipred16_ssse3_aday(12, "cfl_pred[LEFT_DC_PRED]"),
    ipred16_ssse3_aday(12, "cfl_pred[TOP_DC_PRED]"),
    ipred16_ssse3_aday(12, "cfl_pred[DC_128_PRED]"),
    ipred16_ssse3_aday(12, "pal_pred"),
    ipred16_avx2_aday(10, "intra_pred[DC_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[VERT_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[HOR_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[LEFT_DC_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[TOP_DC_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[DC_128_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[Z1_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[Z2_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[Z3_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[SMOOTH_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[SMOOTH_V_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[SMOOTH_H_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[PAETH_PRED]"),
    ipred16_avx2_aday(10, "intra_pred[FILTER_PRED]"),
    ipred16_avx2_aday(10, "cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]"),
    ipred16_avx2_aday(10, "cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]"),
    ipred16_avx2_aday(10, "cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]"),
    ipred16_avx2_aday(10, "cfl_pred[DC_PRED]"),
    ipred16_avx2_aday(10, "cfl_pred[LEFT_DC_PRED]"),
    ipred16_avx2_aday(10, "cfl_pred[TOP_DC_PRED]"),
    ipred16_avx2_aday(10, "cfl_pred[DC_128_PRED]"),
    ipred16_avx2_aday(10, "pal_pred"),
    ipred16_avx2_aday(12, "intra_pred[DC_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[VERT_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[HOR_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[LEFT_DC_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[TOP_DC_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[DC_128_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[Z1_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[Z2_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[Z3_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[SMOOTH_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[SMOOTH_V_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[SMOOTH_H_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[PAETH_PRED]"),
    ipred16_avx2_aday(12, "intra_pred[FILTER_PRED]"),
    ipred16_avx2_aday(12, "cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]"),
    ipred16_avx2_aday(12, "cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]"),
    ipred16_avx2_aday(12, "cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]"),
    ipred16_avx2_aday(12, "cfl_pred[DC_PRED]"),
    ipred16_avx2_aday(12, "cfl_pred[LEFT_DC_PRED]"),
    ipred16_avx2_aday(12, "cfl_pred[TOP_DC_PRED]"),
    ipred16_avx2_aday(12, "cfl_pred[DC_128_PRED]"),
    ipred16_avx2_aday(12, "pal_pred"),
    ipred16_avx512_aday(10, "intra_pred[Z1_PRED]"),
    ipred16_avx512_aday(10, "intra_pred[Z2_PRED]"),
    ipred16_avx512_aday(10, "intra_pred[Z3_PRED]"),
    ipred16_avx512_aday(10, "intra_pred[SMOOTH_PRED]"),
    ipred16_avx512_aday(10, "intra_pred[SMOOTH_V_PRED]"),
    ipred16_avx512_aday(10, "intra_pred[SMOOTH_H_PRED]"),
    ipred16_avx512_aday(10, "intra_pred[PAETH_PRED]"),
    ipred16_avx512_aday(10, "intra_pred[FILTER_PRED]"),
    ipred16_avx512_aday(10, "pal_pred"),
    ipred16_avx512_aday(12, "intra_pred[Z1_PRED]"),
    ipred16_avx512_aday(12, "intra_pred[Z2_PRED]"),
    ipred16_avx512_aday(12, "intra_pred[Z3_PRED]"),
    ipred16_avx512_aday(12, "intra_pred[SMOOTH_PRED]"),
    ipred16_avx512_aday(12, "intra_pred[SMOOTH_V_PRED]"),
    ipred16_avx512_aday(12, "intra_pred[SMOOTH_H_PRED]"),
    ipred16_avx512_aday(12, "intra_pred[PAETH_PRED]"),
    ipred16_avx512_aday(12, "intra_pred[FILTER_PRED]"),
    ipred16_avx512_aday(12, "pal_pred"),
    // -------------------------------------------------------------------------
    // P46 Adım 1: Bit derinliğinden bağımsız `refmvs` (`bit_depth == 0`, 14 matris kaydı / 7 aday:
    // D03054/55 `splat_mv` SSE2, D03048/49 `save_tmvs` SSSE3, D03042/43 `load_tmvs` SSE4.1,
    // D03044/45 `save_tmvs` AVX2, D03050/51 `splat_mv` AVX2, D03046/47 `save_tmvs` AVX512ICL,
    // D03052/53 `splat_mv` AVX512ICL).
    // -------------------------------------------------------------------------
    IsaCandidate {
        family: "refmvs",
        field: "splat_mv",
        bit_depth: 0,
        implementation: "sse2",
        requires: CpuFlags::SSE2,
        default: true,
        decision: "varsayılan (P46 Adım 1, 14/14 F4-REFMVS — D03054/D03055): 128-bit SSE2 `3 × XMM` permüteli `RefBlock` (12 B) yazımı (`src/x86/refmvs.asm:298-360`; c6 `0.62×..1.77×`, c14 `0.62×..1.79×` vs dav1d SSE2; `0.49×..0.96×` vs skaler)",
    },
    IsaCandidate {
        family: "refmvs",
        field: "save_tmvs",
        bit_depth: 0,
        implementation: "ssse3",
        requires: CpuFlags::SSSE3,
        default: true,
        decision: "varsayılan (P46 Adım 1, 14/14 F4-REFMVS — D03048/D03049): 128-bit SSSE3 `pshufb` + `pabsw` + `pcmpgtd` 2 adaylı `save_tmvs` (`src/x86/refmvs.asm:128-296`; c6 `0.56×..0.90×`, c14 `0.56×..0.92×` vs skaler; w<=16 `1.24×..1.45×` vs dav1d SSSE3)",
    },
    IsaCandidate {
        family: "refmvs",
        field: "load_tmvs",
        bit_depth: 0,
        implementation: "sse4",
        requires: CpuFlags::SSE4_1.union(CpuFlags::SSSE3),
        default: true,
        decision: "varsayılan (P46 Adım 1, 14/14 F4-REFMVS — D03042/D03043): 128-bit SSE4.1 `pmulld` + `pmuldq` + `packssdw` + `psignd` temporal MV projeksiyonu (`src/x86/refmvs.asm:362-596`; c6 `1.17×..1.27×`, c14 `1.17×..1.28×` vs dav1d SSE4.1; `0.44×..0.51×` vs skaler)",
    },
    IsaCandidate {
        family: "refmvs",
        field: "save_tmvs",
        bit_depth: 0,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "varsayılan (P46 Adım 1, 14/14 F4-REFMVS — D03044/D03045): 256-bit AVX2 `vpshufb` + `vpabsw` + `vpcmpgtd` 2 adaylı `save_tmvs` (`src/x86/refmvs.asm:597-704`; w<=16 c6 `1.34×..1.46×`, c14 `1.35×..1.46×` vs dav1d AVX2)",
    },
    IsaCandidate {
        family: "refmvs",
        field: "splat_mv",
        bit_depth: 0,
        implementation: "avx2",
        requires: CpuFlags::AVX2,
        default: true,
        decision: "varsayılan (P46 Adım 1, 14/14 F4-REFMVS — D03050/D03051): 256-bit AVX2 `3 × YMM` permüteli `RefBlock` (12 B) yazımı (`src/x86/refmvs.asm:705-758`; w>=8 c6 `1.01×..1.64×`, c14 `0.99×..1.45×` vs dav1d AVX2)",
    },
    IsaCandidate {
        family: "refmvs",
        field: "save_tmvs",
        bit_depth: 0,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "varsayılan (P46 Adım 1, 14/14 F4-REFMVS — D03046/D03047): 512-bit AVX512ICL `vpshufb` + `vpabsw` + `vpternlogd` 4 adaylı `save_tmvs` (`src/x86/refmvs.asm:759-876`; `SLOW_GATHER` engeli yok, 0 gather/scatter; w=64 c6 `0.96×..0.99×` vs AVX2)",
    },
    IsaCandidate {
        family: "refmvs",
        field: "splat_mv",
        bit_depth: 0,
        implementation: "avx512icl",
        requires: CpuFlags::AVX512ICL,
        default: true,
        decision: "varsayılan (P46 Adım 1, 14/14 F4-REFMVS — D03052/D03053): 512-bit AVX512ICL `3 × ZMM` permüteli `RefBlock` (12 B) yazımı (`src/x86/refmvs.asm:878-952`; `SLOW_GATHER` engeli yok, 0 gather/scatter; w>=8 c6 `1.26×..1.73×`, c14 `1.16×..1.57×` vs dav1d AVX512ICL)",
    },
    // -------------------------------------------------------------------------
    // P46 Adım 2: 8 ve 16 bpc (`10/12` bit ortak) Deblocking Loop Filter (`loopfilter`,
    // 24 kayıt / 36 aday: 8 bpc 4 SSSE3 + 4 AVX2 + 4 AVX512ICL = 12;
    // 16 bpc 4 SSSE3 + 4 AVX2 + 4 AVX512ICL = 12 kayıt / 24 aday `@10+@12`).
    // -------------------------------------------------------------------------
    lf_aday(
        8,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[0][0]",
        LF8_SSSE3_KARAR,
    ),
    lf_aday(
        8,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[0][1]",
        LF8_SSSE3_KARAR,
    ),
    lf_aday(
        8,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[1][0]",
        LF8_SSSE3_KARAR,
    ),
    lf_aday(
        8,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[1][1]",
        LF8_SSSE3_KARAR,
    ),
    lf_aday(
        8,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[0][0]",
        LF8_AVX2_KARAR,
    ),
    lf_aday(
        8,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[0][1]",
        LF8_AVX2_KARAR,
    ),
    lf_aday(
        8,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[1][0]",
        LF8_AVX2_KARAR,
    ),
    lf_aday(
        8,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[1][1]",
        LF8_AVX2_KARAR,
    ),
    lf_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[0][0]",
        LF8_AVX512_KARAR,
    ),
    lf_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[0][1]",
        LF8_AVX512_KARAR,
    ),
    lf_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[1][0]",
        LF8_AVX512_KARAR,
    ),
    lf_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[1][1]",
        LF8_AVX512_KARAR,
    ),
    lf_aday(
        10,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[0][0]",
        LF16_SSSE3_KARAR,
    ),
    lf_aday(
        10,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[0][1]",
        LF16_SSSE3_KARAR,
    ),
    lf_aday(
        10,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[1][0]",
        LF16_SSSE3_KARAR,
    ),
    lf_aday(
        10,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[1][1]",
        LF16_SSSE3_KARAR,
    ),
    lf_aday(
        12,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[0][0]",
        LF16_SSSE3_KARAR,
    ),
    lf_aday(
        12,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[0][1]",
        LF16_SSSE3_KARAR,
    ),
    lf_aday(
        12,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[1][0]",
        LF16_SSSE3_KARAR,
    ),
    lf_aday(
        12,
        "ssse3",
        CpuFlags::SSSE3,
        "loop_filter_sb[1][1]",
        LF16_SSSE3_KARAR,
    ),
    lf_aday(
        10,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[0][0]",
        LF16_AVX2_KARAR,
    ),
    lf_aday(
        10,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[0][1]",
        LF16_AVX2_KARAR,
    ),
    lf_aday(
        10,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[1][0]",
        LF16_AVX2_KARAR,
    ),
    lf_aday(
        10,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[1][1]",
        LF16_AVX2_KARAR,
    ),
    lf_aday(
        12,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[0][0]",
        LF16_AVX2_KARAR,
    ),
    lf_aday(
        12,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[0][1]",
        LF16_AVX2_KARAR,
    ),
    lf_aday(
        12,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[1][0]",
        LF16_AVX2_KARAR,
    ),
    lf_aday(
        12,
        "avx2",
        CpuFlags::AVX2,
        "loop_filter_sb[1][1]",
        LF16_AVX2_KARAR,
    ),
    lf_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[0][0]",
        LF16_AVX512_KARAR,
    ),
    lf_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[0][1]",
        LF16_AVX512_KARAR,
    ),
    lf_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[1][0]",
        LF16_AVX512_KARAR,
    ),
    lf_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[1][1]",
        LF16_AVX512_KARAR,
    ),
    lf_aday(
        12,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[0][0]",
        LF16_AVX512_KARAR,
    ),
    lf_aday(
        12,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[0][1]",
        LF16_AVX512_KARAR,
    ),
    lf_aday(
        12,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[1][0]",
        LF16_AVX512_KARAR,
    ),
    lf_aday(
        12,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "loop_filter_sb[1][1]",
        LF16_AVX512_KARAR,
    ),
    // -------------------------------------------------------------------------
    // P46 Adım 3: 8 ve 16 bpc (`10/12` bit Wiener, `10` bit SGR) Loop Restoration (`looprestoration`,
    // 31 kayıt / 38 aday: 8 bpc 2 SSE2 + 5 SSSE3 + 5 AVX2 + 4 AVX512ICL + wiener[1]@8 AVX512ICL eş-bağlama = 17;
    // 16 bpc 5 SSSE3 + 5 AVX2 + 5 AVX512ICL = 15 kayıt / 21 aday: 12 Wiener `@10+@12` + 9 SGR `@10`).
    // -------------------------------------------------------------------------
    lr_aday(8, "sse2", CpuFlags::SSE2, "wiener[0]", LR8_SSE2_KARAR),
    lr_aday(8, "sse2", CpuFlags::SSE2, "wiener[1]", LR8_SSE2_KARAR),
    lr_aday(8, "ssse3", CpuFlags::SSSE3, "wiener[0]", LR8_SSSE3_KARAR),
    lr_aday(8, "ssse3", CpuFlags::SSSE3, "wiener[1]", LR8_SSSE3_KARAR),
    lr_aday(8, "ssse3", CpuFlags::SSSE3, "sgr[0]", LR8_SSSE3_KARAR),
    lr_aday(8, "ssse3", CpuFlags::SSSE3, "sgr[1]", LR8_SSSE3_KARAR),
    lr_aday(8, "ssse3", CpuFlags::SSSE3, "sgr[2]", LR8_SSSE3_KARAR),
    lr_aday(8, "avx2", CpuFlags::AVX2, "wiener[0]", LR8_AVX2_KARAR),
    lr_aday(8, "avx2", CpuFlags::AVX2, "wiener[1]", LR8_AVX2_KARAR),
    lr_aday(8, "avx2", CpuFlags::AVX2, "sgr[0]", LR8_AVX2_KARAR),
    lr_aday(8, "avx2", CpuFlags::AVX2, "sgr[1]", LR8_AVX2_KARAR),
    lr_aday(8, "avx2", CpuFlags::AVX2, "sgr[2]", LR8_AVX2_KARAR),
    lr_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "wiener[0]",
        LR8_AVX512_KARAR,
    ),
    lr_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "wiener[1]",
        LR8_AVX512_KARAR,
    ),
    lr_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "sgr[0]",
        LR8_AVX512_KARAR,
    ),
    lr_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "sgr[1]",
        LR8_AVX512_KARAR,
    ),
    lr_aday(
        8,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "sgr[2]",
        LR8_AVX512_KARAR,
    ),
    lr_aday(10, "ssse3", CpuFlags::SSSE3, "wiener[0]", LR16_SSSE3_KARAR),
    lr_aday(10, "ssse3", CpuFlags::SSSE3, "wiener[1]", LR16_SSSE3_KARAR),
    lr_aday(10, "ssse3", CpuFlags::SSSE3, "sgr[0]", LR16_SSSE3_KARAR),
    lr_aday(10, "ssse3", CpuFlags::SSSE3, "sgr[1]", LR16_SSSE3_KARAR),
    lr_aday(10, "ssse3", CpuFlags::SSSE3, "sgr[2]", LR16_SSSE3_KARAR),
    lr_aday(12, "ssse3", CpuFlags::SSSE3, "wiener[0]", LR16_SSSE3_KARAR),
    lr_aday(12, "ssse3", CpuFlags::SSSE3, "wiener[1]", LR16_SSSE3_KARAR),
    lr_aday(10, "avx2", CpuFlags::AVX2, "wiener[0]", LR16_AVX2_KARAR),
    lr_aday(10, "avx2", CpuFlags::AVX2, "wiener[1]", LR16_AVX2_KARAR),
    lr_aday(10, "avx2", CpuFlags::AVX2, "sgr[0]", LR16_AVX2_KARAR),
    lr_aday(10, "avx2", CpuFlags::AVX2, "sgr[1]", LR16_AVX2_KARAR),
    lr_aday(10, "avx2", CpuFlags::AVX2, "sgr[2]", LR16_AVX2_KARAR),
    lr_aday(12, "avx2", CpuFlags::AVX2, "wiener[0]", LR16_AVX2_KARAR),
    lr_aday(12, "avx2", CpuFlags::AVX2, "wiener[1]", LR16_AVX2_KARAR),
    lr_aday(12, "avx2", CpuFlags::AVX2, "sgr[0]", LR16_AVX2_KARAR),
    lr_aday(12, "avx2", CpuFlags::AVX2, "sgr[1]", LR16_AVX2_KARAR),
    lr_aday(12, "avx2", CpuFlags::AVX2, "sgr[2]", LR16_AVX2_KARAR),
    lr_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "wiener[0]",
        LR16_AVX512_KARAR,
    ),
    lr_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "wiener[1]",
        LR16_AVX512_KARAR,
    ),
    lr_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "sgr[0]",
        LR16_AVX512_KARAR,
    ),
    lr_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "sgr[1]",
        LR16_AVX512_KARAR,
    ),
    lr_aday(
        10,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "sgr[2]",
        LR16_AVX512_KARAR,
    ),
    lr_aday(
        12,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "wiener[0]",
        LR16_AVX512_KARAR,
    ),
    lr_aday(
        12,
        "avx512icl",
        CpuFlags::AVX512ICL,
        "wiener[1]",
        LR16_AVX512_KARAR,
    ),
    // -------------------------------------------------------------------------
    // P47 Adım 1: Bit derinliğinden bağımsız (`bit_depth == 0`) MSAC (`msac`,
    // 8 sembol / 7 yuva / 8 aday: 7 SSE2 + 1 AVX2 `decode_symbol_adapt16`).
    // -------------------------------------------------------------------------
    msac_aday(
        "sse2",
        CpuFlags::SSE2,
        "decode_symbol_adapt4",
        MSAC_SSE2_KARAR,
    ),
    msac_aday(
        "sse2",
        CpuFlags::SSE2,
        "decode_symbol_adapt8",
        MSAC_SSE2_KARAR,
    ),
    msac_aday(
        "sse2",
        CpuFlags::SSE2,
        "decode_symbol_adapt16",
        MSAC_SSE2_KARAR,
    ),
    msac_aday(
        "avx2",
        CpuFlags::AVX2,
        "decode_symbol_adapt16",
        MSAC_AVX2_KARAR,
    ),
    msac_aday("sse2", CpuFlags::SSE2, "decode_bool_adapt", MSAC_SSE2_KARAR),
    msac_aday("sse2", CpuFlags::SSE2, "decode_bool_equi", MSAC_SSE2_KARAR),
    msac_aday("sse2", CpuFlags::SSE2, "decode_bool", MSAC_SSE2_KARAR),
    msac_aday("sse2", CpuFlags::SSE2, "decode_hi_tok", MSAC_SSE2_KARAR),
];

/// Bu derlemedeki ISA adayları — S15.4 (CDEF) ve S15.5 (MC) AVX2 pilotları;
/// x86_64'te [`X86_ISA_CANDIDATES`], `aarch64`'te [`ARM_ISA_CANDIDATES`],
/// `wasm32` (`+simd128`) derlemesinde `wasm::WASM_ISA_CANDIDATES`.
#[cfg(target_arch = "x86_64")]
pub const ISA_CANDIDATES: &[IsaCandidate] = X86_ISA_CANDIDATES;

/// `aarch64` derlemesindeki ISA adayları (`16` yuva, `P51 / F5a`).
#[cfg(target_arch = "aarch64")]
pub const ISA_CANDIDATES: &[IsaCandidate] = ARM_ISA_CANDIDATES;

/// `wasm32` (`target_feature = "simd128"`) derlemesindeki ISA adayları (`31` yuva, `WP-10`).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub const ISA_CANDIDATES: &[IsaCandidate] = wasm::WASM_ISA_CANDIDATES;

/// Diğer hedeflerde (ve skaler `wasm32` artefaktında) ISA gövdesi derlenmez.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub const ISA_CANDIDATES: &[IsaCandidate] = &[];

/// Yuva anahtarının ISA adayı indeksleri, düzey sırasıyla ([`isa_rank`]).
fn candidate_indices(key: &SlotKey) -> Vec<usize> {
    let mut v: Vec<usize> = (0..ISA_CANDIDATES.len())
        .filter(|&i| {
            let c = &ISA_CANDIDATES[i];
            c.family == key.family
                && c.field == key.field
                && (Some(c.bit_depth) == key.bit_depth
                    || (key.bit_depth.is_none() && c.bit_depth == 0))
        })
        .collect();
    v.sort_by_key(|&i| isa_rank(ISA_CANDIDATES[i].implementation));
    v
}

/// Adaylardan (düzey sırasında) ilk varsayılan; yoksa ilk aday.
fn lowest_default(candidates: &[usize]) -> Option<usize> {
    candidates
        .iter()
        .copied()
        .find(|&i| ISA_CANDIDATES[i].default)
        .or_else(|| candidates.first().copied())
}

/// MSAC işlevleri — derleme zamanı seçimi (dav1d `src/x86/msac.h`).
///
/// `(dav1d işlevi, Rust karşılığı, gerekçe)`. Çalışma zamanı işaretçisi yok;
/// sembol başına dolaylı çağrı maliyeti dav1d'nin de kaçındığı maliyettir.
pub const MSAC_FUNCTIONS: &[(&str, &str, &str)] = &[
    (
        "decode_symbol_adapt4",
        "MsacContext::decode_symbol_adapt4 (dsp::x86::msac::decode_symbol_adapt4_sse2)",
        "dav1d x86-64: SSE2 (msac.h:47); Rust MsacIsa::Sse2/Avx2 ile SSE2 gövdesi (P47)",
    ),
    (
        "decode_symbol_adapt8",
        "MsacContext::decode_symbol_adapt8 (dsp::x86::msac::decode_symbol_adapt8_sse2)",
        "dav1d x86-64: SSE2 (msac.h:48); Rust MsacIsa::Sse2/Avx2 ile SSE2 gövdesi (P47)",
    ),
    (
        "decode_symbol_adapt16",
        "MsacContext::decode_symbol_adapt16 (dsp::x86::msac::decode_symbol_adapt16_sse2 / msac_avx2::decode_symbol_adapt16_avx2)",
        "dav1d x86-64: SSE2/AVX2 (msac.h:57-69); Rust MsacIsa::Sse2 ile SSE2, MsacIsa::Avx2 ile AVX2 gövdesi (P47)",
    ),
    (
        "decode_bool_adapt",
        "MsacContext::decode_bool_adapt (dsp::x86::msac::decode_bool_adapt_sse2)",
        "dav1d: SSE2 (msac.h:52); Rust MsacIsa::Sse2/Avx2 ile SSE2 gövdesi (P47)",
    ),
    (
        "decode_bool_equi",
        "MsacContext::decode_bool_equi (dsp::x86::msac::decode_bool_equi_sse2)",
        "dav1d: SSE2 (msac.h:53); Rust MsacIsa::Sse2/Avx2 ile SSE2 gövdesi (P47)",
    ),
    (
        "decode_bool",
        "MsacContext::decode_bool (dsp::x86::msac::decode_bool_sse2)",
        "dav1d: SSE2 (msac.h:54); Rust MsacIsa::Sse2/Avx2 ile SSE2 gövdesi (P47)",
    ),
    (
        "decode_hi_tok",
        "MsacContext::decode_hi_tok (dsp::x86::msac::decode_hi_tok_sse2)",
        "dav1d x86-64: SSE2 (msac.h:49); Rust MsacIsa::Sse2/Avx2 ile SSE2 gövdesi (P47)",
    ),
];

/// `aarch64` MSAC işlevleri — derleme zamanı + `NeonToken` seçimi (`dav1d` `src/arm/msac.h`, `P54 / F5-MSAC`).
pub const ARM_MSAC_FUNCTIONS: &[(&str, &str, &str)] = &[
    (
        "decode_symbol_adapt4",
        "MsacContext::decode_symbol_adapt4 (dsp::arm::msac::adapt4_neon)",
        "dav1d arm64: NEON (src/arm/msac.h:61); Rust MsacIsa::Neon ile NEON gövdesi (P54)",
    ),
    (
        "decode_symbol_adapt8",
        "MsacContext::decode_symbol_adapt8 (dsp::arm::msac::adapt8_neon)",
        "dav1d arm64: NEON (src/arm/msac.h:62); Rust MsacIsa::Neon ile NEON gövdesi (P54)",
    ),
    (
        "decode_symbol_adapt16",
        "MsacContext::decode_symbol_adapt16 (dsp::arm::msac::adapt16_neon)",
        "dav1d arm64: NEON (src/arm/msac.h:63); Rust MsacIsa::Neon ile NEON gövdesi (P54)",
    ),
    (
        "decode_bool_adapt",
        "MsacContext::decode_bool_adapt (dsp::arm::msac::bool_adapt_neon)",
        "dav1d arm64: NEON (src/arm/msac.h:64); Rust MsacIsa::Neon ile NEON gövdesi (P54)",
    ),
    (
        "decode_bool_equi",
        "MsacContext::decode_bool_equi (dsp::arm::msac::bool_equi_neon)",
        "dav1d arm64: NEON (src/arm/msac.h:65); Rust MsacIsa::Neon ile NEON gövdesi (P54)",
    ),
    (
        "decode_bool",
        "MsacContext::decode_bool (dsp::arm::msac::bool_neon)",
        "dav1d arm64: NEON (src/arm/msac.h:66); Rust MsacIsa::Neon ile NEON gövdesi (P54)",
    ),
    (
        "decode_hi_tok",
        "MsacContext::decode_hi_tok (dsp::arm::msac::hi_tok_neon)",
        "dav1d arm64: NEON (src/arm/msac.h:67); Rust MsacIsa::Neon ile NEON gövdesi (P54)",
    ),
];

/// `mc[<FILTER_2D_…>]` / `mct[<FILTER_2D_…>]` yuva alanı → (`prep` mi, filtre).
#[cfg(target_arch = "x86_64")]
fn mc_slot_filter(field: &str) -> Option<(bool, mc::Filter2d)> {
    let (prep, ad) = if let Some(r) = field.strip_prefix("mct[") {
        (true, r)
    } else {
        (false, field.strip_prefix("mc[")?)
    };
    let ad = ad.strip_suffix(']')?;
    let i = FILTER_2D_NAMES.iter().position(|n| *n == ad)?;
    Some((prep, mc::Filter2d::from_index(i)?))
}

/// dav1d `enum Filter2d` adları, [`mc::Filter2d`] indeks sırasıyla
/// (`src/levels.h:184-195`).
const FILTER_2D_NAMES: [&str; 10] = [
    "FILTER_2D_8TAP_REGULAR",
    "FILTER_2D_8TAP_REGULAR_SMOOTH",
    "FILTER_2D_8TAP_REGULAR_SHARP",
    "FILTER_2D_8TAP_SHARP_REGULAR",
    "FILTER_2D_8TAP_SHARP_SMOOTH",
    "FILTER_2D_8TAP_SHARP",
    "FILTER_2D_8TAP_SMOOTH_REGULAR",
    "FILTER_2D_8TAP_SMOOTH",
    "FILTER_2D_8TAP_SMOOTH_SHARP",
    "FILTER_2D_BILINEAR",
];

/// dav1d intra tahmin uygulama kipleri, [`ipred::IntraPredMode`] sırasıyla
/// (`src/levels.h:117-135`).
const INTRA_PRED_NAMES: [&str; 14] = [
    "DC_PRED",
    "VERT_PRED",
    "HOR_PRED",
    "LEFT_DC_PRED",
    "TOP_DC_PRED",
    "DC_128_PRED",
    "Z1_PRED",
    "Z2_PRED",
    "Z3_PRED",
    "SMOOTH_PRED",
    "SMOOTH_V_PRED",
    "SMOOTH_H_PRED",
    "PAETH_PRED",
    "FILTER_PRED",
];

/// `cfl_pred` kipleri, [`ipred::CflPredMode`] indeks sırasıyla.
const CFL_PRED_NAMES: [&str; 4] = ["DC_PRED", "LEFT_DC_PRED", "TOP_DC_PRED", "DC_128_PRED"];

/// `[DAV1D_PIXEL_LAYOUT_I4xx - 1]` dizinleri (`cfl_ac`, `generate_grain_uv`,
/// `fguv_32x32xn`), indeks sırasıyla (I420, I422, I444).
const LAYOUT_INDEX: [&str; 3] = [
    "DAV1D_PIXEL_LAYOUT_I420-1",
    "DAV1D_PIXEL_LAYOUT_I422-1",
    "DAV1D_PIXEL_LAYOUT_I444-1",
];

/// P39: `<işlev>[DAV1D_PIXEL_LAYOUT_I4xx-1]` alanının düzen dizini (0 = 420,
/// 1 = 422, 2 = 444); başka alan `None`.
#[cfg(target_arch = "x86_64")]
fn fg_duzen(field: &str, islev: &str) -> Option<usize> {
    let ic = field
        .strip_prefix(islev)?
        .strip_prefix('[')?
        .strip_suffix(']')?;
    LAYOUT_INDEX.iter().position(|&l| l == ic)
}

/// P43: `itxfm_add[<TX>][<TXTP>]` yuva alanını `(TxSize, TxType)` çiftine çözer.
#[cfg(target_arch = "x86_64")]
fn itx_slot(field: &str) -> Option<(itx::TxSize, itx::TxType)> {
    let rest = field.strip_prefix("itxfm_add[")?;
    let (tx_ad, txtp_kalan) = rest.split_once("][")?;
    let txtp_ad = txtp_kalan.strip_suffix(']')?;
    let tx = TX_SIZES.into_iter().find(|t| t.dav1d_name() == tx_ad)?;
    let txtp = TX_TYPES
        .into_iter()
        .chain([TxType::WhtWht])
        .find(|t| t.dav1d_name() == txtp_ad)?;
    Some((tx, txtp))
}

/// P45: `intra_pred[<MODE>]` yuva alanını [`ipred::IntraPredMode`]'a çözer.
#[cfg(target_arch = "x86_64")]
fn ipred_intra_slot(field: &str) -> Option<ipred::IntraPredMode> {
    let ad = field.strip_prefix("intra_pred[")?.strip_suffix(']')?;
    let idx = INTRA_PRED_NAMES.iter().position(|&n| n == ad)?;
    ipred::IntraPredMode::from_index(idx)
}

/// P45: `cfl_pred[<MODE>]` yuva alanını [`ipred::CflPredMode`]'a çözer.
#[cfg(target_arch = "x86_64")]
fn ipred_cfl_pred_slot(field: &str) -> Option<ipred::CflPredMode> {
    let ad = field.strip_prefix("cfl_pred[")?.strip_suffix(']')?;
    let idx = CFL_PRED_NAMES.iter().position(|&n| n == ad)?;
    ipred::CflPredMode::ALL.get(idx).copied()
}

/// P46: `loop_filter_sb[<plane>][<dir>]` yuva alanını `(plane, dir)` çiftine çözer.
#[cfg(target_arch = "x86_64")]
fn lf_sb_slot(field: &str) -> Option<(usize, usize)> {
    match field {
        "loop_filter_sb[0][0]" => Some((0, 0)),
        "loop_filter_sb[0][1]" => Some((0, 1)),
        "loop_filter_sb[1][0]" => Some((1, 0)),
        "loop_filter_sb[1][1]" => Some((1, 1)),
        _ => None,
    }
}

/// P46: `wiener[<tap>]` yuva alanını `0..2` dizinine çözer.
#[cfg(target_arch = "x86_64")]
fn lr_wiener_slot(field: &str) -> Option<usize> {
    match field {
        "wiener[0]" => Some(0),
        "wiener[1]" => Some(1),
        _ => None,
    }
}

/// P46: `sgr[<idx>]` yuva alanını `0..3` dizinine çözer.
#[cfg(target_arch = "x86_64")]
fn lr_sgr_slot(field: &str) -> Option<usize> {
    match field {
        "sgr[0]" => Some(0),
        "sgr[1]" => Some(1),
        "sgr[2]" => Some(2),
        _ => None,
    }
}

/// Bütün yuva anahtarları (sabit sıra): bit derinliğine bağlı aileler 8, 10,
/// 12 için; sonra pal, refmvs ve MSAC.
#[must_use]
pub fn slot_keys() -> Vec<SlotKey> {
    let mut keys = Vec::new();
    for bd in [8u8, 10, 12] {
        let mut push = |family: &'static str, field: String| {
            keys.push(SlotKey {
                family,
                field,
                bit_depth: Some(bd),
                selection: Selection::Runtime,
            });
        };
        for tx in TX_SIZES {
            for txtp in TX_TYPES.iter().copied().chain([TxType::WhtWht]) {
                if is_supported(tx, txtp) {
                    push(
                        "itx",
                        format!("itxfm_add[{}][{}]", tx.dav1d_name(), txtp.dav1d_name()),
                    );
                }
            }
        }
        for group in ["mc", "mc_scaled", "mct", "mct_scaled"] {
            for name in FILTER_2D_NAMES {
                push("mc", format!("{group}[{name}]"));
            }
        }
        for field in ["avg", "w_avg", "mask"] {
            push("mc", field.to_string());
        }
        for i in 0..mc::N_W_MASK {
            push("mc", format!("w_mask[{i}]"));
        }
        for field in [
            "blend", "blend_v", "blend_h", "warp8x8", "warp8x8t", "emu_edge",
        ] {
            push("mc", field.to_string());
        }
        push("resize", "resize".to_string());
        for name in INTRA_PRED_NAMES {
            push("ipred", format!("intra_pred[{name}]"));
        }
        for layout in LAYOUT_INDEX {
            push("ipred", format!("cfl_ac[{layout}]"));
        }
        for name in CFL_PRED_NAMES {
            push("ipred", format!("cfl_pred[{name}]"));
        }
        push("ipred", "pal_pred".to_string());
        for plane in 0..2 {
            for dir in 0..2 {
                push("loopfilter", format!("loop_filter_sb[{plane}][{dir}]"));
            }
        }
        push("cdef", "dir".to_string());
        for i in 0..3 {
            push("cdef", format!("fb[{i}]"));
        }
        for i in 0..2 {
            push("looprestoration", format!("wiener[{i}]"));
        }
        for i in 0..3 {
            push("looprestoration", format!("sgr[{i}]"));
        }
        push("filmgrain", "generate_grain_y".to_string());
        for layout in LAYOUT_INDEX {
            push("filmgrain", format!("generate_grain_uv[{layout}]"));
        }
        push("filmgrain", "fgy_32x32xn".to_string());
        for layout in LAYOUT_INDEX {
            push("filmgrain", format!("fguv_32x32xn[{layout}]"));
        }
    }
    let mut push = |family: &'static str, field: &str, selection: Selection| {
        keys.push(SlotKey {
            family,
            field: field.to_string(),
            bit_depth: None,
            selection,
        });
    };
    push("pal", "pal_idx_finish", Selection::Runtime);
    for field in ["load_tmvs", "save_tmvs", "splat_mv"] {
        push("refmvs", field, Selection::Runtime);
    }
    for (name, _, _) in MSAC_FUNCTIONS {
        push("msac", name, Selection::CompileTime);
    }
    keys
}

#[cfg(all(test, feature = "std"))]
impl DspDispatch {
    /// Test kurucusu: skaler tabloyu `edit` ile değiştirir (yalnız birim
    /// testleri; üretim yolu [`DspDispatch::select`]'tir).
    pub(crate) fn scalar_for_test(edit: impl FnOnce(&mut [BitDepthDsp; 3])) -> Self {
        let mut table = Self::scalar_only();
        edit(&mut table.tables);
        table
    }

    /// Test: bit derinliği tablosunun CDEF ailesine yazma.
    pub(crate) fn test_cdef_mut(t: &mut BitDepthDsp) -> &mut CdefDsp {
        &mut t.cdef
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
