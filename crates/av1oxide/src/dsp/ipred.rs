//! Intra tahmin (ipred) sözleşmesi — S2b.2.
//!
//! Bu modül dav1d `Dav1dIntraPredDSPContext` (`src/ipred.h:71-80`) karşılığının
//! **tiplerini, mod numaralandırmasını ve bellek sözleşmesini** tanımlar;
//! gövdeler bit derinliği başına monomorfize edilmiş skaler çekirdeklerdir
//! ([`crate::dsp::scalar::ipred`]). Kapsam: `intra_pred[]` (14 mod), `cfl_ac[3]`,
//! `cfl_pred[]` ve `pal_pred`. `pal_idx_finish` ayrı bağlamdır
//! ([`crate::dsp::pal`]).
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! Taşınan her kuralın dav1d kaynağındaki satırı. Çelişki hâlinde kaynak
//! esastır; bu tablo denetim izidir.
//!
//! | Kural | Kaynak |
//! |---|---|
//! | `intra_pred[]` mod sırası / `N_IMPL_INTRA_PRED_MODES` | `src/levels.h:104-120`, `src/ipred_tmpl.c:744-760` |
//! | `Dav1dIntraPredDSPContext` alanları | `src/ipred.h:71-80` |
//! | `angle` üst bayrakları: bit 9 = yumuşak blok, bit 10+ = kenar filtresi | `src/ipred_prepare.h:100-103` |
//! | `angle & 511`, `is_sm = (angle>>9)&1`, `enable = angle>>10` | `src/ipred_tmpl.c:409-412, 463-466, 543-546` |
//! | `max_width`/`max_height` yalnız Z2 kenar filtresinin sınırı | `src/ipred_tmpl.c:475-484` |
//! | DC / DC_TOP / DC_LEFT / DC_128 üretimi ve `splat_dc` | `src/ipred_tmpl.c:62-105, 207-226` |
//! | `dc_gen` 1/2 ve 1/4 çarpanları (`0x5556`/`0x3334` 8-bit, `0xAAAB`/`0x6667` 16bpc) | `src/ipred_tmpl.c:129-157` |
//! | V/H/PAETH | `src/ipred_tmpl.c:241-278` |
//! | SMOOTH ailesi ve `(pred+256)>>9` / `(pred+128)>>8` yuvarlamaları | `src/ipred_tmpl.c:280-331` |
//! | `get_filter_strength` eşikleri | `src/ipred_tmpl.c:333-368` |
//! | `filter_edge` çekirdeği `{0,4,8,4,0}`, `{0,5,6,5,0}`, `{2,4,4,4,2}`, `(s+8)>>4` | `src/ipred_tmpl.c:370-394` |
//! | `get_upsample`: `angle < 40 && wh <= 16>>is_sm` | `src/ipred_tmpl.c:396-398` |
//! | `upsample_edge` çekirdeği `{-1,9,9,-1}`, `iclip_pixel((s+8)>>4)` | `src/ipred_tmpl.c:400-418` |
//! | Z1: yalnız üst kenar; `max_base_x`, `base_inc = 1 + upsample` | `src/ipred_tmpl.c:420-460` |
//! | Z2: üst + sol; `max_width`/`max_height` filtre sınırı; sol için `base_y` | `src/ipred_tmpl.c:462-540` |
//! | Z3: yalnız sol; `left_out` ve `max_base_y` | `src/ipred_tmpl.c:542-616` |
//! | `FILTER_PRED`: `filt_idx &= 511`, `filt_idx < 5`, 2×4 bloklar, `(acc+8)>>4` | `src/ipred_tmpl.c:618-670` |
//! | CfL AC: `w_pad`/`h_pad`, `ss_hor`/`ss_ver`, DC çıkarma | `src/ipred_tmpl.c:672-740` |
//! | CfL tahmin: `iclip_pixel(dc + apply_sign((abs(diff)+32)>>6, diff))` | `src/ipred_tmpl.c:62-75` |
//! | `pal_pred`: iki nibble / 2 piksel | `src/ipred_tmpl.c:723-735` |
//! | 16bpc `PXSTRIDE(x) = x >> 1` (çağrı stride'ı **bayt**) | `include/common/bitdepth.h:79-82` |
//! | 16bpc `iclip_pixel(x) = iclip(x, 0, bitdepth_max)` | `include/common/intops.h:137-153` |
//!
//! ## Mod numaralandırması
//!
//! dav1d'nin `enum IntraPredMode` değerleri (`src/levels.h:104-120`) ile
//! **birebir** aynı sayısal değerler kullanılır. dav1d'nin takma adları
//! (`LEFT_DC_PRED = DIAG_DOWN_LEFT_PRED` vb.) bu portta kanonik adlardır; 14
//! uygulama modu 0..=13 aralığındadır ve `intra_pred[]` dizisinin indeksidir:
//!
//! | Değer | [`IntraPredMode`] | dav1d adı |
//! |---|---|---|
//! | 0 | `Dc` | `DC_PRED` |
//! | 1 | `Vert` | `VERT_PRED` |
//! | 2 | `Hor` | `HOR_PRED` |
//! | 3 | `LeftDc` | `LEFT_DC_PRED` (= `DIAG_DOWN_LEFT_PRED`) |
//! | 4 | `TopDc` | `TOP_DC_PRED` |
//! | 5 | `Dc128` | `DC_128_PRED` |
//! | 6 | `Z1` | `Z1_PRED` |
//! | 7 | `Z2` | `Z2_PRED` |
//! | 8 | `Z3` | `Z3_PRED` |
//! | 9 | `Smooth` | `SMOOTH_PRED` |
//! | 10 | `SmoothV` | `SMOOTH_V_PRED` |
//! | 11 | `SmoothH` | `SMOOTH_H_PRED` |
//! | 12 | `Paeth` | `PAETH_PRED` |
//! | 13 | `Filter` | `FILTER_PRED` (= `N_INTRA_PRED_MODES`) |
//!
//! ## Kenar (edge) tamponu ve okuma aralıkları
//!
//! dav1d tek bir `topleft` işaretçisi alır; `topleft[0]` sol-üst piksel,
//! `topleft[1..=w]` üst kenar, `topleft[-1..=-h]` sol kenardır
//! (`src/ipred_prepare.h:82-96`). Rust'ta negatif indeksleme olmadığı için
//! çağrı [`Edge`] ile yapılır: ham bayt dilimi + `topleft` pikselinin **bayt**
//! ofseti. Çekirdek, **dilim içinde kalan** ve aşağıdaki tabloda verilen
//! aralığın dışına **çıkmaz**; aralık dışı dolgu deseni çıktıyı değiştirmez
//! (`tests/ipred_sentinel.rs`).
//!
//! | Mod | Geri (piksel) | İleri (piksel) |
//! |---|---|---|
//! | DC, PAETH, SMOOTH, SMOOTH_V, SMOOTH_H | `h` | `w` |
//! | DC_TOP | 0 | `w` |
//! | DC_LEFT, HOR | `h` | 0 |
//! | VERT | 0 | `w` |
//! | DC_128 | 0 | 0 |
//! | Z1 | 0 | `w + min(w,h)` |
//! | Z2 | `h` | `w` |
//! | Z3 | upsample/filtre varsa `w+h`, yoksa `h + min(w,h)` | 0 |
//! | FILTER | `h` | `w` |
//!
//! Z2'de upsample/kenar filtresi **geçici tamponun** uzunluğunu değiştirir,
//! kenar okuma aralığını değiştirmez: upsample girdisi de yalnız
//! `topleft_in[0..=w]` ve `topleft_in[-h..=0]` aralığını okur
//! (`src/ipred_tmpl.c:468-490`). `max_width`/`max_height` **yalnız** kenar
//! filtresinin hangi bölümüne uygulanacağını sınırlar (aynı satırlar).
//!
//! ## `angle` semantiği
//!
//! - `angle & 511`: 3..87 → Z1, 91..177 → Z2, 183..269 → Z3 (dav1d tablosu
//!   `src/tables.c:719-758`; tek/çift indeksler `dav1d_dr_intra_derivative`).
//! - `(angle >> 9) & 1` = `ANGLE_SMOOTH_EDGE_FLAG` (512): kenar filtresi daha
//!   yumuşak blokta daha güçlü (`get_filter_strength(..., is_sm=1)`).
//! - `angle >> 10` (1024 = `ANGLE_USE_EDGE_FILTER_FLAG`): kenar filtresi ve
//!   upsample etkin. Yalnız bu bit 1 ise filtre/upsample yapılır.
//! - `FILTER_PRED`'de `angle & 511` = filtre indeksi (0..4); üst bitler yok
//!   sayılır (`filt_idx &= 511`).
//!
//! ## Bellek sözleşmesi (ADR-0003)
//!
//! - `dst`: **bayt** dilimi; `dst_stride` **bayt**. 8-bit'te piksel 1, 10/12-
//!   bit'te 2 bayttır (küçük sonlu `u16`). Hedef dilim en az
//!   `(h-1)·dst_stride + w·bpp` bayt olmalıdır; dışına yazılmaz.
//! - `edge`: ham bayt dilimi; `edge.topleft` piksel başına `bpp` ile hizalı
//!   olmalıdır. Çekirdek yukarıdaki aralığı **önce O(1) doğrular**, sonra
//!   güvenli indeksleme kullanır; sözleşme dışı girdide panikler.
//! - Hizalama varsayımı yoktur; `dst_stride` keyfî olabilir (satır
//!   uzunluğundan büyük/eşit). 16bpc'de çift olmalıdır (`PXSTRIDE`).
//! - `bitdepth_max`: 16bpc'de `iclip_pixel` tavanı (0x3ff veya 0xfff). dav1d
//!   gibi 10/12-bit **aynı 16bpc gövdesini** kullanır; ayrım yalnız bu
//!   parametredir (`include/common/bitdepth.h`). 8-bit'te yok sayılır.
//!
//! ## Dispatch (ADR-0003 §2.1)
//!
//! Bit derinliği başına **ayrı tablo** (F3c): [`crate::dsp::BitDepthDsp::ipred`]
//! 8/10/12 için o bit derinliğinin monomorflarını taşır
//! ([`IntraPredDsp::scalar`]); 16bpc gövdesi 10 ve 12 bitte ortaktır,
//! `bitdepth_max` girişte derleme zamanı sabitidir. Bu adımda SIMD yoktur.

#![forbid(unsafe_code)]

use crate::dsp::ipred_tables::DR_INTRA_DERIVATIVE;
use crate::dsp::scalar::ipred as scalar;
use crate::frame::BitDepth;

/// `intra_pred[]` dizisinin uzunluğu (dav1d `N_IMPL_INTRA_PRED_MODES`).
pub const N_IMPL_INTRA_PRED_MODES: usize = 14;

/// CfL tahmin modu sayısı (DC, LEFT_DC, TOP_DC, DC_128).
pub const N_CFL_PRED_MODES: usize = 4;

/// CfL AC biçim sayısı (4:2:0, 4:2:2, 4:4:4).
pub const N_CFL_AC_LAYOUTS: usize = 3;

/// Intra tahmin modu — dav1d `enum IntraPredMode` değerleriyle **birebir**
/// aynı diskriminantlar (bkz. modül belgesi tablosu).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IntraPredMode {
    /// 0 — `DC_PRED`: üst + sol ortalaması.
    Dc = 0,
    /// 1 — `VERT_PRED`: üst kenar kopyası.
    Vert = 1,
    /// 2 — `HOR_PRED`: sol kenar kopyası.
    Hor = 2,
    /// 3 — `LEFT_DC_PRED`: yalnız sol ortalaması.
    LeftDc = 3,
    /// 4 — `TOP_DC_PRED`: yalnız üst ortalaması.
    TopDc = 4,
    /// 5 — `DC_128_PRED`: sabit orta değer.
    Dc128 = 5,
    /// 6 — `Z1_PRED`: 3..87 derece yönlü (üst kenar).
    Z1 = 6,
    /// 7 — `Z2_PRED`: 91..177 derece yönlü (üst + sol).
    Z2 = 7,
    /// 8 — `Z3_PRED`: 183..269 derece yönlü (sol kenar).
    Z3 = 8,
    /// 9 — `SMOOTH_PRED`.
    Smooth = 9,
    /// 10 — `SMOOTH_V_PRED`.
    SmoothV = 10,
    /// 11 — `SMOOTH_H_PRED`.
    SmoothH = 11,
    /// 12 — `PAETH_PRED`.
    Paeth = 12,
    /// 13 — `FILTER_PRED` (recursive filtre; yalnız ≤32×32).
    Filter = 13,
}

impl IntraPredMode {
    /// Tüm uygulama modları, `intra_pred[]` sırasıyla.
    pub const ALL: [Self; N_IMPL_INTRA_PRED_MODES] = [
        Self::Dc,
        Self::Vert,
        Self::Hor,
        Self::LeftDc,
        Self::TopDc,
        Self::Dc128,
        Self::Z1,
        Self::Z2,
        Self::Z3,
        Self::Smooth,
        Self::SmoothV,
        Self::SmoothH,
        Self::Paeth,
        Self::Filter,
    ];

    /// Diskriminanttan mod üretir; 13'ten büyük değerler `None`'dur.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Dc),
            1 => Some(Self::Vert),
            2 => Some(Self::Hor),
            3 => Some(Self::LeftDc),
            4 => Some(Self::TopDc),
            5 => Some(Self::Dc128),
            6 => Some(Self::Z1),
            7 => Some(Self::Z2),
            8 => Some(Self::Z3),
            9 => Some(Self::Smooth),
            10 => Some(Self::SmoothV),
            11 => Some(Self::SmoothH),
            12 => Some(Self::Paeth),
            13 => Some(Self::Filter),
            _ => None,
        }
    }

    /// Dizideki indeks (dav1d `intra_pred[]` indeksi).
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Yönlü mod mu (Z1/Z2/Z3)? Bu modlarda `angle` anlamlıdır.
    #[must_use]
    pub const fn is_directional(self) -> bool {
        matches!(self, Self::Z1 | Self::Z2 | Self::Z3)
    }

    /// dav1d adı (raporlama ve fixture'lar için).
    #[must_use]
    pub const fn dav1d_name(self) -> &'static str {
        match self {
            Self::Dc => "dc",
            Self::Vert => "v",
            Self::Hor => "h",
            Self::LeftDc => "dc_left",
            Self::TopDc => "dc_top",
            Self::Dc128 => "dc_128",
            Self::Z1 => "z1",
            Self::Z2 => "z2",
            Self::Z3 => "z3",
            Self::Smooth => "smooth",
            Self::SmoothV => "smooth_v",
            Self::SmoothH => "smooth_h",
            Self::Paeth => "paeth",
            Self::Filter => "filter",
        }
    }
}

/// CfL (chroma-from-luma) tahmin modu.
///
/// dav1d `cfl_pred[DC_128_PRED + 1]` dizisinin kullanılan dört girişidir
/// (`src/ipred_tmpl.c:764-767`): DC=0, LEFT_DC=3, TOP_DC=4, DC_128=5.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CflPredMode {
    /// DC_PRED — üst + sol ortalaması.
    Dc = 0,
    /// LEFT_DC_PRED — yalnız sol ortalaması.
    LeftDc = 3,
    /// TOP_DC_PRED — yalnız üst ortalaması.
    TopDc = 4,
    /// DC_128_PRED — sabit orta değer.
    Dc128 = 5,
}

impl CflPredMode {
    /// Kullanılan dört mod, dav1d `cfl_pred[]` indeks sırasıyla.
    pub const ALL: [Self; N_CFL_PRED_MODES] = [Self::Dc, Self::LeftDc, Self::TopDc, Self::Dc128];

    /// Dizideki konum (0..3); `DspDispatch` yuvası bu sırayı kullanır.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Dc => 0,
            Self::LeftDc => 1,
            Self::TopDc => 2,
            Self::Dc128 => 3,
        }
    }

    /// dav1d `cfl_pred[]` indeksi (0/3/4/5).
    #[must_use]
    pub const fn dav1d_index(self) -> usize {
        self as usize
    }
}

/// Kenar tamponu görünümü: ham bayt dilimi + sol-üst pikselin bayt ofseti.
///
/// `bytes[topleft]` sol-üst pikseldir; negatif yön `topleft`'ten **önceki**
/// baytlardır. 16bpc'de her piksel 2 bayttır (küçük sonlu) ve `topleft` çift
/// olmalıdır. Ayrıntılı okuma aralıkları modül belgesindedir.
#[derive(Clone, Copy, Debug)]
pub struct Edge<'a> {
    /// Ham kenar tamponu (bayt).
    pub bytes: &'a [u8],
    /// Sol-üst pikselin `bytes` içindeki bayt ofseti.
    pub topleft: usize,
}

impl<'a> Edge<'a> {
    /// Yeni kenar görünümü.
    #[must_use]
    pub const fn new(bytes: &'a [u8], topleft: usize) -> Self {
        Self { bytes, topleft }
    }
}

/// Bir `intra_pred` çekirdeğinin imzası (ADR-0003: dilim + bayt stride + açık
/// boyut + açık bit derinliği).
pub type AngularPredFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
);

/// Bir `cfl_ac` çekirdeğinin imzası — dav1d `decl_cfl_ac_fn` (`src/ipred.h:51-55`).
///
/// `ac` çıktısı `cw × ch` `i16`'dır (satır adımı `cw`). `y` ham bayt dilimi,
/// `y_stride` **bayt**; luma boyutu biçime göre `cw << ss_hor`,
/// `ch << ss_ver`'dir.
pub type CflAcFn = fn(
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
    bit_depth: BitDepth,
);

/// Bir `cfl_pred` çekirdeğinin imzası — dav1d `decl_cfl_pred_fn` (`src/ipred.h:59-63`).
pub type CflPredFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    ac: &[i16],
    alpha: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
);

/// Bir `pal_pred` çekirdeğinin imzası — dav1d `decl_pal_pred_fn` (`src/ipred.h:66-70`).
///
/// `pal` ham bayt dilimi: 8 piksel (8-bit'te 8 bayt, 16bpc'de 16 bayt).
/// `idx` paketlenmiş nibble'lar: her bayt iki piksel (`idx[y·(w/2)+x/2]`,
/// düşük nibble sol piksel, yüksek nibble sağ piksel).
pub type PalPredFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
);

/// Intra tahmin çekirdek ailesi — dav1d `Dav1dIntraPredDSPContext`.
#[derive(Clone, Copy)]
pub struct IntraPredDsp {
    /// `intra_pred[]` — 14 mod, [`IntraPredMode::index`] ile indekslenir.
    pub intra_pred: [AngularPredFn; N_IMPL_INTRA_PRED_MODES],
    /// `cfl_ac[3]` — 4:2:0 / 4:2:2 / 4:4:4 sırasıyla
    /// ([`ChromaFormat::C420`](crate::frame::ChromaFormat::C420), `C422`, `C444`).
    pub cfl_ac: [CflAcFn; N_CFL_AC_LAYOUTS],
    /// `cfl_pred[]` — [`CflPredMode::index`] ile indekslenir.
    pub cfl_pred: [CflPredFn; N_CFL_PRED_MODES],
    /// `pal_pred`.
    pub pal_pred: PalPredFn,
}

impl IntraPredDsp {
    /// `BPC` bit derinliğinin skaler tablosu (F3c: her giriş o bit derinliğinin
    /// monomorfudur; 16bpc gövdesinin `bitdepth_max`'ı derleme zamanı sabitidir).
    #[must_use]
    pub const fn scalar<const BPC: u32>() -> Self {
        Self {
            intra_pred: [
                scalar::ipred_dc::<BPC>,
                scalar::ipred_vert::<BPC>,
                scalar::ipred_hor::<BPC>,
                scalar::ipred_left_dc::<BPC>,
                scalar::ipred_top_dc::<BPC>,
                scalar::ipred_dc_128::<BPC>,
                scalar::ipred_z1::<BPC>,
                scalar::ipred_z2::<BPC>,
                scalar::ipred_z3::<BPC>,
                scalar::ipred_smooth::<BPC>,
                scalar::ipred_smooth_v::<BPC>,
                scalar::ipred_smooth_h::<BPC>,
                scalar::ipred_paeth::<BPC>,
                scalar::ipred_filter::<BPC>,
            ],
            cfl_ac: [
                scalar::cfl_ac_420::<BPC>,
                scalar::cfl_ac_422::<BPC>,
                scalar::cfl_ac_444::<BPC>,
            ],
            cfl_pred: [
                scalar::cfl_pred_dc::<BPC>,
                scalar::cfl_pred_left_dc::<BPC>,
                scalar::cfl_pred_top_dc::<BPC>,
                scalar::cfl_pred_dc_128::<BPC>,
            ],
            pal_pred: scalar::pal_pred::<BPC>,
        }
    }

    /// Çalışma zamanı bit derinliğinin skaler tablosu (kurulum ve testler).
    #[must_use]
    pub const fn scalar_for(bit_depth: crate::frame::BitDepth) -> Self {
        match bit_depth {
            crate::frame::BitDepth::Eight => Self::scalar::<8>(),
            crate::frame::BitDepth::Ten => Self::scalar::<10>(),
            crate::frame::BitDepth::Twelve => Self::scalar::<12>(),
        }
    }
}

impl core::fmt::Debug for IntraPredDsp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IntraPredDsp")
            .field(
                "intra_pred",
                &format_args!("[{} fn]", self.intra_pred.len()),
            )
            .field("cfl_ac", &format_args!("[{} fn]", self.cfl_ac.len()))
            .field("cfl_pred", &format_args!("[{} fn]", self.cfl_pred.len()))
            .field("pal_pred", &format_args!("{}", "fn"))
            .finish()
    }
}

/// P45: SIMD `ipred` girişlerinin sözleşme dışı girdide skaler gövdeye
/// devrettiği çağrı sayısı (süreç başından beri). Çözme hattında 0 olmalıdır.
static IPRED_SKALER_DEVIR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// [`IPRED_SKALER_DEVIR`] değeri.
#[must_use]
pub fn ipred_skaler_devir_sayisi() -> u64 {
    IPRED_SKALER_DEVIR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Skaler devri sayar (yalnız SIMD girişlerinin soğuk yolu).
#[cold]
pub fn ipred_skaler_devri_say() {
    IPRED_SKALER_DEVIR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// `(h − 1)·stride + son` (taşmada `None`).
#[inline(always)]
fn ipred_bolge_sonu(h: usize, stride: usize, son: usize) -> Option<usize> {
    h.checked_sub(1)?.checked_mul(stride)?.checked_add(son)
}

/// Geçerli AV1 dönüşüm / intra tahmin blok boyutu mu (`w, h ∈ {4, 8, 16, 32, 64}`
/// ve en çok 1:4 en-boy oranı).
#[inline(always)]
const fn gecerli_blok_boyutu(width: usize, height: usize, max_dim: usize) -> bool {
    width >= 4
        && width <= max_dim
        && width.is_power_of_two()
        && height >= 4
        && height <= max_dim
        && height.is_power_of_two()
        && width <= 4 * height
        && height <= 4 * width
}

/// Paniksiz kenar okuma aralığı `(geri, ileri)` piksel; geçersiz açı veya
/// `FILTER_PRED` parametresinde `None`.
#[must_use]
#[inline(always)]
pub fn kenar_okuma_araligi(
    mode: IntraPredMode,
    width: usize,
    height: usize,
    angle: usize,
) -> Option<(usize, usize)> {
    let base_angle = angle & 511;
    let is_sm = ((angle >> 9) & 1) == 1;
    let enable = (angle >> 10) != 0;
    match mode {
        IntraPredMode::Dc
        | IntraPredMode::Paeth
        | IntraPredMode::Smooth
        | IntraPredMode::SmoothV
        | IntraPredMode::SmoothH => Some((height, width)),
        IntraPredMode::Filter => {
            if width <= 32 && height <= 32 && base_angle < 5 {
                Some((height, width))
            } else {
                None
            }
        }
        IntraPredMode::TopDc | IntraPredMode::Vert => Some((0, width)),
        IntraPredMode::LeftDc | IntraPredMode::Hor => Some((height, 0)),
        IntraPredMode::Dc128 => Some((0, 0)),
        IntraPredMode::Z1 => {
            if base_angle > 0
                && base_angle < 90
                && DR_INTRA_DERIVATIVE
                    .get(base_angle >> 1)
                    .is_some_and(|&d| d > 0)
            {
                Some((0, width + width.min(height)))
            } else {
                None
            }
        }
        IntraPredMode::Z2 => {
            if base_angle > 90
                && base_angle < 180
                && DR_INTRA_DERIVATIVE
                    .get((base_angle - 90) >> 1)
                    .is_some_and(|&d| d > 0)
                && DR_INTRA_DERIVATIVE
                    .get((180 - base_angle) >> 1)
                    .is_some_and(|&d| d > 0)
            {
                Some((height, width))
            } else {
                None
            }
        }
        IntraPredMode::Z3 => {
            if base_angle > 180
                && base_angle < 270
                && DR_INTRA_DERIVATIVE
                    .get((270 - base_angle) >> 1)
                    .is_some_and(|&d| d > 0)
            {
                let wh = width + height;
                let a = base_angle - 180;
                let upsample = enable && a < 40 && wh <= (16 >> usize::from(is_sm));
                let filtered = enable && !upsample && z3_filtre_etkin_mi(wh, a, is_sm);
                if upsample || filtered {
                    Some((width + height, 0))
                } else {
                    Some((height + width.min(height), 0))
                }
            } else {
                None
            }
        }
    }
}

#[inline(always)]
const fn z3_filtre_etkin_mi(wh: usize, angle: usize, is_sm: bool) -> bool {
    if is_sm {
        if wh <= 8 {
            angle >= 40
        } else if wh <= 16 {
            angle >= 20
        } else if wh <= 24 {
            angle >= 4
        } else {
            true
        }
    } else if wh <= 8 {
        angle >= 56
    } else if wh <= 16 {
        angle >= 40
    } else if wh <= 24 {
        angle >= 8
    } else {
        true
    }
}

/// P45: `intra_pred[mode]` SIMD girişinin O(1) sözleşmesi.
#[must_use]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn angular_sozlesmesi<const BPC: u32>(
    mode: IntraPredMode,
    dst_len: usize,
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) -> bool {
    let bpp = if BPC == 8 { 1 } else { 2 };
    let bpc_max = (1u32 << BPC) - 1;
    let max_dim = if matches!(mode, IntraPredMode::Filter) {
        32
    } else {
        64
    };
    if bit_depth.bits() != BPC
        || u32::from(bitdepth_max) != bpc_max
        || !gecerli_blok_boyutu(width, height, max_dim)
        || (BPC > 8 && (!dst_stride.is_multiple_of(2) || !edge.topleft.is_multiple_of(2)))
    {
        return false;
    }
    let Some((back, front)) = kenar_okuma_araligi(mode, width, height, angle) else {
        return false;
    };
    let Some(back_bytes) = back.checked_mul(bpp) else {
        return false;
    };
    let Some(front_bytes) = (front + 1).checked_mul(bpp) else {
        return false;
    };
    let Some(edge_req) = edge.topleft.checked_add(front_bytes) else {
        return false;
    };
    let w_bytes = width * bpp;
    edge.topleft >= back_bytes
        && edge.bytes.len() >= edge_req
        && dst_stride >= w_bytes
        && ipred_bolge_sonu(height, dst_stride, w_bytes).is_some_and(|req| dst_len >= req)
}

/// P45: `cfl_ac[layout]` SIMD girişinin O(1) sözleşmesi (`layout`: 0 = 4:2:0,
/// 1 = 4:2:2, 2 = 4:4:4).
#[must_use]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn cfl_ac_sozlesmesi<const BPC: u32>(
    layout: usize,
    ac_len: usize,
    y_len: usize,
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
    bit_depth: BitDepth,
) -> bool {
    if bit_depth.bits() != BPC || layout >= N_CFL_AC_LAYOUTS {
        return false;
    }
    let bpp = if BPC == 8 { 1 } else { 2 };
    if BPC > 8 && !y_stride.is_multiple_of(2) {
        return false;
    }
    let ss_hor = usize::from(layout != 2);
    let ss_ver = usize::from(layout == 0);
    let h_step = 2 >> ss_hor;
    let v_step = 2 >> ss_ver;
    let w_max = 32 >> ss_hor;
    let h_max = 32 >> ss_ver;
    if !(4..=w_max).contains(&cw)
        || !cw.is_power_of_two()
        || !(4..=h_max).contains(&ch)
        || !ch.is_power_of_two()
        || cw > 4 * ch
        || ch > 4 * cw
        || !w_pad.is_multiple_of(h_step)
        || !h_pad.is_multiple_of(v_step)
        || !w_pad.checked_mul(4).is_some_and(|p| p < cw)
        || !h_pad.checked_mul(4).is_some_and(|p| p < ch)
        || ac_len < cw * ch
    {
        return false;
    }
    let luma_w = cw << ss_hor;
    let luma_h = ch << ss_ver;
    let row_bytes = luma_w * bpp;
    y_stride >= row_bytes
        && ipred_bolge_sonu(luma_h, y_stride, row_bytes).is_some_and(|req| y_len >= req)
}

/// P45: `cfl_pred[mode]` SIMD girişinin O(1) sözleşmesi.
#[must_use]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn cfl_pred_sozlesmesi<const BPC: u32>(
    mode: CflPredMode,
    dst_len: usize,
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    ac_len: usize,
    alpha: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) -> bool {
    let _ = mode;
    let bpp = if BPC == 8 { 1 } else { 2 };
    let bpc_max = (1u32 << BPC) - 1;
    if bit_depth.bits() != BPC
        || u32::from(bitdepth_max) != bpc_max
        || !gecerli_blok_boyutu(width, height, 32)
        || !(-16..=16).contains(&alpha)
        || ac_len < width * height
        || (BPC > 8 && (!dst_stride.is_multiple_of(2) || !edge.topleft.is_multiple_of(2)))
    {
        return false;
    }
    let back_bytes = height * bpp;
    let front_bytes = (width + 1) * bpp;
    let Some(edge_req) = edge.topleft.checked_add(front_bytes) else {
        return false;
    };
    let w_bytes = width * bpp;
    edge.topleft >= back_bytes
        && edge.bytes.len() >= edge_req
        && dst_stride >= w_bytes
        && ipred_bolge_sonu(height, dst_stride, w_bytes).is_some_and(|req| dst_len >= req)
}

/// P45: `pal_pred` SIMD girişinin O(1) sözleşmesi.
#[must_use]
#[inline(always)]
pub fn pal_pred_sozlesmesi<const BPC: u32>(
    dst_len: usize,
    dst_stride: usize,
    pal_len: usize,
    idx_len: usize,
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) -> bool {
    let bpp = if BPC == 8 { 1 } else { 2 };
    bit_depth.bits() == BPC
        && gecerli_blok_boyutu(w, h, 64)
        && (BPC == 8 || dst_stride.is_multiple_of(2))
        && pal_len >= 8 * bpp
        && idx_len >= (w * h) / 2
        && dst_stride >= w * bpp
        && ipred_bolge_sonu(h, dst_stride, w * bpp).is_some_and(|req| dst_len >= req)
}

/// P45: Sözleşme dışı `intra_pred` girdisinde skaler devir sayacını artırır;
/// skaler gövdenin önkoşulları sağlanıyorsa skaler gövdeyi çağırır, değilse
/// paniklemeden döner.
#[cold]
#[allow(clippy::too_many_arguments)]
pub fn angular_skaler_devret<const BPC: u32>(
    mode: IntraPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    ipred_skaler_devri_say();
    if angular_sozlesmesi::<BPC>(
        mode,
        dst.len(),
        dst_stride,
        edge,
        width,
        height,
        angle,
        bit_depth,
        bitdepth_max,
    ) {
        scalar::intra_pred::<BPC>(
            mode,
            dst,
            dst_stride,
            edge,
            width,
            height,
            angle,
            max_width,
            max_height,
            bit_depth,
            bitdepth_max,
        );
    }
}

/// P45: Sözleşme dışı `cfl_ac` girdisinde skaler devir sayacını artırır;
/// skaler gövdeyi güvenli biçimde çalıştırır veya paniklemeden döner.
#[cold]
#[allow(clippy::too_many_arguments)]
pub fn cfl_ac_skaler_devret<const BPC: u32>(
    layout: usize,
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
    bit_depth: BitDepth,
) {
    ipred_skaler_devri_say();
    if cfl_ac_sozlesmesi::<BPC>(
        layout,
        ac.len(),
        y.len(),
        y_stride,
        w_pad,
        h_pad,
        cw,
        ch,
        bit_depth,
    ) {
        match layout {
            0 => scalar::cfl_ac_420::<BPC>(ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth),
            1 => scalar::cfl_ac_422::<BPC>(ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth),
            2 => scalar::cfl_ac_444::<BPC>(ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth),
            _ => {}
        }
    }
}

/// P45: Sözleşme dışı `cfl_pred` girdisinde skaler devir sayacını artırır;
/// skaler gövdeyi güvenli biçimde çalıştırır veya paniklemeden döner.
#[cold]
#[allow(clippy::too_many_arguments)]
pub fn cfl_pred_skaler_devret<const BPC: u32>(
    mode: CflPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    ac: &[i16],
    alpha: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    ipred_skaler_devri_say();
    if cfl_pred_sozlesmesi::<BPC>(
        mode,
        dst.len(),
        dst_stride,
        edge,
        width,
        height,
        ac.len(),
        alpha,
        bit_depth,
        bitdepth_max,
    ) {
        match mode {
            CflPredMode::Dc => scalar::cfl_pred_dc::<BPC>(
                dst,
                dst_stride,
                edge,
                width,
                height,
                ac,
                alpha,
                bit_depth,
                bitdepth_max,
            ),
            CflPredMode::LeftDc => scalar::cfl_pred_left_dc::<BPC>(
                dst,
                dst_stride,
                edge,
                width,
                height,
                ac,
                alpha,
                bit_depth,
                bitdepth_max,
            ),
            CflPredMode::TopDc => scalar::cfl_pred_top_dc::<BPC>(
                dst,
                dst_stride,
                edge,
                width,
                height,
                ac,
                alpha,
                bit_depth,
                bitdepth_max,
            ),
            CflPredMode::Dc128 => scalar::cfl_pred_dc_128::<BPC>(
                dst,
                dst_stride,
                edge,
                width,
                height,
                ac,
                alpha,
                bit_depth,
                bitdepth_max,
            ),
        }
    }
}

/// P45: Sözleşme dışı `pal_pred` girdisinde skaler devir sayacını artırır;
/// skaler gövdenin önkoşulları (paketlenmiş nibble'lar `0..=7` dâhil)
/// sağlanıyorsa skaler gövdeyi çağırır, değilse paniklemeden döner.
#[cold]
pub fn pal_pred_skaler_devret<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) {
    ipred_skaler_devri_say();
    if pal_pred_sozlesmesi::<BPC>(dst.len(), dst_stride, pal.len(), idx.len(), w, h, bit_depth)
        && idx[..(w * h) / 2].iter().all(|&b| b & 0x88 == 0)
    {
        scalar::pal_pred::<BPC>(dst, dst_stride, pal, idx, w, h, bit_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::{CflPredMode, IntraPredMode, N_IMPL_INTRA_PRED_MODES};

    #[test]
    fn mod_diskriminantlari_dav1d_ile_ayni() {
        assert_eq!(N_IMPL_INTRA_PRED_MODES, 14);
        for (index, mode) in IntraPredMode::ALL.iter().enumerate() {
            assert_eq!(mode.index(), index, "mod sırası bozuk: {mode:?}");
            assert_eq!(IntraPredMode::from_index(index), Some(*mode));
        }
        assert_eq!(IntraPredMode::from_index(14), None);
        assert_eq!(IntraPredMode::Dc.index(), 0);
        assert_eq!(IntraPredMode::Z1.index(), 6);
        assert_eq!(IntraPredMode::Filter.index(), 13);
        assert!(IntraPredMode::Z2.is_directional());
        assert!(!IntraPredMode::Dc.is_directional());
    }

    #[test]
    fn cfl_mod_indeksleri_dav1d_ile_ayni() {
        assert_eq!(CflPredMode::Dc.index(), 0);
        assert_eq!(CflPredMode::LeftDc.index(), 1);
        assert_eq!(CflPredMode::TopDc.index(), 2);
        assert_eq!(CflPredMode::Dc128.index(), 3);
        assert_eq!(CflPredMode::Dc.dav1d_index(), 0);
        assert_eq!(CflPredMode::LeftDc.dav1d_index(), 3);
        assert_eq!(CflPredMode::TopDc.dav1d_index(), 4);
        assert_eq!(CflPredMode::Dc128.dav1d_index(), 5);
    }

    #[test]
    fn p45_ipred_sozlesmeleri_ve_paniksiz_skaler_devir() {
        use super::{
            CflPredMode, Edge, IntraPredMode, angular_skaler_devret, angular_sozlesmesi,
            cfl_ac_skaler_devret, cfl_ac_sozlesmesi, cfl_pred_skaler_devret, cfl_pred_sozlesmesi,
            ipred_skaler_devir_sayisi, pal_pred_skaler_devret, pal_pred_sozlesmesi,
        };
        use crate::frame::BitDepth;

        let edge_buf = [0u8; 64];
        let edge8 = Edge::new(&edge_buf, 8);
        assert!(angular_sozlesmesi::<8>(
            IntraPredMode::Dc,
            16,
            4,
            edge8,
            4,
            4,
            0,
            BitDepth::Eight,
            255,
        ));
        // Kısa dst reddedilir.
        assert!(!angular_sozlesmesi::<8>(
            IntraPredMode::Dc,
            15,
            4,
            edge8,
            4,
            4,
            0,
            BitDepth::Eight,
            255,
        ));
        // FILTER_PRED 64x64 ve filt_idx >= 5 reddedilir.
        assert!(!angular_sozlesmesi::<8>(
            IntraPredMode::Filter,
            64 * 64,
            64,
            Edge::new(&[0u8; 256], 64),
            64,
            64,
            0,
            BitDepth::Eight,
            255,
        ));
        assert!(!angular_sozlesmesi::<8>(
            IntraPredMode::Filter,
            16,
            4,
            edge8,
            4,
            4,
            5,
            BitDepth::Eight,
            255,
        ));
        // 16bpc'de tek stride ve tek topleft reddedilir.
        assert!(!angular_sozlesmesi::<10>(
            IntraPredMode::Dc,
            32,
            9,
            Edge::new(&edge_buf, 16),
            4,
            4,
            0,
            BitDepth::Ten,
            1023,
        ));
        assert!(!angular_sozlesmesi::<10>(
            IntraPredMode::Dc,
            32,
            8,
            Edge::new(&edge_buf, 15),
            4,
            4,
            0,
            BitDepth::Ten,
            1023,
        ));

        // cfl_ac sözleşmesi.
        assert!(cfl_ac_sozlesmesi::<8>(
            0,
            16,
            64,
            8,
            0,
            0,
            4,
            4,
            BitDepth::Eight
        ));
        assert!(!cfl_ac_sozlesmesi::<8>(
            0,
            16,
            64,
            8,
            1,
            0,
            4,
            4,
            BitDepth::Eight
        ));
        assert!(!cfl_ac_sozlesmesi::<10>(
            0,
            16,
            128,
            15,
            0,
            0,
            4,
            4,
            BitDepth::Ten
        ));

        // cfl_pred sözleşmesi.
        assert!(cfl_pred_sozlesmesi::<8>(
            CflPredMode::Dc,
            16,
            4,
            edge8,
            4,
            4,
            16,
            16,
            BitDepth::Eight,
            255,
        ));
        assert!(!cfl_pred_sozlesmesi::<8>(
            CflPredMode::Dc,
            16,
            4,
            edge8,
            4,
            4,
            16,
            17,
            BitDepth::Eight,
            255,
        ));

        // pal_pred sözleşmesi.
        assert!(pal_pred_sozlesmesi::<8>(16, 4, 8, 8, 4, 4, BitDepth::Eight));
        assert!(!pal_pred_sozlesmesi::<8>(
            16,
            4,
            7,
            8,
            4,
            4,
            BitDepth::Eight
        ));
        assert!(!pal_pred_sozlesmesi::<10>(
            32,
            9,
            16,
            8,
            4,
            4,
            BitDepth::Ten
        ));

        // Paniksiz skaler devir ve sayaç artışı.
        let once = ipred_skaler_devir_sayisi();
        let mut kisa_dst = [0u8; 4];
        angular_skaler_devret::<8>(
            IntraPredMode::Dc,
            &mut kisa_dst,
            4,
            edge8,
            4,
            4,
            0,
            0,
            0,
            BitDepth::Eight,
            255,
        );
        let mut kisa_ac = [0i16; 4];
        cfl_ac_skaler_devret::<8>(0, &mut kisa_ac, &edge_buf, 8, 0, 0, 4, 4, BitDepth::Eight);
        cfl_pred_skaler_devret::<8>(
            CflPredMode::Dc,
            &mut kisa_dst,
            4,
            edge8,
            4,
            4,
            &kisa_ac,
            0,
            BitDepth::Eight,
            255,
        );
        pal_pred_skaler_devret::<8>(
            &mut kisa_dst,
            4,
            &edge_buf[..8],
            &edge_buf[..8],
            4,
            4,
            BitDepth::Eight,
        );
        assert!(ipred_skaler_devir_sayisi() >= once + 4);
    }
}
