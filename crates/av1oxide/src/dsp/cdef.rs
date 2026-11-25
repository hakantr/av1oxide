//! CDEF (Constrained Directional Enhancement Filter) çekirdek sözleşmesi — S15.4 pilotu.
//!
//! Bu modül **davranışı** tanımlar, gövde içermez: sabitler, parametre tipleri ve
//! çekirdek işaretçisi imzaları. Gövdeler [`crate::dsp::scalar::cdef`] (tüm bit
//! derinlikleri) ve [`crate::dsp::x86::cdef`] (8 bpc ve 16 bpc AVX2) içindedir.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! Aşağıdaki tablo, taşınan **her** kuralın dav1d kaynağındaki satırını verir.
//! Çelişki hâlinde kaynak esastır; bu tablo denetim izidir.
//!
//! | Kural | Kaynak |
//! |---|---|
//! | Kenar bayrakları: `HAVE_LEFT=1<<0`, `HAVE_RIGHT=1<<1`, `HAVE_TOP=1<<2`, `HAVE_BOTTOM=1<<3` | `src/cdef.h:36-41` |
//! | Çekirdek imzaları (`dir`, `fb[3]`) | `src/cdef.h:53-67` |
//! | `fb[0]=8x8`, `fb[1]=4x8`, `fb[2]=4x4` | `src/cdef_tmpl.c:323-325` |
//! | `tmp_stride = 12`, `tmp_buf[144]`, `tmp = buf + 2*12 + 2` | `src/cdef_tmpl.c:112-115` |
//! | Padding sınırları `x_start=-2, x_end=w+2, y_start=-2, y_end=h+2` | `src/cdef_tmpl.c:63` |
//! | Eksik kenarda dolgu değeri `INT16_MIN` ("unsigned'da büyük pozitif, signed'da büyük negatif") | `src/cdef_tmpl.c:44-54` |
//! | Kenar yoksa dolgu: TOP→satır −2,−1; BOTTOM→satır h,h+1; LEFT→sütun −2,−1; RIGHT→sütun w,w+1 | `src/cdef_tmpl.c:64-79` |
//! | Dolgu sonrası kopyalama sırası: top satırları → left sütunları → dst → bottom satırları | `src/cdef_tmpl.c:81-100` |
//! | `constrain(diff,thr,shift) = apply_sign(min(\|diff\|, max(0, thr − (\|diff\| >> shift))), diff)` | `src/cdef_tmpl.c:37-42` |
//! | `pri_tap = 4 − ((pri_strength >> bitdepth_min_8) & 1)` | `src/cdef_tmpl.c:122` |
//! | `pri_shift = max(0, damping − ulog2(pri_strength))` | `src/cdef_tmpl.c:123` |
//! | `sec_shift = damping − ulog2(sec_strength)` (clamp'sız; negatif değer C'de UB) | `src/cdef_tmpl.c:125` |
//! | `pri_tap_k`: k=0 → `pri_tap`, k=1 → `(pri_tap & 3) \| 2` | `src/cdef_tmpl.c:139` |
//! | `sec_tap = 2 − k` (2, sonra 1) | `src/cdef_tmpl.c:151` |
//! | Yuvarlama: `(sum − (sum < 0) + 8) >> 4` (aritmetik kaydırma) | `src/cdef_tmpl.c:165,184,210` |
//! | Öncelikli+ikincil dalda kırpma `iclip(v, min, max)`; min `umin` (unsigned), max `imax` (signed) | `src/cdef_tmpl.c:165` |
//! | Yalnız-öncelikli ve yalnız-ikincil dallarda **kırpma yok**, `pixel` tipine dönüşüm (8-bit'te mod 256) | `src/cdef_tmpl.c:184,210` |
//! | Yön taraması `px = (img >> bitdepth_min_8) − 128`, 8x8 blok | `src/cdef_tmpl.c:242,249` |
//! | Yön maliyet ağırlıkları `div_table = {840,420,280,210,168,140,120}`, merkez 105 | `src/cdef_tmpl.c:271,279-280` |
//! | En iyi yön: `>` ile **katı** karşılaştırma (eşitlikte küçük indeks kazanır) | `src/cdef_tmpl.c:294-301` |
//! | `var = (best_cost − cost[best_dir ^ 4]) >> 10` | `src/cdef_tmpl.c:303` |
//! | Yön tablosu `dav1d_cdef_directions[12][2]` (12 = 2+8+2, `dir` 0..7 → `dir`, `dir+2`, `dir+4`) | `src/tables.c:400-414` |
//! | x86 dispatch: AVX2 varsa `dir` + `fb[0..2]` AVX2'ye atanır (yalnız `ARCH_X86_64`) | `src/x86/cdef.h:73-79` |
//!
//! ## Bit derinliğine bağlı kurallar
//!
//! - `bitdepth_min_8 = bitdepth_from_max(bitdepth_max) − 8` (`cdef_tmpl.c:121,242`):
//!   8-bit → 0, 10-bit → 2, 12-bit → 4.
//! - `pri_tap` **yalnız** bu kaydırma ile değişir (`cdef_tmpl.c:122`); 3 ya da 4 olur.
//! - `px = (img >> bitdepth_min_8) − 128` (`cdef_tmpl.c:249`): üç bit derinliğinde de
//!   `px ∈ [−128, 127]` olur, bu yüzden yön maliyeti taşması 32-bit'te sınırlıdır.
//! - Dolgu değeri `INT16_MIN` **bit derinliğinden bağımsız** aynıdır; `constrain`
//!   içinde `|diff| >> shift ≥ 512 > thr` olduğu için katkısı her zaman **sıfırdır**,
//!   `umin`'de (unsigned) büyük, `imax`'te (signed) küçük kaldığı için min/max'i
//!   **değiştirmez**. SIMD yolunda bu özellik aynen kullanılır (maske gerekmez).
//! - `sec_shift` negatif olursa C'de kaydırma **tanımsızdır**; bu port
//!   [`clamped_shift`] ile 0'a kırpar ve sözleşme gereği `damping >= ulog2(sec_strength)`
//!   kabul eder (S15.4 test üreteci bu koşulu uygular).
//!
//! ## Bellek sözleşmesi (7(c) sentinel/over-read testinin dayanağı)
//!
//! `dir` girdisi 8x8 pikseli **tam** okur: `img.len() >= 7*stride + 8*bpp`.
//! `filter` girdisi C'nin okuduğu aralıkların tamamını gerektirir:
//!
//! | Tampon | Gereken |
//! |---|---|
//! | `dst` | `len >= (h−1)*stride + (w+2)*bpp` (`HAVE_RIGHT` iken `dst[w]`,`dst[w+1]` de okunur) |
//! | `top` | satır 0 = blok satırı **−2**; `len >= stride + (w+4)*bpp` |
//! | `bottom` | satır 0 = blok satırı **h**; `len >= stride + (w+4)*bpp` |
//! | `left` | `len >= h*2*bpp`; `left[y][0..2]` |
//!
//! `top`/`bottom` dilimlerinde indeks 0 ↔ mantıksal `x = −2`'dir; `dst`/`left`
//! indeksleri mantıksal `x = 0`'dan başlar. `stride` **bayt** cinsindendir ve üç
//! tamponda da aynı satır adımıdır (dav1d `pixel (*left)[2]` + `PXSTRIDE` düzeni).

#![forbid(unsafe_code)]

/// CDEF kenar bayrağı bitleri (dav1d `src/cdef.h:36-41`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CdefEdgeFlags(u8);

impl CdefEdgeFlags {
    /// Kenar yok: dört yön de dolgu ile kapatılır.
    pub const NONE: Self = Self(0);
    /// Sol komşu blok filtrelenmiş veri olarak mevcut (`1 << 0`).
    pub const HAVE_LEFT: Self = Self(1 << 0);
    /// Sağ komşu blok mevcut (`1 << 1`).
    pub const HAVE_RIGHT: Self = Self(1 << 1);
    /// Üst komşu blok mevcut (`1 << 2`).
    pub const HAVE_TOP: Self = Self(1 << 2);
    /// Alt komşu blok mevcut (`1 << 3`).
    pub const HAVE_BOTTOM: Self = Self(1 << 3);
    /// Dört kenarın birleşimi (`0b1111`) — filtre/AVX2 için "hızlı yol".
    pub const ALL: Self = Self(0b1111);

    /// Bilinen bitlerden küme kurar; bilinmeyen bit varsa `None`.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::ALL.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Ham bit deseni.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// `other`'ın **tüm** bitleri bu kümede var mı?
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Birleşim.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Desteklenen bit derinlikleri (S15.4 kapsamı: 8, 10, 12).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CdefBitDepth {
    /// 8 bit/piksel.
    B8,
    /// 10 bit/piksel (16bpc depoda).
    B10,
    /// 12 bit/piksel (16bpc depoda).
    B12,
}

impl CdefBitDepth {
    /// Piksel başına bayt: 8-bit → 1, 16bpc → 2.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::B8 => 1,
            Self::B10 | Self::B12 => 2,
        }
    }

    /// `bitdepth_min_8 = bitdepth − 8` (dav1d `bitdepth_from_max(bitdepth_max) - 8`).
    #[must_use]
    pub const fn min_8(self) -> u32 {
        match self {
            Self::B8 => 0,
            Self::B10 => 2,
            Self::B12 => 4,
        }
    }

    /// Bit cinsinden derinlik.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::B8 => 8,
            Self::B10 => 10,
            Self::B12 => 12,
        }
    }

    /// Bu derinlikte öncelikli/ikincil gücün en büyük geçerli değeri
    /// (dav1d `cdef_apply_tmpl.c:167-170`: `lvl << bitdepth_min_8`).
    #[must_use]
    pub const fn max_pri_strength(self) -> u8 {
        15u8 << self.min_8()
    }

    /// İkincil gücün en büyük geçerli değeri (`{0,1,2,4} << bitdepth_min_8`).
    #[must_use]
    pub const fn max_sec_strength(self) -> u8 {
        4u8 << self.min_8()
    }

    /// En büyük piksel değeri (8-bit → 255, 10-bit → 1023, 12-bit → 4095).
    #[must_use]
    pub const fn max_pixel(self) -> u16 {
        (1u16 << self.bits()) - 1
    }
}

/// CDEF blok boyutu; dav1d `fb[0..2]` yuvalarının sırası (dav1d `src/cdef_tmpl.c:323-325`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CdefBlockSize {
    /// `fb[0]` — 8x8 (luma).
    W8H8,
    /// `fb[1]` — 4x8 (4:2:2 kroma).
    W4H8,
    /// `fb[2]` — 4x4 (4:2:0 kroma).
    W4H4,
}

impl CdefBlockSize {
    /// Genişlik (piksel).
    #[must_use]
    pub const fn w(self) -> usize {
        match self {
            Self::W8H8 => 8,
            Self::W4H8 | Self::W4H4 => 4,
        }
    }

    /// Yükseklik (piksel).
    #[must_use]
    pub const fn h(self) -> usize {
        match self {
            Self::W8H8 | Self::W4H8 => 8,
            Self::W4H4 => 4,
        }
    }

    /// `fb` yuva indeksi (0..2).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::W8H8 => 0,
            Self::W4H8 => 1,
            Self::W4H4 => 2,
        }
    }

    /// Üç blok boyutunun tamamı (test taraması için tek kaynak).
    pub const ALL: [Self; 3] = [Self::W8H8, Self::W4H8, Self::W4H4];
}

/// `dir` çekirdeğinin sonucu: yön (0..7) ve varyans (`var`, 32-bit).
///
/// `var` **u32**'dir: en büyük değer `880_803_840 >> 10 = 860_160` olur ve
/// 16 bite sığmaz (dav1d `unsigned *var`, `src/cdef_tmpl.c:240,303`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CdefDirResult {
    /// Seçilen yön, 0..7 (dav1d `cdef_find_dir_c` dönüşü).
    pub dir: u8,
    /// Yön güvenilirliği: `(best_cost − cost[best_dir ^ 4]) >> 10`.
    pub var: u32,
}

/// CDEF filtre çağrısının tüm sayısal parametreleri.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CdefFilterParams {
    /// Bit derinliği (piksel deposunun yorumu).
    pub bit_depth: CdefBitDepth,
    /// Blok boyutu (`fb` yuvası).
    pub size: CdefBlockSize,
    /// Öncelikli güç (dav1d `pri_strength`, `≤ 15 << bitdepth_min_8`).
    pub pri_strength: u8,
    /// İkincil güç (dav1d `sec_strength`, `≤ 4 << bitdepth_min_8`).
    pub sec_strength: u8,
    /// Yön, 0..7 (dav1d `dir` — yön taramasının çıktısı).
    pub dir: u8,
    /// Sönümleme (dav1d `damping`; `sec_shift` için `damping >= ulog2(sec_strength)` şart).
    pub damping: u8,
    /// Kenar bayrakları.
    pub edges: CdefEdgeFlags,
}

/// `dir` çekirdeği imzası (S15.6 karar kaydının girdisi).
///
/// `img`: 8x8 blok, `stride` bayt satır adımı; `bit_depth` piksel yorumu.
pub type CdefDirFn = fn(img: &[u8], stride: usize, bit_depth: CdefBitDepth) -> CdefDirResult;

/// `filter` çekirdeği imzası (S15.6 karar kaydının girdisi).
///
/// Bellek sözleşmesi modül başlığındaki tablodadır. Dönüş değeri yoktur;
/// sonuç `dst` içine **yerinde** yazılır (dav1d `decl_cdef_fn`, `src/cdef.h:53-57`).
pub type CdefFilterFn = fn(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
);

/// CDEF çekirdek ailesi — dav1d `Dav1dCdefDSPContext` (`src/cdef.h:63-66`):
/// `dir` + `fb[3]` (`fb[0]` = 8×8, `fb[1]` = 4×8, `fb[2]` = 4×4).
///
/// Her bit derinliğinin kendi tablosu vardır (F3c, ADR-0003 §2.1). Skaler
/// `fb[0..2]` aynı monomorfu taşır; blok boyutu `params.size` ile verilir ve
/// çağıran yuvayı `CdefBlockSize` ile seçer.
#[derive(Clone, Copy)]
pub struct CdefDsp {
    /// `dir`.
    pub dir: CdefDirFn,
    /// `fb[0..2]`.
    pub fb: [CdefFilterFn; 3],
}

impl CdefDsp {
    /// `BPC` bit derinliğinin skaler tablosu.
    #[must_use]
    pub const fn scalar<const BPC: u32>() -> Self {
        let fb: CdefFilterFn = crate::dsp::scalar::cdef::filter::<BPC>;
        Self {
            dir: crate::dsp::scalar::cdef::find_dir::<BPC>,
            fb: [fb; 3],
        }
    }

    /// Çalışma zamanı bit derinliğinin skaler tablosu (kurulum ve testler).
    #[must_use]
    pub const fn scalar_for(bit_depth: CdefBitDepth) -> Self {
        match bit_depth {
            CdefBitDepth::B8 => Self::scalar::<8>(),
            CdefBitDepth::B10 => Self::scalar::<10>(),
            CdefBitDepth::B12 => Self::scalar::<12>(),
        }
    }
}

impl std::fmt::Debug for CdefDsp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CdefDsp { dir: fn, fb: [fn; 3] }")
    }
}

/// Yön tablosu: `dav1d_cdef_directions[2 + 8 + 2][2]` (dav1d `src/tables.c:400-414`).
///
/// Değerler `y * 12 + x` biçiminde **paketlenmiş** ofsetlerdir; `12` padding
/// satır adımıdır (dav1d `src/cdef_tmpl.c:112`). `dir` 0..7 için erişim
/// `[dir]`, `[dir+2]`, `[dir+4]`'tür.
// Tablo dav1d kaynağındaki yazımla **birebir** tutulur (`1 * 12 + 0` vb.),
// böylece satır satır denetlenebilir; `identity_op` bu yüzden bilinçli susturulur.
#[allow(clippy::identity_op, clippy::neg_multiply, clippy::erasing_op)]
pub const CDEF_DIRECTIONS: [[i16; 2]; 12] = [
    [1 * 12 + 0, 2 * 12 + 0],   // 6
    [1 * 12 + 0, 2 * 12 - 1],   // 7
    [-1 * 12 + 1, -2 * 12 + 2], // 0
    [0 * 12 + 1, -1 * 12 + 2],  // 1
    [0 * 12 + 1, 0 * 12 + 2],   // 2
    [0 * 12 + 1, 1 * 12 + 2],   // 3
    [1 * 12 + 1, 2 * 12 + 2],   // 4
    [1 * 12 + 0, 2 * 12 + 1],   // 5
    [1 * 12 + 0, 2 * 12 + 0],   // 6
    [1 * 12 + 0, 2 * 12 - 1],   // 7
    [-1 * 12 + 1, -2 * 12 + 2], // 0
    [0 * 12 + 1, -1 * 12 + 2],  // 1
];

/// Yön maliyetinde kullanılan bölen tablosu (dav1d `src/cdef_tmpl.c:271`).
pub const DIV_TABLE: [u32; 7] = [840, 420, 280, 210, 168, 140, 120];

/// Yön taramasının merkez/ofset ağırlığı (dav1d `src/cdef_tmpl.c:268,279-280,286`).
pub const DIR_CENTER_WEIGHT: u32 = 105;

/// Padding satır adımı (dav1d `tmp_stride = 12`, `src/cdef_tmpl.c:112`).
pub(crate) const TMP_STRIDE: usize = 12;

/// Padding tamponunun uzunluğu (`12*12`, dav1d `tmp_buf[144]`, `src/cdef_tmpl.c:114`).
pub(crate) const TMP_BUF_LEN: usize = 144;

/// `tmp` işaretçisinin tampon başına göre konumu (`2 * 12 + 2`, `src/cdef_tmpl.c:115`).
pub(crate) const TMP_BASE: usize = 2 * TMP_STRIDE + 2;

/// Eksik kenarın dolgu değeri (`INT16_MIN`, dav1d `src/cdef_tmpl.c:51`).
pub(crate) const TMP_FILL: i16 = i16::MIN;

/// Mantıksal `(x, y)` konumunun `tmp` tamponundaki indeksi.
///
/// Okuma penceresi `x ∈ [−2, w+2)`, `y ∈ [−2, h+2)` ve `w, h ≤ 8` olduğundan
/// sonuç **her zaman** `[0, 144)` içindedir: en küçük `26 + (−2)·12 + (−2) = 0`,
/// en büyük `26 + 9·12 + 9 = 143`. SIMD yolunda bu sınır kanıtı, sınır denetimi
/// üretmeyen (`get_unchecked`) erişimin `// SAFETY:` gerekçesidir.
#[must_use]
pub(crate) const fn tmp_index(x: i32, y: i32) -> usize {
    (TMP_BASE as i32 + y * TMP_STRIDE as i32 + x) as usize
}

/// `apply_sign(v, s)`: `s < 0` ise `−v`, değilse `v` (dav1d `intops.h:59-61`).
#[must_use]
pub const fn apply_sign(v: i32, s: i32) -> i32 {
    if s < 0 { -v } else { v }
}

/// `ulog2(v) = 31 ^ clz(v)`, `v > 0` (dav1d `intops.h:67-69`).
///
/// # Panics
///
/// `v == 0` için tanımsızdır (`clz(0)`); çağıran sözleşmesi gereği güç sıfırdan
/// farklıdır. Hata ayıklamada `debug_assert` ile yakalanır.
#[must_use]
pub(crate) const fn ulog2(v: u32) -> u32 {
    debug_assert!(v > 0, "ulog2(0) tanımsız (dav1d clz(0))");
    31 - v.leading_zeros()
}

/// Kaydırmayı `[0, 31]` aralığına kırpar.
///
/// C'de negatif kaydırma **tanımsızdır** (dav1d `src/cdef_tmpl.c:125` clamp'sız
/// `sec_shift`); bu port tanımlı davranışa indirger ve sözleşme gereği
/// `damping >= ulog2(sec_strength)` bekler.
#[must_use]
pub const fn clamped_shift(shift: i32) -> u32 {
    if shift < 0 {
        0
    } else if shift > 31 {
        31
    } else {
        shift as u32
    }
}

/// `constrain(diff, threshold, shift)` — dav1d `src/cdef_tmpl.c:37-42`.
///
/// `apply_sign(imin(adiff, imax(0, threshold - (adiff >> shift))), diff)`,
/// `adiff = |diff|`. Tüm ara değerler `i32`; taşma yoktur (`|diff| <= 2^15 + 2^12`).
#[must_use]
#[inline]
pub fn constrain(diff: i32, threshold: i32, shift: u32) -> i32 {
    let adiff = diff.wrapping_abs();
    let limit = (threshold - (adiff >> shift)).max(0);
    apply_sign(adiff.min(limit), diff)
}

/// Piksel okuma: `plane[byte_offset..]` baytlarını bit derinliğine göre yorumlar.
///
/// 8-bit'te tek bayt; 16bpc'de **yerel bayt sırasında** iki bayt (dav1d `pixel`
/// deposunun bellek görünümü). `bpc` derleme zamanı sabitidir, bu yüzden dal
/// monomorfizasyonda kaybolur.
#[must_use]
#[inline(always)]
pub fn px_at<const BPC: u32>(plane: &[u8], byte_offset: usize) -> u16 {
    if BPC == 8 {
        u16::from(plane[byte_offset])
    } else {
        u16::from_ne_bytes([plane[byte_offset], plane[byte_offset + 1]])
    }
}

/// Piksel yazma — [`px_at`]'in tersi.
///
/// 8-bit'te değer **mod 256** kırpılır: dav1d `dst[x] = int` yazımı `uint8_t`
/// dönüşümüdür (`src/cdef_tmpl.c:184,210`), yani taşma tanımlıdır ve taklit edilir.
#[inline(always)]
pub fn px_put<const BPC: u32>(plane: &mut [u8], byte_offset: usize, value: u16) {
    if BPC == 8 {
        plane[byte_offset] = value as u8;
    } else {
        plane[byte_offset..byte_offset + 2].copy_from_slice(&value.to_ne_bytes());
    }
}
