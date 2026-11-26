//! Hareket telafisi (MC) çekirdek sözleşmesi — S15.5 pilotu.
//!
//! Bu modül **davranışı ve bellek sözleşmesini** tanımlar, gövde içermez:
//! sabitler, `put_8tap_regular` filtre tabloları, bit derinliği tipi ve çekirdek
//! işaretçisi imzası. Gövdeler [`crate::dsp::scalar::mc`] (8-bit + 16bpc) ve
//! `crate::dsp::x86::mc` (8-bit AVX2) içindedir. Kapsam **yalnız**
//! `FILTER_2D_8TAP_REGULAR` varyantıdır (`prep`/mct, diğer filtre türleri,
//! scaled MC ayrı yuvadadır — S15.5 kapsamı).
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! Taşınan **her** kuralın dav1d kaynağındaki satırı. Çelişki hâlinde kaynak
//! esastır; bu tablo denetim izidir.
//!
//! | Kural | Kaynak |
//! |---|---|
//! | 8-bit: `intermediate_bits = 4`, `PREP_BIAS = 0` | `src/mc_tmpl.c:39-42` |
//! | 16bpc: `intermediate_bits = 14 − bitdepth_from_max(bitdepth_max)` (10-bit → 4, 12-bit → 2); `PREP_BIAS = 8192` **yalnız `prep`** içindir | `src/mc_tmpl.c:44-48` |
//! | `put_c`: satır satır `pixel_copy(dst, src, w)` (kopya yolu) | `src/mc_tmpl.c:51-60` |
//! | `FILTER_8TAP(src,x,F,stride) = F[0]·src[x−3·stride] + … + F[7]·src[x+4·stride]` | `src/mc_tmpl.c:77-85` |
//! | `DAV1D_FILTER_8TAP_RND = (sum + ((1<<sh)>>1)) >> sh` | `src/mc_tmpl.c:97-98` |
//! | `DAV1D_FILTER_8TAP_RND2 = (sum + rnd) >> sh` (açık yuvarlama sabiti) | `src/mc_tmpl.c:100-101` |
//! | `…_CLIP` / `…_CLIP2` sonucu `iclip_pixel` ile kırpar | `src/mc_tmpl.c:106-111` |
//! | `iclip_pixel(x) = iclip(x, 0, bitdepth_max)` (16bpc); 8-bit'te `iclip_u8` | `include/common/bitdepth.h:49,71` |
//! | `pixel` tipi: 8-bit `uint8_t`, 16bpc `uint16_t`; `PXSTRIDE(x) = x >> 1` (16bpc) | `include/common/bitdepth.h:31-79` |
//! | `GET_H_FILTER(mx)`: `mx == 0` → yok (kopya); `w > 4` → tablo `filter_type & 3`, aksi hâlde `3 + (filter_type & 1)` | `src/mc_tmpl.c:115-118` |
//! | `GET_V_FILTER(my)`: `my == 0` → yok; `h > 4` → tablo `filter_type >> 2`, aksi hâlde `3 + ((filter_type >> 2) & 1)` | `src/mc_tmpl.c:120-123` |
//! | `FILTER_2D_8TAP_REGULAR = 0` (enum'un ilk değeri) | `src/levels.h:184-195` |
//! | `put_8tap_regular_c` → `put_8tap_c(..., type_h \| type_v<<2)` → `filter_type = 0` | `src/mc_tmpl.c:365-370,511-520` |
//! | `intermediate_rnd = 32 + ((1 << (6 − intermediate_bits)) >> 1)` (yalnız yatay yol) | `src/mc_tmpl.c:136` |
//! | 2D ara tampon `int16_t mid[128*135]`, `tmp_h = h + 7`, `src -= 3·src_stride` | `src/mc_tmpl.c:144-158` |
//! | 2D yatay: `mid = (Σ F·src + ((1 << (6−ib)) >> 1)) >> (6−ib)` | `src/mc_tmpl.c:149-151` |
//! | 2D dikey: `dst = iclip_pixel((Σ F·mid + ((1 << (6+ib)) >> 1)) >> (6+ib))` | `src/mc_tmpl.c:160-172` |
//! | Yalnız yatay: `dst = iclip_pixel((Σ F·src + intermediate_rnd) >> 6)` | `src/mc_tmpl.c:173-181` |
//! | Yalnız dikey: `dst = iclip_pixel((Σ F·src + 32) >> 6)` | `src/mc_tmpl.c:182-185` |
//! | Hiç filtre yoksa `put_c` (kopya) | `src/mc_tmpl.c:186` |
//! | 8-tap "regular" katsayı tablosu (15 alt piksel konumu) | `src/tables.c:445-461` |
//! | `w ≤ 4` için 4-tap "regular" tablosu (`[3 + DAV1D_FILTER_8TAP_REGULAR]`) | `src/tables.c:497-516` |
//! | x86 AVX2 dispatch: AVX2 varsa 8-tap `mc[]` yuvaları AVX2'ye atanır | `src/x86/mc.h:92-105` |
//!
//! ## Bit derinliğine bağlı kurallar
//!
//! | Bit derinliği | `bpp` | `ib` | yatay yalnız | 2D yatay | 2D dikey | dikey yalnız |
//! |---|---|---|---|---|---|---|
//! | 8 | 1 | 4 | `(Σ+34)>>6` | `(Σ+2)>>2` | `(Σ+512)>>10` | `(Σ+32)>>6` |
//! | 10 | 2 | 4 | `(Σ+34)>>6` | `(Σ+2)>>2` | `(Σ+512)>>10` | `(Σ+32)>>6` |
//! | 12 | 2 | 2 | `(Σ+40)>>6` | `(Σ+8)>>4` | `(Σ+128)>>8` | `(Σ+32)>>6` |
//!
//! Ara değerler `int16_t`'ya sığar (dav1d `src/mc_tmpl.c:144-158`'deki `int16_t
//! mid`): 12-bit'te bile 2D yatay çıktı en çok `(4095·75 + 8) >> 4 = 19195`,
//! en az `(4095·(−13)) >> 4 = −3327`. Bu port aynı taşıma sınırını
//! `debug_assert` ile denetler; taşma yoktur.
//!
//! ## Bellek sözleşmesi (8(a)/8(c) testlerinin dayanağı)
//!
//! Çekirdek dav1d `put_8tap_c` gibi **kaynak bloğun sol üst pikselini** değil,
//! ondan **3 satır yukarı ve 3 sütun soldan başlayan dolgu hücresini** görür:
//!
//! ```text
//! src (piksel 0)  = blok satırı −3, blok sütunu −3   [dolgu başlangıcı]
//! src[3·src_stride + 3·bpp ..] = blok satırı 0, sütun 0  [blok başlangıcı]
//! dst (piksel 0)  = blok satırı 0, sütun 0
//! ```
//!
//! - `src_stride` / `dst_stride` **bayt** cinsindendir (dav1d `PXSTRIDE` ile
//!   böler; bu port bayt indekslemesi kullanır).
//! - Blok `w ∈ [1, 128]`, `h ∈ [1, 128]`; dav1d blok boyutu tablosunun
//!   (`src/levels.h:154-177`) genişlik/yükseklik değerleri 2, 4, 8, 16, 32, 64,
//!   128'dir.
//! - `mx, my ∈ [0, 15]` (1/16 alt piksel konumu). `0` = filtre yok (o eksende
//!   kopya); `1..=15` tabloda `mx−1`/`my−1` satırıdır.
//! - Kaynak okuma penceresi: satır `−3..=h+3`, sütun `−3..=w+3` → toplam
//!   `h+7` satır × `w+7` piksel. Gereken dilim uzunluğu
//!   `(h+6)·src_stride + (w+7)·bpp`; `src_stride >= (w+7)·bpp` olmalıdır.
//! - Hedef yazma penceresi: `h` satır × `w` piksel; gereken dilim uzunluğu
//!   `(h−1)·dst_stride + w·bpp`; `dst_stride >= w·bpp`.
//!
//! Sözleşme dışı girdide skaler yol **panikler** (güvenli Rust indekslemesi);
//! AVX2 güvenli sarmalayıcısı ise AVX2 yokmuş gibi skaler yola düşer. Bu, S15.4
//! CDEF sözleşmesiyle aynı ilkedir: ISA yolu sınır dışı okuma yapmaz, yanlış
//! girdide bile bellek güvenliği korunur.
//!
//! ## Neden `put_8tap_regular` adı?
//!
//! dav1d sembolü `dav1d_put_8tap_regular_8bpc_c` / `…_16bpc_c`'dir. Bu port
//! tek bir `const BPC: u32` ile monomorfize edildiği için bit derinliği
//! parametredir; ad dav1d'dekiyle aynı bırakılmıştır (denetim izi).

#![forbid(unsafe_code)]

use crate::dsp::scalar::mc as scalar_mc;

/// Bir `put_8tap` çağrısının alt piksel konum sayısı (1/16).
pub const MC_SUBPEL_POSITIONS: u8 = 16;
/// Kenar dolgusu: bloğun solunda/üstünde **3**, sağında/altında **4** piksel.
pub const MC_EDGE: usize = 3;
/// Desteklenen en büyük blok genişliği (dav1d `BS_128x128`).
pub const MC_MAX_W: usize = 128;
/// Desteklenen en büyük blok yüksekliği (dav1d `BS_128x128`).
pub const MC_MAX_H: usize = 128;
/// 8-tap filtrenin katsayı sayısı.
pub const MC_TAPS: usize = 8;

/// 8-tap "regular" skaler filtre katsayıları — dav1d `src/tables.c:445-461`.
///
/// Satır `i` = alt piksel konumu `mx = i+1` (1/16). Katsayı toplamı 64'tür.
pub const MC_8TAP_REGULAR: [[i8; MC_TAPS]; 15] = [
    [0, 1, -3, 63, 4, -1, 0, 0],
    [0, 1, -5, 61, 9, -2, 0, 0],
    [0, 1, -6, 58, 14, -4, 1, 0],
    [0, 1, -7, 55, 19, -5, 1, 0],
    [0, 1, -7, 51, 24, -6, 1, 0],
    [0, 1, -8, 47, 29, -6, 1, 0],
    [0, 1, -7, 42, 33, -6, 1, 0],
    [0, 1, -7, 38, 38, -7, 1, 0],
    [0, 1, -6, 33, 42, -7, 1, 0],
    [0, 1, -6, 29, 47, -8, 1, 0],
    [0, 1, -6, 24, 51, -7, 1, 0],
    [0, 1, -5, 19, 55, -7, 1, 0],
    [0, 1, -4, 14, 58, -6, 1, 0],
    [0, 0, -2, 9, 61, -5, 1, 0],
    [0, 0, -1, 4, 63, -3, 1, 0],
];

/// `w ≤ 4` (yatay) / `h ≤ 4` (dikey) için 4-tap "regular" katsayıları —
/// dav1d `src/tables.c:497-516` (`dav1d_mc_subpel_filters[3]`).
///
/// İlk iki ve son iki katsayı sıfırdır; `FILTER_8TAP` yine 8 terim toplar,
/// ama etkin pencere dardır. dav1d dar bloklarda bu tabloyu seçer
/// (`src/mc_tmpl.c:115-123`).
pub const MC_4TAP_REGULAR: [[i8; MC_TAPS]; 15] = [
    [0, 0, -2, 63, 4, -1, 0, 0],
    [0, 0, -4, 61, 9, -2, 0, 0],
    [0, 0, -5, 58, 14, -3, 0, 0],
    [0, 0, -6, 55, 19, -4, 0, 0],
    [0, 0, -6, 51, 24, -5, 0, 0],
    [0, 0, -7, 47, 29, -5, 0, 0],
    [0, 0, -6, 42, 33, -5, 0, 0],
    [0, 0, -6, 38, 38, -6, 0, 0],
    [0, 0, -5, 33, 42, -6, 0, 0],
    [0, 0, -5, 29, 47, -7, 0, 0],
    [0, 0, -5, 24, 51, -6, 0, 0],
    [0, 0, -4, 19, 55, -6, 0, 0],
    [0, 0, -3, 14, 58, -5, 0, 0],
    [0, 0, -2, 9, 61, -4, 0, 0],
    [0, 0, -1, 4, 63, -2, 0, 0],
];

/// MC bit derinliği (dav1d `BITDEPTH` + `bitdepth_max` çiftinin portu).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum McBitDepth {
    /// 8-bit (`BITDEPTH=8`, `bitdepth_max=255`).
    B8,
    /// 10-bit (`BITDEPTH=16`, `bitdepth_max=1023`).
    B10,
    /// 12-bit (`BITDEPTH=16`, `bitdepth_max=4095`).
    B12,
}

impl McBitDepth {
    /// Desteklenen üç bit derinliği (testler ve raporlar için).
    pub const ALL: [Self; 3] = [Self::B8, Self::B10, Self::B12];

    /// Bit sayısı (8/10/12).
    #[must_use]
    pub const fn depth(self) -> u32 {
        match self {
            Self::B8 => 8,
            Self::B10 => 10,
            Self::B12 => 12,
        }
    }

    /// Bit sayısından tip (yalnız 8/10/12).
    #[must_use]
    pub const fn from_depth(depth: u32) -> Option<Self> {
        match depth {
            8 => Some(Self::B8),
            10 => Some(Self::B10),
            12 => Some(Self::B12),
            _ => None,
        }
    }

    /// Piksel başına bayt: 8-bit → 1, 16bpc → 2 (dav1d `pixel` tipi).
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        if matches!(self, Self::B8) { 1 } else { 2 }
    }

    /// `intermediate_bits` (dav1d `get_intermediate_bits`): 8/10-bit → 4, 12-bit → 2.
    #[must_use]
    pub const fn intermediate_bits(self) -> u32 {
        if matches!(self, Self::B12) { 2 } else { 4 }
    }

    /// En büyük piksel değeri (`bitdepth_max`): 255 / 1023 / 4095.
    #[must_use]
    pub const fn max_pixel(self) -> i32 {
        (1i32 << self.depth()) - 1
    }

    /// Çerçeve tipine (`BitDepth`) dönüşüm — S2b.4 aile yuvaları bu tipi
    /// kullanır (ADR-0003: açık bit derinliği).
    #[must_use]
    pub const fn to_bit_depth(self) -> crate::frame::BitDepth {
        match self {
            Self::B8 => crate::frame::BitDepth::Eight,
            Self::B10 => crate::frame::BitDepth::Ten,
            Self::B12 => crate::frame::BitDepth::Twelve,
        }
    }
}

/// MC çekirdeği işlev imzası (dav1d `decl_mc_fn` portu).
///
/// Parametreler: `(dst, dst_stride, src, src_stride, w, h, mx, my, bit_depth)`.
/// Stride'lar **bayt** cinsindendir; `src` dolgu hücresinden (satır −3, sütun
/// −3) başlar, `dst` blok sol üstünden. Ayrıntı modül belgesindedir.
pub type McFn = fn(&mut [u8], usize, &[u8], usize, usize, usize, u8, u8, McBitDepth);

/// `w` ve `mx` için yatay filtre; `mx == 0` ise `None` (kopya).
///
/// dav1d `GET_H_FILTER` (`src/mc_tmpl.c:115-118`): `w > 4` → 8-tap tablo 0,
/// aksi hâlde 4-tap tablo (`3 + (filter_type & 1)`, `filter_type = 0` → 3).
#[must_use]
pub fn horizontal_filter(w: usize, mx: u8) -> Option<&'static [i8; MC_TAPS]> {
    if mx == 0 {
        return None;
    }
    debug_assert!(usize::from(mx) < usize::from(MC_SUBPEL_POSITIONS));
    let table = if w > 4 {
        &MC_8TAP_REGULAR
    } else {
        &MC_4TAP_REGULAR
    };
    Some(&table[usize::from(mx) - 1])
}

/// `h` ve `my` için dikey filtre; `my == 0` ise `None` (kopya).
///
/// dav1d `GET_V_FILTER` (`src/mc_tmpl.c:120-123`): `h > 4` → 8-tap tablo 0,
/// aksi hâlde 4-tap tablo (`3 + ((filter_type >> 2) & 1)`, `filter_type = 0` → 3).
#[must_use]
pub fn vertical_filter(h: usize, my: u8) -> Option<&'static [i8; MC_TAPS]> {
    if my == 0 {
        return None;
    }
    debug_assert!(usize::from(my) < usize::from(MC_SUBPEL_POSITIONS));
    let table = if h > 4 {
        &MC_8TAP_REGULAR
    } else {
        &MC_4TAP_REGULAR
    };
    Some(&table[usize::from(my) - 1])
}

/// Çağrının bellek sözleşmesine uyup uymadığı (AVX2 yolu için ön koşul).
///
/// Bu denetim **her çağrıda** güvenli sarmalayıcı tarafından yapılır (S15.4
/// deseni); başarısızsa ISA gövdesi **hiç** çağrılmaz.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn is_valid_call(
    dst: &[u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: McBitDepth,
) -> bool {
    if !(1..=MC_MAX_W).contains(&w) || !(1..=MC_MAX_H).contains(&h) {
        return false;
    }
    if usize::from(mx) >= usize::from(MC_SUBPEL_POSITIONS)
        || usize::from(my) >= usize::from(MC_SUBPEL_POSITIONS)
    {
        return false;
    }
    let bpp = bit_depth.bytes_per_pixel();
    let src_row = (w + 7) * bpp;
    let dst_row = w * bpp;
    src_stride >= src_row
        && dst_stride >= dst_row
        && src.len() >= (h + 6) * src_stride + src_row
        && dst.len() >= (h - 1) * dst_stride + dst_row
}

// ---------------------------------------------------------------------------
// P32 — kopyasız SIMD sözleşmesi (kuyruk dolgusu `K`)
// ---------------------------------------------------------------------------

/// Kuyruk dolgusu `K` (bayt): SIMD `put`/`prep` çekirdekleri (P32, 8 bpc
/// AVX2) kaynak dilimin `(h+7)` satır × `(w+7)` piksellik okuma bölgesinin
/// sonundan (son satırın `w+6`. sütunundan) sonra **en çok** `K` bayt okuyabilir.
///
/// Çekirdek çağrı başına yerel dolgu kopyası yapmaz; kaynağa doğrudan okur.
/// Satır içi 16 baytlık yüklemeler bir satırın `w+7` piksellik bölgesini
/// aşabilir; aşan baytlar dilimin içindedir (sonraki satır ya da satır
/// adımı boşluğu) ve çıktıya karışmaz. Yalnız son satırda bu baytlar bölgenin
/// ötesine düşer; bu yüzden dilim bölge sonu + `K` bayt uzunluğunda olmalıdır
/// ([`simd_sozlesmesi`]). Çağıran dolguyu sağlar: referans düzlem tahsisi ve
/// `emu_edge` tamponu kuyruğa en az `K` bayt ekler (ADR-0003 F4-MC a notu).
///
/// Ölçülen en büyük aşım 1 bayttır (8 bpc `w = 8` yatay geçişi: 15 baytlık
/// satır bölgesine 16 baytlık yükleme); `K` gelecek gövdeler için pay bırakır.
pub const MC_KUYRUK: usize = 16;

static SKALER_DEVIR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// SIMD `put`/`prep` girişinin sözleşme dışı girdide skaler gövdeye devrettiği
/// çağrı sayısı (süreç başından beri). Çözme hattında 0 olmalıdır (P32).
#[must_use]
pub fn skaler_devir_sayisi() -> u64 {
    SKALER_DEVIR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Skaler devri sayar (yalnız SIMD girişlerinin soğuk yolu).
#[cold]
pub fn skaler_devri_say() {
    SKALER_DEVIR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// SIMD `put`/`prep` girişinin O(1) sözleşme denetimi (8 bpc, P32).
///
/// `w` ∈ {2, 4, …, 128} (2'nin kuvveti), `h` çift ve `2..=128`, `mx, my < 16`;
/// kaynak satır adımı `>= w+7`, kaynak dilimi **bölge sonu + [`MC_KUYRUK`]**
/// bayt; hedef (`put`) `h` satır × `w` bayt ya da `tmp` (`prep`) `w·h` öğe.
/// `dst_gerekli` hedef diliminin gereken en küçük uzunluğudur (çağıran
/// `put`/`prep`'e göre hesaplar) ve `dst_len` gerçek uzunluk.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn simd_sozlesmesi(
    src_len: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    dst_len: usize,
    dst_gerekli: usize,
) -> bool {
    w.is_power_of_two()
        && (2..=MC_MAX_W).contains(&w)
        && h.is_multiple_of(2)
        && (2..=MC_MAX_H).contains(&h)
        && mx < MC_SUBPEL_POSITIONS
        && my < MC_SUBPEL_POSITIONS
        && src_stride >= w + 7
        && src_len >= (h + 6) * src_stride + w + 7 + MC_KUYRUK
        && dst_len >= dst_gerekli
}

/// 16 bpc SIMD gövdelerinin bit derinliği ön koşulu (P34): 10 ya da 12 bit ve
/// `bitdepth_max` onunla tutarlı (1023 / 4095). İki bit derinliği aynı gövdeyi
/// paylaşır; ara kaydırma `bitdepth_max`'tan türetilir.
#[must_use]
pub fn bd16_tutarli(bit_depth: crate::frame::BitDepth, bitdepth_max: u16) -> bool {
    matches!(
        (bit_depth, bitdepth_max),
        (crate::frame::BitDepth::Ten, 1023) | (crate::frame::BitDepth::Twelve, 4095)
    )
}

/// SIMD 16 bpc `put`/`prep` girişinin O(1) sözleşme denetimi (10/12 bit, P34).
///
/// [`simd_sozlesmesi`]'nin 16 bpc karşılığı: pikseller `u16` (2 bayt), satır
/// adımları **bayt**. `w` ∈ {2, 4, …, 128}, `h` çift `2..=128`, `mx, my < 16`;
/// kaynak satır adımı `>= (w+7)·2`, kaynak dilimi **bölge sonu +
/// [`MC_KUYRUK`]** bayt (`(h+6)·ss + (w+7)·2 + K`); bit derinliği 10 ya da 12 ve
/// `bitdepth_max` onunla tutarlı (1023 / 4095). Hedef uzunluğu çağıranda
/// (`dst_gerekli`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn simd16_sozlesmesi(
    src_len: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
    dst_len: usize,
    dst_gerekli: usize,
) -> bool {
    bd16_tutarli(bit_depth, bitdepth_max)
        && w.is_power_of_two()
        && (2..=MC_MAX_W).contains(&w)
        && h.is_multiple_of(2)
        && (2..=MC_MAX_H).contains(&h)
        && mx < MC_SUBPEL_POSITIONS
        && my < MC_SUBPEL_POSITIONS
        && src_stride >= (w + 7) * 2
        && src_len >= (h + 6) * src_stride + (w + 7) * 2 + MC_KUYRUK
        && dst_len >= dst_gerekli
}

/// SIMD bileşik (`avg`/`w_avg`/`mask`/`w_mask`) girişinin O(1) sözleşme
/// denetimi (8 bpc, P33).
///
/// `w`, `h` 2'nin kuvveti ve `en_kucuk..=128` (`w_mask` 8, öbürleri 4);
/// `tmp1`, `tmp2` en az `w·h` öğe (satırlar bitişik), `dst_stride ≥ w`,
/// `dst` en az `(h−1)·dst_stride + w` bayt. İşleve özgü ek denetimler
/// (`mask` uzunluğu, `w_avg` ağırlığı, `w_mask` alt türü) girişte yapılır.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn bilesik_sozlesmesi(
    w: usize,
    h: usize,
    en_kucuk: usize,
    tmp1_len: usize,
    tmp2_len: usize,
    n: usize,
    dst_len: usize,
    dst_stride: usize,
) -> bool {
    w.is_power_of_two()
        && (en_kucuk..=MC_MAX_W).contains(&w)
        && h.is_power_of_two()
        && (en_kucuk..=MC_MAX_H).contains(&h)
        && n == w * h
        && tmp1_len >= n
        && tmp2_len >= n
        && dst_stride >= w
        && dst_len >= (h - 1) * dst_stride + w
}

/// SIMD 16 bpc bileşik (`avg`/`w_avg`/`mask`/`w_mask`) girişinin O(1)
/// sözleşme denetimi (10/12 bit, P34).
///
/// [`bilesik_sozlesmesi`]'nin 16 bpc karşılığı: [`bd16_tutarli`]; `w`, `h`
/// 2'nin kuvveti ve `en_kucuk..=128`; `tmp1`, `tmp2` en az `w·h` öğe; hedef
/// satır adımı `>= 2w` **bayt**, `dst` en az `(h−1)·dst_stride + 2w` bayt.
/// İşleve özgü ek denetimler girişte yapılır.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn bilesik16_sozlesmesi(
    w: usize,
    h: usize,
    en_kucuk: usize,
    tmp1_len: usize,
    tmp2_len: usize,
    dst_len: usize,
    dst_stride: usize,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
) -> bool {
    bd16_tutarli(bit_depth, bitdepth_max)
        && w.is_power_of_two()
        && (en_kucuk..=MC_MAX_W).contains(&w)
        && h.is_power_of_two()
        && (en_kucuk..=MC_MAX_H).contains(&h)
        && tmp1_len >= w * h
        && tmp2_len >= w * h
        && dst_stride >= 2 * w
        && dst_len >= (h - 1) * dst_stride + 2 * w
}

/// SIMD karışım (`blend`/`blend_v`/`blend_h`) girişinin O(1) sözleşme
/// denetimi (8 bpc, P33).
///
/// `w`, `h` 2'nin kuvveti ve verilen aralıklarda (`w_aralik`, `h_aralik`);
/// `tmp` en az `w·h` öğe (satırlar bitişik), `dst_stride ≥ w`, `dst` yazılan
/// son satırın sonuna kadar (`yazilan_satir` satır × `w` bayt; `blend_h`
/// yalnız ilk `(h·3) >> 2` satıra yazar).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn karisim_sozlesmesi(
    w: usize,
    w_aralik: (usize, usize),
    h: usize,
    h_aralik: (usize, usize),
    yazilan_satir: usize,
    tmp_len: usize,
    dst_len: usize,
    dst_stride: usize,
) -> bool {
    w.is_power_of_two()
        && (w_aralik.0..=w_aralik.1).contains(&w)
        && h.is_power_of_two()
        && (h_aralik.0..=h_aralik.1).contains(&h)
        && yazilan_satir >= 1
        && tmp_len >= w * h
        && dst_stride >= w
        && dst_len >= (yazilan_satir - 1) * dst_stride + w
}

/// SIMD 16 bpc karışım (`blend`/`blend_v`/`blend_h`) girişinin O(1)
/// sözleşme denetimi (10/12 bit, P34).
///
/// [`karisim_sozlesmesi`]'nin 16 bpc karşılığı: [`bd16_tutarli`]; `w`, `h`
/// 2'nin kuvveti ve verilen aralıklarda; `tmp` en az `w·h` piksel (`2·w·h`
/// bayt, satırlar bitişik); hedef satır adımı `>= 2w` **bayt**, `dst` yazılan
/// son satırın sonuna kadar (`yazilan_satir` satır × `2w` bayt).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn karisim16_sozlesmesi(
    w: usize,
    w_aralik: (usize, usize),
    h: usize,
    h_aralik: (usize, usize),
    yazilan_satir: usize,
    tmp_len: usize,
    dst_len: usize,
    dst_stride: usize,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
) -> bool {
    bd16_tutarli(bit_depth, bitdepth_max)
        && w.is_power_of_two()
        && (w_aralik.0..=w_aralik.1).contains(&w)
        && h.is_power_of_two()
        && (h_aralik.0..=h_aralik.1).contains(&h)
        && yazilan_satir >= 1
        && tmp_len >= 2 * w * h
        && dst_stride >= 2 * w
        && dst_len >= (yazilan_satir - 1) * dst_stride + 2 * w
}

/// SIMD `warp8x8`/`warp8x8t` girişinin O(1) sözleşme denetimi (8 bpc, P33).
///
/// Kaynak dolgu hücresinden `15 × 15` piksellik okuma bölgesi: `src_stride ≥
/// 15`, kaynak dilimi `14·ss + 15 + MC_KUYRUK` bayt; hedef 8 satır × 8 öğe
/// (`dst_stride ≥ 8`, `dst` en az `7·ds + 8` öğe); filtre indeksi
/// `64 + ((t + 512) >> 10)` yatayda `t = mx + r·b + x·a` (`r` 0…14), dikeyde
/// `t = my + y·d + x·c` (`y` 0…7), `x` 0…7 için `[0, 192]`. `t` her
/// değişkende doğrusal olduğundan dört köşe yeter.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn warp_sozlesmesi(
    src_len: usize,
    src_stride: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    dst_len: usize,
    dst_stride: usize,
) -> bool {
    let icinde = |t: i64| (0..=192).contains(&(64 + ((t + 512) >> 10)));
    let [a, b, c, d] = abcd.map(i64::from);
    let koseler = |t0: i64, satir: i64, sutun: i64, son_satir: i64| {
        [0, son_satir]
            .iter()
            .all(|&r| [0, 7].iter().all(|&x| icinde(t0 + r * satir + x * sutun)))
    };
    src_stride >= 15
        && src_len >= 14 * src_stride + 15 + MC_KUYRUK
        && dst_stride >= 8
        && dst_len >= 7 * dst_stride + 8
        && koseler(i64::from(mx), b, a, 14)
        && koseler(i64::from(my), d, c, 7)
}

/// Warp filtre indeksi `64 + ((t + 512) >> 10)` yatayda (`t = mx + r·b + x·a`,
/// `r` 0…14) ve dikeyde (`t = my + y·d + x·c`, `y` 0…7), `x` 0…7 için
/// `[0, 192]` içinde mi; `t` doğrusal olduğundan dört köşe yeter. P34 16 bpc
/// sözleşmesinin denetimi; 8 bpc [`warp_sozlesmesi`] aynı hesabı kendi
/// gövdesinde yapar (ortak çağrı 8 bpc girişinin satır içi alınmasını değiştirirdi).
#[must_use]
pub fn warp_indeksleri_icinde(abcd: &[i16; 4], mx: i32, my: i32) -> bool {
    let icinde = |t: i64| (0..=192).contains(&(64 + ((t + 512) >> 10)));
    let [a, b, c, d] = abcd.map(i64::from);
    let koseler = |t0: i64, satir: i64, sutun: i64, son_satir: i64| {
        [0, son_satir]
            .iter()
            .all(|&r| [0, 7].iter().all(|&x| icinde(t0 + r * satir + x * sutun)))
    };
    koseler(i64::from(mx), b, a, 14) && koseler(i64::from(my), d, c, 7)
}

/// SIMD 16 bpc `warp8x8`/`warp8x8t` girişinin O(1) sözleşme denetimi (10/12
/// bit, P34).
///
/// [`bd16_tutarli`]; kaynak dolgu hücresinden `15 × 15` piksellik okuma
/// bölgesi: `src_stride ≥ 30` bayt, kaynak dilimi `14·ss + 30 + MC_KUYRUK`
/// bayt; hedef 8 satır × 8 piksel — `warp8x8` (`oge_bayt = 2`) satır adımı
/// `≥ 16` bayt ve `dst` en az `7·ds + 16` bayt, `warp8x8t` (`oge_bayt = 1`,
/// `i16` öğe) satır adımı `≥ 8` öğe ve en az `7·ds + 8` öğe; filtre indeksi
/// [`warp_indeksleri_icinde`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn warp16_sozlesmesi(
    src_len: usize,
    src_stride: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    dst_len: usize,
    dst_stride: usize,
    oge_bayt: usize,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
) -> bool {
    bd16_tutarli(bit_depth, bitdepth_max)
        && src_stride >= 30
        && src_len >= 14 * src_stride + 30 + MC_KUYRUK
        && dst_stride >= 8 * oge_bayt
        && dst_len >= 7 * dst_stride + 8 * oge_bayt
        && warp_indeksleri_icinde(abcd, mx, my)
}

/// SIMD `emu_edge` girişinin O(1) sözleşme denetimi (8 bpc, P33).
///
/// Boyutlar 1…65536, `dst_stride ≥ bw`, hedef en az `(bh−1)·ds + bw` bayt;
/// kenar uzantıları blok içinde (`sol + sağ < bw`, `üst + alt < bh`, skaler
/// gövdenin varsayımı) ve kırpılmış okuma penceresi (`iclip(y, 0, ih−1)`
/// satırından `merkez_h` satır × `iclip(x, 0, iw−1)` sütunundan `merkez_w`
/// bayt) kaynak diliminin içinde.
#[must_use]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn kenar_sozlesmesi(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst_len: usize,
    dst_stride: usize,
    src_len: usize,
    src_stride: usize,
) -> bool {
    let sinir = 1..=65_536usize;
    if !(sinir.contains(&bw) && sinir.contains(&bh) && sinir.contains(&iw) && sinir.contains(&ih)) {
        return false;
    }
    let [bw, bh, iw, ih] = [bw, bh, iw, ih].map(|v| v as i64);
    let (x, y) = (i64::from(x), i64::from(y));
    let sol = (-x).clamp(0, bw - 1);
    let sag = (x + bw - iw).clamp(0, bw - 1);
    let ust = (-y).clamp(0, bh - 1);
    let alt = (y + bh - ih).clamp(0, bh - 1);
    let (mw, mh) = (bw - sol - sag, bh - ust - alt);
    let (cx, cy) = (x.clamp(0, iw - 1), y.clamp(0, ih - 1));
    let (ss, ds) = (src_stride as i64, dst_stride as i64);
    mw >= 1
        && mh >= 1
        && ds >= bw
        && dst_len as i64 >= (bh - 1) * ds + bw
        && (cy + mh - 1) * ss + cx + mw <= src_len as i64
}

/// SIMD 16 bpc `emu_edge` girişinin O(1) sözleşme denetimi (10/12 bit, P34):
/// [`bd16_tutarli`] ve [`kenar_sozlesmesi`]'nin piksel başına 2 baytlık
/// karşılığı (`dst_stride ≥ 2·bw` bayt, hedef `(bh−1)·ds + 2·bw` bayt, okuma
/// penceresi `merkez_w` piksel = `2·merkez_w` bayt).
#[must_use]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn kenar16_sozlesmesi(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst_len: usize,
    dst_stride: usize,
    src_len: usize,
    src_stride: usize,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
) -> bool {
    if !bd16_tutarli(bit_depth, bitdepth_max) {
        return false;
    }
    let bpp = 2i64;
    let sinir = 1..=65_536usize;
    if !(sinir.contains(&bw) && sinir.contains(&bh) && sinir.contains(&iw) && sinir.contains(&ih)) {
        return false;
    }
    let [bw, bh, iw, ih] = [bw, bh, iw, ih].map(|v| v as i64);
    let (x, y) = (i64::from(x), i64::from(y));
    let sol = (-x).clamp(0, bw - 1);
    let sag = (x + bw - iw).clamp(0, bw - 1);
    let ust = (-y).clamp(0, bh - 1);
    let alt = (y + bh - ih).clamp(0, bh - 1);
    let (mw, mh) = (bw - sol - sag, bh - ust - alt);
    let (cx, cy) = (x.clamp(0, iw - 1), y.clamp(0, ih - 1));
    let (ss, ds) = (src_stride as i64, dst_stride as i64);
    mw >= 1
        && mh >= 1
        && ds >= bw * bpp
        && dst_len as i64 >= (bh - 1) * ds + bw * bpp
        && (cy + mh - 1) * ss + (cx + mw) * bpp <= src_len as i64
}

/// SIMD ölçekli `put`/`prep` girişinin O(1) sözleşme denetimi (8 bpc, P33).
///
/// `w` 2'nin kuvveti 2…128, `h` 1…128 (çözme hattı OBMC üst komşusunu
/// `3·2^k` yükseklikle de çağırır: dav1d `(oh4·3 + 3) >> 2` blok satırı → 6, 12,
/// 24); `mx`, `my` 0…1023; `dx`, `dy` 1…2048 (AV1: referans en çok 2× büyük). Okuma bölgesinin son baytı kaynak diliminde:
/// 8-tap dolgu satırı `((my + dy·(h−1)) >> 10) + 7`, sütunu
/// `((mx + dx·(w−1)) >> 10) + 7`; bilinear `+ 4` (satır ve sütun 3'ten başlar).
/// Hedef denetimi girişte (`put` `h` satır × `w`, `prep` `w·h`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn olcekli_sozlesmesi(
    bilin: bool,
    src_len: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
) -> bool {
    if !(w.is_power_of_two()
        && (2..=MC_MAX_W).contains(&w)
        && (1..=MC_MAX_H).contains(&h)
        && (0..1024).contains(&mx)
        && (0..1024).contains(&my)
        && (1..=2048).contains(&dx)
        && (1..=2048).contains(&dy))
    {
        return false;
    }
    let son_y = ((my + dy * (h as i32 - 1)) >> 10) as usize;
    let son_x = ((mx + dx * (w as i32 - 1)) >> 10) as usize;
    let ek = if bilin { 4 } else { 7 };
    src_len > (son_y + ek) * src_stride + son_x + ek
}

/// SIMD 16 bpc ölçekli `put`/`prep` girişinin O(1) sözleşme denetimi (10/12
/// bit, P34): [`bd16_tutarli`] ve [`olcekli_sozlesmesi`]'nin aralıkları; okuma
/// bölgesinin son pikseli (`(son_y + ek)` satırı, `(son_x + ek)` sütunu, 2
/// bayt) kaynak diliminde. Hedef denetimi girişte (`put` `h` satır × `2w`
/// bayt, `prep` `w·h`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn olcekli16_sozlesmesi(
    bilin: bool,
    src_len: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
) -> bool {
    if !(bd16_tutarli(bit_depth, bitdepth_max)
        && w.is_power_of_two()
        && (2..=MC_MAX_W).contains(&w)
        && (1..=MC_MAX_H).contains(&h)
        && (0..1024).contains(&mx)
        && (0..1024).contains(&my)
        && (1..=2048).contains(&dx)
        && (1..=2048).contains(&dy))
    {
        return false;
    }
    let son_y = ((my + dy * (h as i32 - 1)) >> 10) as usize;
    let son_x = ((mx + dx * (w as i32 - 1)) >> 10) as usize;
    let ek = if bilin { 4 } else { 7 };
    src_len >= (son_y + ek) * src_stride + 2 * (son_x + ek) + 2
}

// ---------------------------------------------------------------------------
// S2b.4 — `Dav1dMCDSPContext` aile yüzeyi (resize hariç)
// ---------------------------------------------------------------------------

/// 2B filtre ailesi — dav1d `enum Filter2d` değerleri (`FILTER_2D_*`).
///
/// **Dikkat:** bu numaralandırma dav1d'nin `init_mc_fns` sırasıdır (dispatch
/// yuvası indeksi). dav1d'nin `put_8tap_c` içindeki `filter_type` bundan
/// **farklıdır**: `type_h | (type_v << 2)` olarak kurulur
/// (`src/mc_tmpl.c:360-414`). Eşleme [`Filter2d::filter_type`] içindedir.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Filter2d {
    /// `FILTER_2D_8TAP_REGULAR` — dav1d `8tap_regular` (`filter_type = 0`).
    Regular = 0,
    /// `FILTER_2D_8TAP_REGULAR_SMOOTH` — yatay `regular`, dikey `smooth`.
    RegularSmooth = 1,
    /// `FILTER_2D_8TAP_REGULAR_SHARP` — yatay `regular`, dikey `sharp`.
    RegularSharp = 2,
    /// `FILTER_2D_8TAP_SHARP_REGULAR` — yatay `sharp`, dikey `regular`.
    SharpRegular = 3,
    /// `FILTER_2D_8TAP_SHARP_SMOOTH` — yatay `sharp`, dikey `smooth`.
    SharpSmooth = 4,
    /// `FILTER_2D_8TAP_SHARP`.
    Sharp = 5,
    /// `FILTER_2D_8TAP_SMOOTH_REGULAR` — yatay `smooth`, dikey `regular`.
    SmoothRegular = 6,
    /// `FILTER_2D_8TAP_SMOOTH`.
    Smooth = 7,
    /// `FILTER_2D_8TAP_SMOOTH_SHARP` — yatay `smooth`, dikey `sharp`.
    SmoothSharp = 8,
    /// `FILTER_2D_BILINEAR` — `put_bilin*_c` (8-tap tablosu kullanılmaz).
    Bilinear = 9,
}

impl Filter2d {
    /// Bütün filtre birleşimleri (dav1d `N_2D_FILTERS = 10`).
    pub const ALL: [Self; 10] = [
        Self::Regular,
        Self::RegularSmooth,
        Self::RegularSharp,
        Self::SharpRegular,
        Self::SharpSmooth,
        Self::Sharp,
        Self::SmoothRegular,
        Self::Smooth,
        Self::SmoothSharp,
        Self::Bilinear,
    ];

    /// Yuva indeksi (dav1d `FILTER_2D_*` değeri).
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// İndeksten filtre; 10 dışındaki değerler `None`.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Regular),
            1 => Some(Self::RegularSmooth),
            2 => Some(Self::RegularSharp),
            3 => Some(Self::SharpRegular),
            4 => Some(Self::SharpSmooth),
            5 => Some(Self::Sharp),
            6 => Some(Self::SmoothRegular),
            7 => Some(Self::Smooth),
            8 => Some(Self::SmoothSharp),
            9 => Some(Self::Bilinear),
            _ => None,
        }
    }

    /// dav1d `filter_type = type_h | (type_v << 2)` (`src/mc_tmpl.c:360-414`).
    ///
    /// `type_*`: 0 = `regular`, 1 = `smooth`, 2 = `sharp`.
    #[must_use]
    pub const fn filter_type(self) -> usize {
        match self {
            Self::Regular => 0,       // regular | regular << 2
            Self::RegularSmooth => 4, // regular | smooth  << 2
            Self::RegularSharp => 8,  // regular | sharp   << 2
            Self::SharpRegular => 2,  // sharp   | regular << 2
            Self::SharpSmooth => 6,   // sharp   | smooth  << 2
            Self::Sharp => 10,        // sharp   | sharp   << 2
            Self::SmoothRegular => 1, // smooth  | regular << 2
            Self::Smooth => 5,        // smooth  | smooth  << 2
            Self::SmoothSharp => 9,   // smooth  | sharp   << 2
            Self::Bilinear => 0,      // 8-tap tablosu kullanılmaz
        }
    }

    /// dav1d `GET_H_FILTER`: `filter_type & 3` (`src/mc_tmpl.c:115-118`).
    #[must_use]
    pub const fn horizontal_table(self) -> usize {
        self.filter_type() & 3
    }

    /// dav1d `GET_V_FILTER`: `filter_type >> 2` (`src/mc_tmpl.c:120-123`).
    #[must_use]
    pub const fn vertical_table(self) -> usize {
        self.filter_type() >> 2
    }

    /// Bilinear mı (8-tap tablosu yerine `put_bilin*_c`)?
    #[must_use]
    pub const fn is_bilinear(self) -> bool {
        matches!(self, Self::Bilinear)
    }
}

/// `put` çekirdeği imzası — dav1d `decl_mc_fn` (`src/mc.h:36-40`) + ADR-0003:
/// dilim + bayt stride + açık boyut + açık bit derinliği.
pub type McPutFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// Ölçekli `put` imzası — dav1d `decl_mc_scaled_fn` (`src/mc.h:42-47`).
pub type McPutScaledFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `prep` çekirdeği imzası — dav1d `decl_mct_fn` (`src/mc.h:56-60`).
pub type McPrepFn = fn(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// Ölçekli `prep` imzası — dav1d `decl_mct_scaled_fn` (`src/mc.h:62-67`).
pub type McPrepScaledFn = fn(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `avg` imzası — dav1d `decl_avg_fn` (`src/mc.h:73-78`).
pub type AvgFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `w_avg` imzası — dav1d `decl_w_avg_fn` (`src/mc.h:80-85`).
pub type WAvgFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `mask` imzası — dav1d `decl_mask_fn` (`src/mc.h:87-92`).
pub type MaskFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `w_mask` imzası — dav1d `decl_w_mask_fn` (`src/mc.h:94-100`).
///
/// `subtype` (4:4:4 / 4:2:2 / 4:2:0) **yuva indeksidir** ([`N_W_MASK`]).
pub type WMaskFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `blend` imzası — dav1d `decl_blend_fn` (`src/mc.h:102-106`).
pub type BlendFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `blend_v` / `blend_h` imzası — dav1d `decl_blend_dir_fn` (`src/mc.h:108-111`).
pub type BlendDirFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `warp8x8` imzası — dav1d `decl_warp8x8_fn` (`src/mc.h:49-54`).
pub type Warp8x8Fn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `warp8x8t` imzası — dav1d `decl_warp8x8t_fn` (`src/mc.h:69-74`).
pub type Warp8x8tFn = fn(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `emu_edge` imzası — dav1d `decl_emu_edge_fn` (`src/mc.h:113-118`).
pub type EmuEdgeFn = fn(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    bit_depth: crate::frame::BitDepth,
    bitdepth_max: u16,
);

/// `w_mask` alt tür sayısı (4:4:4, 4:2:2, 4:2:0).
pub const N_W_MASK: usize = 3;

/// MC çekirdek ailesi — dav1d `Dav1dMCDSPContext` (yalnız `resize` hariç).
///
/// Aile, **bit derinliği başına ayrı yuvaya** yerleştirilir
/// ([`crate::dsp::BitDepthDsp::mc`]) ve her yuva 10 `mc`/`mct` filtresi, 3 `w_mask`
/// alt türü ve skaler ölçekli karşılıkları içerir (ADR-0003 §2.1 "alt tür /
/// filtre birleşimi yuvada").
#[derive(Clone, Copy)]
pub struct McDsp {
    /// `mc[10]` — `put` (filtre birleşimi yuva indeksidir).
    pub mc: [McPutFn; 10],
    /// `mc_scaled[10]`.
    pub mc_scaled: [McPutScaledFn; 10],
    /// `mct[10]` — `prep` (ara tampon, 16bpc'de `PREP_BIAS`).
    pub mct: [McPrepFn; 10],
    /// `mct_scaled[10]`.
    pub mct_scaled: [McPrepScaledFn; 10],
    /// `avg`.
    pub avg: AvgFn,
    /// `w_avg`.
    pub w_avg: WAvgFn,
    /// `mask`.
    pub mask: MaskFn,
    /// `w_mask[3]` — 4:4:4, 4:2:2, 4:2:0.
    pub w_mask: [WMaskFn; N_W_MASK],
    /// `blend` (OBMC).
    pub blend: BlendFn,
    /// `blend_v` (OBMC).
    pub blend_v: BlendDirFn,
    /// `blend_h` (OBMC).
    pub blend_h: BlendDirFn,
    /// `warp8x8`.
    pub warp8x8: Warp8x8Fn,
    /// `warp8x8t`.
    pub warp8x8t: Warp8x8tFn,
    /// `emu_edge`.
    pub emu_edge: EmuEdgeFn,
}

impl McDsp {
    /// `BPC` bit derinliğinin skaler tablosu (F3c: her giriş o bit derinliğinin
    /// monomorfudur).
    ///
    /// `resize` ayrı yuvadır ([`crate::dsp::resize`], dav1d `mc.resize`).
    #[must_use]
    pub const fn scalar<const BPC: u32>() -> Self {
        Self {
            mc: [
                mac_wrap::put_0::<BPC>,
                mac_wrap::put_1::<BPC>,
                mac_wrap::put_2::<BPC>,
                mac_wrap::put_3::<BPC>,
                mac_wrap::put_4::<BPC>,
                mac_wrap::put_5::<BPC>,
                mac_wrap::put_6::<BPC>,
                mac_wrap::put_7::<BPC>,
                mac_wrap::put_8::<BPC>,
                mac_wrap::put_bilin::<BPC>,
            ],
            mc_scaled: [
                mac_wrap::put_scaled_0::<BPC>,
                mac_wrap::put_scaled_1::<BPC>,
                mac_wrap::put_scaled_2::<BPC>,
                mac_wrap::put_scaled_3::<BPC>,
                mac_wrap::put_scaled_4::<BPC>,
                mac_wrap::put_scaled_5::<BPC>,
                mac_wrap::put_scaled_6::<BPC>,
                mac_wrap::put_scaled_7::<BPC>,
                mac_wrap::put_scaled_8::<BPC>,
                mac_wrap::put_scaled_bilin::<BPC>,
            ],
            mct: [
                mac_wrap::prep_0::<BPC>,
                mac_wrap::prep_1::<BPC>,
                mac_wrap::prep_2::<BPC>,
                mac_wrap::prep_3::<BPC>,
                mac_wrap::prep_4::<BPC>,
                mac_wrap::prep_5::<BPC>,
                mac_wrap::prep_6::<BPC>,
                mac_wrap::prep_7::<BPC>,
                mac_wrap::prep_8::<BPC>,
                mac_wrap::prep_bilin::<BPC>,
            ],
            mct_scaled: [
                mac_wrap::prep_scaled_0::<BPC>,
                mac_wrap::prep_scaled_1::<BPC>,
                mac_wrap::prep_scaled_2::<BPC>,
                mac_wrap::prep_scaled_3::<BPC>,
                mac_wrap::prep_scaled_4::<BPC>,
                mac_wrap::prep_scaled_5::<BPC>,
                mac_wrap::prep_scaled_6::<BPC>,
                mac_wrap::prep_scaled_7::<BPC>,
                mac_wrap::prep_scaled_8::<BPC>,
                mac_wrap::prep_scaled_bilin::<BPC>,
            ],
            avg: scalar_mc::avg::<BPC>,
            w_avg: scalar_mc::w_avg::<BPC>,
            mask: scalar_mc::mask::<BPC>,
            w_mask: [
                mac_wrap::w_mask_444::<BPC>,
                mac_wrap::w_mask_422::<BPC>,
                mac_wrap::w_mask_420::<BPC>,
            ],
            blend: scalar_mc::blend::<BPC>,
            blend_v: scalar_mc::blend_v::<BPC>,
            blend_h: scalar_mc::blend_h::<BPC>,
            warp8x8: scalar_mc::warp8x8::<BPC>,
            warp8x8t: scalar_mc::warp8x8t::<BPC>,
            emu_edge: scalar_mc::emu_edge::<BPC>,
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

impl core::fmt::Debug for McDsp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("McDsp")
            .field("mc", &format_args!("[{} fn]", self.mc.len()))
            .field("mc_scaled", &format_args!("[{} fn]", self.mc_scaled.len()))
            .field("mct", &format_args!("[{} fn]", self.mct.len()))
            .field(
                "mct_scaled",
                &format_args!("[{} fn]", self.mct_scaled.len()),
            )
            .field("w_mask", &format_args!("[{} fn]", self.w_mask.len()))
            .field("warp8x8", &format_args!("fn"))
            .field("emu_edge", &format_args!("fn"))
            .finish()
    }
}

/// Filtre birleşimine göre ince sarmalayıcılar — dav1d `filter_fns` makrosu
/// (`src/mc_tmpl.c:360-414`).
///
/// dav1d'de her filtre için ayrı bir C işlevi üretilir; burada her biri
/// [`scalar_mc::put_8tap`] gibi genel gövdeyi sabit bir [`Filter2d`] ile
/// çağıran ince sarmalayıcılardır (işlev işaretçisi tablosu dav1d ile aynı
/// şekle sahip olur).
mod mac_wrap {
    use super::{
        AvgFn, BlendDirFn, BlendFn, EmuEdgeFn, Filter2d, MaskFn, McPrepFn, McPrepScaledFn, McPutFn,
        McPutScaledFn, WAvgFn, WMaskFn, Warp8x8Fn, Warp8x8tFn,
    };

    macro_rules! put_wrap {
        ($name:ident, $filter:expr) => {
            #[allow(clippy::too_many_arguments)]
            pub(super) fn $name<const BPC: u32>(
                dst: &mut [u8],
                dst_stride: usize,
                src: &[u8],
                src_stride: usize,
                w: usize,
                h: usize,
                mx: u8,
                my: u8,
                bit_depth: crate::frame::BitDepth,
                bitdepth_max: u16,
            ) {
                super::scalar_mc::put_8tap::<BPC>(
                    dst,
                    dst_stride,
                    src,
                    src_stride,
                    w,
                    h,
                    mx,
                    my,
                    $filter,
                    bit_depth,
                    bitdepth_max,
                );
            }
        };
    }

    macro_rules! prep_wrap {
        ($name:ident, $filter:expr) => {
            #[allow(clippy::too_many_arguments)]
            pub(super) fn $name<const BPC: u32>(
                tmp: &mut [i16],
                src: &[u8],
                src_stride: usize,
                w: usize,
                h: usize,
                mx: u8,
                my: u8,
                bit_depth: crate::frame::BitDepth,
                bitdepth_max: u16,
            ) {
                super::scalar_mc::prep_8tap::<BPC>(
                    tmp,
                    src,
                    src_stride,
                    w,
                    h,
                    mx,
                    my,
                    $filter,
                    bit_depth,
                    bitdepth_max,
                );
            }
        };
    }

    macro_rules! put_scaled_wrap {
        ($name:ident, $filter:expr) => {
            #[allow(clippy::too_many_arguments)]
            pub(super) fn $name<const BPC: u32>(
                dst: &mut [u8],
                dst_stride: usize,
                src: &[u8],
                src_stride: usize,
                w: usize,
                h: usize,
                mx: i32,
                my: i32,
                dx: i32,
                dy: i32,
                bit_depth: crate::frame::BitDepth,
                bitdepth_max: u16,
            ) {
                super::scalar_mc::put_8tap_scaled::<BPC>(
                    dst,
                    dst_stride,
                    src,
                    src_stride,
                    w,
                    h,
                    mx,
                    my,
                    dx,
                    dy,
                    $filter,
                    bit_depth,
                    bitdepth_max,
                );
            }
        };
    }

    macro_rules! prep_scaled_wrap {
        ($name:ident, $filter:expr) => {
            #[allow(clippy::too_many_arguments)]
            pub(super) fn $name<const BPC: u32>(
                tmp: &mut [i16],
                src: &[u8],
                src_stride: usize,
                w: usize,
                h: usize,
                mx: i32,
                my: i32,
                dx: i32,
                dy: i32,
                bit_depth: crate::frame::BitDepth,
                bitdepth_max: u16,
            ) {
                super::scalar_mc::prep_8tap_scaled::<BPC>(
                    tmp,
                    src,
                    src_stride,
                    w,
                    h,
                    mx,
                    my,
                    dx,
                    dy,
                    $filter,
                    bit_depth,
                    bitdepth_max,
                );
            }
        };
    }

    put_wrap!(put_0, Filter2d::Regular);
    put_wrap!(put_1, Filter2d::RegularSmooth);
    put_wrap!(put_2, Filter2d::RegularSharp);
    put_wrap!(put_3, Filter2d::SharpRegular);
    put_wrap!(put_4, Filter2d::SharpSmooth);
    put_wrap!(put_5, Filter2d::Sharp);
    put_wrap!(put_6, Filter2d::SmoothRegular);
    put_wrap!(put_7, Filter2d::Smooth);
    put_wrap!(put_8, Filter2d::SmoothSharp);
    prep_wrap!(prep_0, Filter2d::Regular);
    prep_wrap!(prep_1, Filter2d::RegularSmooth);
    prep_wrap!(prep_2, Filter2d::RegularSharp);
    prep_wrap!(prep_3, Filter2d::SharpRegular);
    prep_wrap!(prep_4, Filter2d::SharpSmooth);
    prep_wrap!(prep_5, Filter2d::Sharp);
    prep_wrap!(prep_6, Filter2d::SmoothRegular);
    prep_wrap!(prep_7, Filter2d::Smooth);
    prep_wrap!(prep_8, Filter2d::SmoothSharp);
    put_scaled_wrap!(put_scaled_0, Filter2d::Regular);
    put_scaled_wrap!(put_scaled_1, Filter2d::RegularSmooth);
    put_scaled_wrap!(put_scaled_2, Filter2d::RegularSharp);
    put_scaled_wrap!(put_scaled_3, Filter2d::SharpRegular);
    put_scaled_wrap!(put_scaled_4, Filter2d::SharpSmooth);
    put_scaled_wrap!(put_scaled_5, Filter2d::Sharp);
    put_scaled_wrap!(put_scaled_6, Filter2d::SmoothRegular);
    put_scaled_wrap!(put_scaled_7, Filter2d::Smooth);
    put_scaled_wrap!(put_scaled_8, Filter2d::SmoothSharp);
    prep_scaled_wrap!(prep_scaled_0, Filter2d::Regular);
    prep_scaled_wrap!(prep_scaled_1, Filter2d::RegularSmooth);
    prep_scaled_wrap!(prep_scaled_2, Filter2d::RegularSharp);
    prep_scaled_wrap!(prep_scaled_3, Filter2d::SharpRegular);
    prep_scaled_wrap!(prep_scaled_4, Filter2d::SharpSmooth);
    prep_scaled_wrap!(prep_scaled_5, Filter2d::Sharp);
    prep_scaled_wrap!(prep_scaled_6, Filter2d::SmoothRegular);
    prep_scaled_wrap!(prep_scaled_7, Filter2d::Smooth);
    prep_scaled_wrap!(prep_scaled_8, Filter2d::SmoothSharp);

    /// `mc[FILTER_2D_BILINEAR]` — dav1d `put_bilin_c`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn put_bilin<const BPC: u32>(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: u8,
        my: u8,
        bit_depth: crate::frame::BitDepth,
        bitdepth_max: u16,
    ) {
        super::scalar_mc::put_bilin::<BPC>(
            dst,
            dst_stride,
            src,
            src_stride,
            w,
            h,
            mx,
            my,
            bit_depth,
            bitdepth_max,
        );
    }

    /// `mc_scaled[FILTER_2D_BILINEAR]` — dav1d `put_bilin_scaled_c`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn put_scaled_bilin<const BPC: u32>(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        dx: i32,
        dy: i32,
        bit_depth: crate::frame::BitDepth,
        bitdepth_max: u16,
    ) {
        super::scalar_mc::put_bilin_scaled::<BPC>(
            dst,
            dst_stride,
            src,
            src_stride,
            w,
            h,
            mx,
            my,
            dx,
            dy,
            bit_depth,
            bitdepth_max,
        );
    }

    /// `mct[FILTER_2D_BILINEAR]` — dav1d `prep_bilin_c`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prep_bilin<const BPC: u32>(
        tmp: &mut [i16],
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: u8,
        my: u8,
        bit_depth: crate::frame::BitDepth,
        bitdepth_max: u16,
    ) {
        super::scalar_mc::prep_bilin::<BPC>(
            tmp,
            src,
            src_stride,
            w,
            h,
            mx,
            my,
            bit_depth,
            bitdepth_max,
        );
    }

    /// `mct_scaled[FILTER_2D_BILINEAR]` — dav1d `prep_bilin_scaled_c`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prep_scaled_bilin<const BPC: u32>(
        tmp: &mut [i16],
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        dx: i32,
        dy: i32,
        bit_depth: crate::frame::BitDepth,
        bitdepth_max: u16,
    ) {
        super::scalar_mc::prep_bilin_scaled::<BPC>(
            tmp,
            src,
            src_stride,
            w,
            h,
            mx,
            my,
            dx,
            dy,
            bit_depth,
            bitdepth_max,
        );
    }

    /// `w_mask[0]` — 4:4:4.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w_mask_444<const BPC: u32>(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16],
        tmp2: &[i16],
        w: usize,
        h: usize,
        mask: &mut [u8],
        sign: i32,
        ss_hor: bool,
        ss_ver: bool,
        bit_depth: crate::frame::BitDepth,
        bitdepth_max: u16,
    ) {
        debug_assert!(!ss_hor && !ss_ver, "444 yuvasında alt örnekleme yok");
        super::scalar_mc::w_mask::<BPC>(
            dst,
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            ss_hor,
            ss_ver,
            bit_depth,
            bitdepth_max,
        );
    }

    /// `w_mask[1]` — 4:2:2.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w_mask_422<const BPC: u32>(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16],
        tmp2: &[i16],
        w: usize,
        h: usize,
        mask: &mut [u8],
        sign: i32,
        ss_hor: bool,
        ss_ver: bool,
        bit_depth: crate::frame::BitDepth,
        bitdepth_max: u16,
    ) {
        debug_assert!(
            ss_hor && !ss_ver,
            "422 yuvasında yalnız yatay alt örnekleme"
        );
        super::scalar_mc::w_mask::<BPC>(
            dst,
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            ss_hor,
            ss_ver,
            bit_depth,
            bitdepth_max,
        );
    }

    /// `w_mask[2]` — 4:2:0.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w_mask_420<const BPC: u32>(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16],
        tmp2: &[i16],
        w: usize,
        h: usize,
        mask: &mut [u8],
        sign: i32,
        ss_hor: bool,
        ss_ver: bool,
        bit_depth: crate::frame::BitDepth,
        bitdepth_max: u16,
    ) {
        debug_assert!(ss_hor && ss_ver, "420 yuvasında iki eksende alt örnekleme");
        super::scalar_mc::w_mask::<BPC>(
            dst,
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            ss_hor,
            ss_ver,
            bit_depth,
            bitdepth_max,
        );
    }

    /// İşlev işaretçisi tiplerinin derleme zamanı denetimi (imza kayması olmasın).
    const _: [McPutFn; 1] = [put_0::<8>];
    const _: [McPutScaledFn; 1] = [put_scaled_0::<8>];
    const _: [McPrepFn; 1] = [prep_0::<8>];
    const _: [McPrepScaledFn; 1] = [prep_scaled_0::<8>];
    const _: [WMaskFn; 1] = [w_mask_444::<8>];
    const _: [AvgFn; 0] = [];
    const _: [WAvgFn; 0] = [];
    const _: [MaskFn; 0] = [];
    const _: [BlendFn; 0] = [];
    const _: [BlendDirFn; 0] = [];
    const _: [Warp8x8Fn; 0] = [];
    const _: [Warp8x8tFn; 0] = [];
    const _: [EmuEdgeFn; 0] = [];
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{
        MC_4TAP_REGULAR, MC_8TAP_REGULAR, MC_MAX_H, MC_MAX_W, MC_SUBPEL_POSITIONS, MC_TAPS,
        McBitDepth, horizontal_filter, is_valid_call, vertical_filter,
    };

    #[test]
    fn filtre_tablolari_dav1d_ile_ayni() {
        // dav1d `src/tables.c:445-461` (8-tap) ve `497-516` (4-tap).
        assert_eq!(MC_8TAP_REGULAR[0], [0, 1, -3, 63, 4, -1, 0, 0]);
        assert_eq!(MC_8TAP_REGULAR[7], [0, 1, -7, 38, 38, -7, 1, 0]);
        assert_eq!(MC_8TAP_REGULAR[14], [0, 0, -1, 4, 63, -3, 1, 0]);
        assert_eq!(MC_4TAP_REGULAR[0], [0, 0, -2, 63, 4, -1, 0, 0]);
        assert_eq!(MC_4TAP_REGULAR[14], [0, 0, -1, 4, 63, -2, 0, 0]);
        // Her satırın katsayı toplamı 64'tür (dav1d tablosunun değişmezi).
        for row in MC_8TAP_REGULAR.iter().chain(MC_4TAP_REGULAR.iter()) {
            assert_eq!(row.iter().map(|&c| i32::from(c)).sum::<i32>(), 64);
        }
    }

    #[test]
    fn filtre_secimi_genislige_gore() {
        // mx = 0 → filtre yok (kopya).
        assert!(horizontal_filter(8, 0).is_none());
        assert!(vertical_filter(8, 0).is_none());
        // w = 4 ve 2 → 4-tap; w ≥ 8 → 8-tap (`src/mc_tmpl.c:115-118`).
        for w in [1, 2, 3, 4] {
            assert_eq!(horizontal_filter(w, 3).unwrap(), &MC_4TAP_REGULAR[2]);
        }
        for w in [5, 8, 16, 128] {
            assert_eq!(horizontal_filter(w, 3).unwrap(), &MC_8TAP_REGULAR[2]);
        }
        // h = 4 ve altı → 4-tap; h ≥ 5 → 8-tap (`src/mc_tmpl.c:120-123`).
        for h in [1, 2, 4] {
            assert_eq!(vertical_filter(h, 15).unwrap(), &MC_4TAP_REGULAR[14]);
        }
        for h in [5, 8, 128] {
            assert_eq!(vertical_filter(h, 15).unwrap(), &MC_8TAP_REGULAR[14]);
        }
    }

    #[test]
    fn bit_derinligi_sabitleri() {
        assert_eq!(McBitDepth::B8.intermediate_bits(), 4);
        assert_eq!(McBitDepth::B10.intermediate_bits(), 4);
        assert_eq!(McBitDepth::B12.intermediate_bits(), 2);
        assert_eq!(McBitDepth::B8.max_pixel(), 255);
        assert_eq!(McBitDepth::B10.max_pixel(), 1023);
        assert_eq!(McBitDepth::B12.max_pixel(), 4095);
        assert_eq!(McBitDepth::B8.bytes_per_pixel(), 1);
        assert_eq!(McBitDepth::B12.bytes_per_pixel(), 2);
        for depth in [8, 10, 12] {
            assert_eq!(McBitDepth::from_depth(depth).unwrap().depth(), depth);
        }
        assert!(McBitDepth::from_depth(9).is_none());
        assert_eq!(MC_TAPS, 8);
        assert_eq!(MC_SUBPEL_POSITIONS, 16);
        assert_eq!((MC_MAX_W, MC_MAX_H), (128, 128));
    }

    #[test]
    fn bellek_sozlesmesi_sinirlari() {
        let bpp = 1usize;
        let (w, h, stride) = (8usize, 8usize, 32usize);
        let src_len = (h + 6) * stride + (w + 7) * bpp;
        let dst_len = (h - 1) * stride + w * bpp;
        let src = vec![0u8; src_len];
        let dst = vec![0u8; dst_len];
        assert!(is_valid_call(
            &dst,
            stride,
            &src,
            stride,
            w,
            h,
            1,
            1,
            McBitDepth::B8
        ));
        // Bir bayt eksik kaynak reddedilir.
        assert!(!is_valid_call(
            &dst,
            stride,
            &src[..src_len - 1],
            stride,
            w,
            h,
            1,
            1,
            McBitDepth::B8
        ));
        // Küçük stride reddedilir.
        assert!(!is_valid_call(
            &dst,
            stride,
            &src,
            w + 6,
            w,
            h,
            1,
            1,
            McBitDepth::B8
        ));
        // Alt piksel konumu 16 aralık dışıdır.
        assert!(!is_valid_call(
            &dst,
            stride,
            &src,
            stride,
            w,
            h,
            16,
            0,
            McBitDepth::B8
        ));
        // Desteklenmeyen boyut reddedilir.
        assert!(!is_valid_call(
            &dst,
            stride,
            &src,
            stride,
            0,
            h,
            1,
            1,
            McBitDepth::B8
        ));
        assert!(!is_valid_call(
            &dst,
            stride,
            &src,
            stride,
            w,
            129,
            1,
            1,
            McBitDepth::B8
        ));
    }
}
