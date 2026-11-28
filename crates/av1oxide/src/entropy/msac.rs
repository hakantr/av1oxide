//! MSAC — çok sembollü aralık çözücü (S2a.3).
//!
//! Bu modül, dav1d `src/msac.c` + `src/msac.h` dosyalarının **C yolunun**
//! (`asm = OFF`) skaler Rust karşılığıdır. Bit-exact hedef: aynı veri
//! tamponu, aynı CDF dizisi ve aynı `disable_cdf_update` bayrağıyla çözülen
//! her sembol, son `dif`/`rng`/`cnt`/`buf_pos` durumu ve adapte olmuş CDF
//! hücreleri dav1d ile **birebir** aynıdır (bkz. `scripts/oracle/msac_oracle.c`
//! ve `scripts/oracle/msac_diff.py`).
//!
//! ## Kaynak izlenebilirliği (SRC-07)
//!
//! | Öğe | dav1d kaynağı (@ `99f3354`) | AV1 spec |
//! |---|---|---|
//! | sabitler `EC_PROB_SHIFT = 6`, `EC_MIN_PROB = 4` | `src/msac.c:36-37` | 8.2.6 (aralık çözücü) |
//! | pencere `ec_win` = `size_t` (x86_64: 64 bit) | `src/msac.h:36`, `msac.c:39` | — |
//! | `ctx_refill` — veri sonu dolgusu, `cnt` | `src/msac.c:41-58` | 8.2.6 |
//! | `ctx_norm` — yeniden ölçekleme, `d` | `src/msac.c:85-97` | 8.2.6 |
//! | `decode_bool_equi` (`f = 1/2`, çarpma yerine kaydırma) | `src/msac.c:99-112` | 8.3.2 (`read_bit`) |
//! | `decode_bool(f)` | `src/msac.c:117-128` | 8.3.2 |
//! | `decode_symbol_adapt` (2–16 sembol) | `src/msac.c:132-166` | 8.3.2 (`read_symbol`) |
//! | `decode_bool_adapt` | `src/msac.c:168-185` | 8.3.2 |
//! | `decode_hi_tok` | `src/msac.c:187-201` | 8.3.2 (`read_coeff_token` üst token) |
//! | `decode_subexp` | `src/msac.c:60-74` | 8.3.2 (`read_subexp`) |
//! | `decode_bools`, `decode_uniform` | `src/msac.h:94-108` | 8.3.2 (`read_literal`, `read_uniform`) |
//! | `dav1d_msac_init` | `src/msac.c:204-220` | 8.2.6 |
//!
//! `inv_recenter` dav1d `include/common/intops.h:75-82`; `ulog2` aynı dosya
//! 67-69. AV1 spec §8.2.6 "Symbol Decoding Process" ve §8.3.2 "Symbol
//! Decoding Functions" referanstır; dav1d'nin **asm** yolları
//! (`src/x86/msac.asm`, `src/arm/msac.S`) bu modülde **referans alınmaz**
//! (F2 kapsam sınırı: yalnız C yolu).
//!
//! ## Sözleşme tablosu (SMD-01)
//!
//! | İşlem | Taşma/kaydırma davranışı |
//! |---|---|
//! | `dif`, `rng` | işaretsiz; C'deki `unsigned`/`size_t` sarması ile aynı (`wrapping_*`) |
//! | `cnt` | işaretli; `0 <= cnt < d` karşılaştırması C'deki `unsigned` dönüşümüyle yapılır (negatif `cnt` "sonsuz" sayılır ve dolgu yapılmaz) |
//! | `dif << d`, `rng << d` | `d <= 15`; üst bitler bilerek atılır (C ile aynı) |
//! | veri sonu | kalan bayt yuvaları `0xff` ile doldurulur, yani çözülen bayt `0x00` gibi davranır (`msac.c:47-51`) |
//! | CDF adaptasyonu | `u16` hücrelere yazım C'nin `uint16_t` dönüşümüyle aynı (mod 2^16) |
//!
//! ## Ön koşullar (çağıranın sözleşmesi)
//!
//! - `decode_symbol_adapt(cdf, n_symbols)`: `1 <= n_symbols <= 15` ve
//!   `cdf.len() >= n_symbols + 1`; `cdf[n_symbols]` sayaç hücresidir ve
//!   `<= 32` olmalıdır. Bu ön koşullar sağlanmazsa sonuç tanımsızdır
//!   (dav1d'de `assert`; burada `debug_assert`). Bozuk **bit akışı** ise
//!   panik üretmez: veri sonu dolgusu ve `val <= n_symbols` sınırı bunu
//!   garanti eder (bkz. `scripts/oracle/run_msac_robustness.sh`).
//! - `decode_bool_adapt(cdf)`: `cdf.len() >= 2`.
//! - `decode_uniform(n)`: `n >= 2`; `decode_subexp(r, n, k)`: `n >> k == 8`.

#![forbid(unsafe_code)]

#[cfg(feature = "msac-trace")]
use crate::trace;

/// Olasılık bit kaydırması (`msac.c:36`).
const EC_PROB_SHIFT: u32 = 6;
/// En küçük olasılık (`msac.c:37`, `<= (1 << EC_PROB_SHIFT) / 16`).
const EC_MIN_PROB: u32 = 4;
/// `ec_win` bit genişliği (`sizeof(size_t) << 3`; x86_64 = 64).
const EC_WIN_BITS: i32 = 64;

/// `EC_WIN_SIZE - 16` = `dif` içindeki 16 bitlik kod değerinin kaydırması.
const EC_CODE_SHIFT: u32 = (EC_WIN_BITS - 16) as u32;

/// MSAC ISA seçim düzeyi (`P47 / F4-MSAC`, `P54 / F5-MSAC`).
///
/// Yetki belirteçlerini ([`crate::cpu::Sse2Token`], [`crate::cpu::Avx2Token`],
/// [`crate::cpu::NeonToken`]) doğrudan taşır; çalışma zamanında fonksiyon
/// işaretçisi dolaylı çağrısı yapmadan `#[inline(always)]` dallanmayla ilgili
/// SIMD gövdesini seçer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MsacIsa {
    /// Skaler yol (`--cpu-mask none` veya SIMD dışı hedef).
    #[default]
    Scalar,
    /// x86_64 SSE2 (`decode_symbol_adapt4/8/16_sse2`, `decode_hi_tok_sse2`, `decode_bool*_sse2`).
    #[cfg(target_arch = "x86_64")]
    Sse2(crate::cpu::Sse2Token),
    /// x86_64 SSE2 + AVX2 (`decode_symbol_adapt16_avx2` + 6 SSE2 işlevi).
    #[cfg(target_arch = "x86_64")]
    Avx2(crate::cpu::Sse2Token, crate::cpu::Avx2Token),
    /// aarch64 NEON (`decode_symbol_adapt4/8/16_neon`, `decode_hi_tok_neon`, `decode_bool*_neon`).
    #[cfg(target_arch = "aarch64")]
    Neon(crate::cpu::NeonToken),
}

/// Aşırı okumasız hızlı 64-bit kelime `ctx_refill` (`msac.c:41-58`, `msac.asm:163-220`, `arm/64/msac.S:307-359`).
///
/// `pos + 8 <= buf.len()` hızlı yolunda tek `u64::from_be_bytes` yüklemesiyle
/// tam `num_bytes = ((c as u32) >> 3) + 1` (`1..=7`) bayt okur ve `keep_mask`
/// ile yalnızca okunan baytları `dif`'e yerleştirir; böylece `(dif, pos, cnt)`
/// üçlüsü her tekil çağrıda skaler bayt döngüsüyle **%100 bit-birebir aynı**
/// kalır ve tampon sınırı 1 bayt bile aşılmaz.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[cfg_attr(target_arch = "x86_64", inline(always))]
#[cfg_attr(target_arch = "aarch64", inline(never))]
pub(crate) fn ctx_refill_fast(buf: &[u8], pos: &mut usize, dif: &mut u64, cnt: &mut i32) {
    let c: i32 = EC_WIN_BITS - *cnt - 24;
    debug_assert!((0..=55).contains(&c), "ctx_refill kaydırması: {c}");
    let p = *pos;
    if p <= buf.len().saturating_sub(8) {
        let c8: [u8; 8] = buf[p..p + 8].try_into().unwrap();
        let raw = u64::from_be_bytes(c8);
        let cu = c as u32;
        let num_bytes = (cu >> 3) + 1;
        let shift_r = 56 - (cu & !7);
        let shift_l = cu & 7;
        *dif |= ((!raw) >> shift_r) << shift_l;
        *pos = p + num_bytes as usize;
        *cnt += (num_bytes * 8) as i32;
    } else {
        ctx_refill_tail(buf, pos, dif, cnt, c);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[cold]
#[inline(never)]
fn ctx_refill_tail(buf: &[u8], pos: &mut usize, dif: &mut u64, cnt: &mut i32, mut c: i32) {
    loop {
        if *pos >= buf.len() {
            // `dif |= ~(~(ec_win)0xff << c)`: [0, c+8) bitleri 1 olur.
            *dif |= !(u64::MAX << (c as u32 + 8));
            break;
        }
        *dif |= ((buf[*pos] ^ 0xff) as u64) << c;
        *pos += 1;
        c -= 8;
        if c < 0 {
            break;
        }
    }
    *cnt = EC_WIN_BITS - c - 24;
}

/// `ctx_norm` (`msac.c:85-97`): `32768 <= rng < 65536` olacak biçimde yeniden
/// ölçekler; gerekirse [`ctx_refill_fast`] ile dolgu yapar.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn ctx_norm_fast(
    buf: &[u8],
    pos: &mut usize,
    dif: &mut u64,
    rng: &mut u32,
    cnt: &mut i32,
    new_dif: u64,
    new_rng: u32,
) {
    debug_assert!(new_rng <= 65535);
    let d = 15 ^ (31 ^ new_rng.leading_zeros());
    let old_cnt = *cnt;
    *dif = new_dif << d;
    *rng = new_rng << d;
    *cnt = old_cnt - d as i32;
    if (old_cnt as u32) < d {
        ctx_refill_fast(buf, pos, dif, cnt);
    }
}

/// `ctx_norm_fast` doğrudan `&mut MsacContext` üzerinde: `s.buf` yalnız dolgu
/// gerektiğinde (`(old_cnt as u32) < d`) yüklenir.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn ctx_norm_ctx(s: &mut MsacContext<'_>, new_dif: u64, new_rng: u32) {
    debug_assert!(new_rng <= 65535);
    let d = 15 ^ (31 ^ new_rng.leading_zeros());
    let old_cnt = s.cnt;
    s.dif = new_dif << d;
    s.rng = new_rng << d;
    s.cnt = old_cnt - d as i32;
    if (old_cnt as u32) < d {
        ctx_refill_fast(s.buf, &mut s.pos, &mut s.dif, &mut s.cnt);
    }
}

/// MSAC çözücü durumu (dav1d `MsacContext`).
///
/// `dif`/`rng`/`cnt` alanları dav1d ile **birebir** aynı anlamdadır; `pos`
/// `buf_pos` işaretçisinin tampon başına göre uzaklığıdır.
#[derive(Debug, Clone)]
pub struct MsacContext<'a> {
    pub(crate) buf: &'a [u8],
    pub(crate) pos: usize,
    pub(crate) dif: u64,
    pub(crate) rng: u32,
    pub(crate) cnt: i32,
    pub(crate) allow_update_cdf: bool,
    isa: MsacIsa,
}

impl<'a> MsacContext<'a> {
    /// `dav1d_msac_init` (`msac.c:204`): aralık `0x8000`, sayaç `-15`,
    /// ardından ilk dolgu (`MsacIsa::Scalar` varsayılanı).
    pub fn init(buf: &'a [u8], disable_cdf_update: bool) -> Self {
        Self::init_with_isa(buf, disable_cdf_update, MsacIsa::Scalar)
    }

    /// `dav1d_msac_init` + `msac_init_x86` (`msac.h:59-69`): verilen [`MsacIsa`]
    /// yetkisiyle MSAC bağlamını kurar.
    pub fn init_with_isa(buf: &'a [u8], disable_cdf_update: bool, isa: MsacIsa) -> Self {
        let mut s = Self {
            buf,
            pos: 0,
            dif: 0,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: !disable_cdf_update,
            isa,
        };
        s.ctx_refill();
        #[cfg(feature = "msac-trace")]
        trace::tile(buf);
        s
    }

    /// Etkin MSAC ISA düzeyi.
    #[must_use]
    pub const fn isa(&self) -> MsacIsa {
        self.isa
    }

    /// CDF adaptasyonu açık mı (`allow_update_cdf`)?
    pub fn allow_update_cdf(&self) -> bool {
        self.allow_update_cdf
    }

    /// Aralık değeri (`rng`).
    pub fn rng(&self) -> u32 {
        self.rng
    }

    /// Kod değeri (`dif`).
    pub fn dif(&self) -> u64 {
        self.dif
    }

    /// Bit sayacı (`cnt`); negatif olabilir (dav1d ile aynı).
    pub fn cnt(&self) -> i32 {
        self.cnt
    }

    /// Okunan bayt sayısı (`buf_pos - buf`).
    pub fn buf_pos(&self) -> usize {
        self.pos
    }

    /// Karo sonu MSAC izleyen bit (`1` işaret biti + `0` dolgusu) ve kalan sıfır bayt
    /// doğrulaması (`dav1d` `check_trailing_bits_after_symbol_coder`, `src/decode.c:2620-2635`).
    ///
    /// `true` geçerli, `false` bozuk/geçersiz bit dizisini gösterir.
    #[inline]
    #[must_use]
    pub fn check_trailing_bits(&self) -> bool {
        if self.cnt <= -15 {
            return false;
        }
        let n_bits = -(self.cnt + 14);
        debug_assert!(n_bits <= 0);
        let n_bytes = ((n_bits + 7) >> 3) as isize;
        let Some(p_idx) = self.pos.checked_add_signed(n_bytes) else {
            return false;
        };
        if p_idx == 0 || p_idx > self.buf.len() {
            return false;
        }
        let pattern = 128u8 >> (((n_bits - 1) & 7) as u32);
        let mask = pattern.wrapping_mul(2).wrapping_sub(1);
        if (self.buf[p_idx - 1] & mask) != pattern {
            return false;
        }
        self.buf[p_idx..].iter().all(|&b| b == 0)
    }

    /// `ctx_refill` (`msac.c:41-58`).
    ///
    /// Pencereyi 24 bitlik "serbest" alanı dolduracak biçimde besler. Veri
    /// bittiğinde **kalan bütün bitler** 1'e çekilir (`msac.c:47-51`); `dif`
    /// baytları tümleyen olarak tuttuğu için bu, akışın `0x00` baytlarıyla
    /// devam etmesi demektir. Dolgu maskesi `[0, c+8)` bitleridir — yalnız
    /// bayt yuvası değil: `~(~(ec_win)0xff << c)` ifadesi tam olarak bunu verir.
    #[inline(always)]
    fn ctx_refill(&mut self) {
        let mut c: i32 = EC_WIN_BITS - self.cnt - 24;
        debug_assert!((0..=55).contains(&c), "ctx_refill kaydırması: {c}");
        loop {
            if self.pos >= self.buf.len() {
                // `dif |= ~(~(ec_win)0xff << c)`: [0, c+8) bitleri 1 olur.
                self.dif |= !(u64::MAX << (c as u32 + 8));
                break;
            }
            self.dif |= ((self.buf[self.pos] ^ 0xff) as u64) << c;
            self.pos += 1;
            c -= 8;
            if c < 0 {
                break;
            }
        }
        self.cnt = EC_WIN_BITS - c - 24;
    }

    /// `ctx_norm` (`msac.c:85-97`): `32768 <= rng < 65536` olacak biçimde
    /// yeniden ölçekler; gerekirse dolgu yapar.
    #[inline(always)]
    fn ctx_norm(&mut self, dif: u64, rng: u32) {
        debug_assert!(rng <= 65535);
        // `15 ^ (31 ^ clz(rng))` = clz(rng) - 16 (rng < 65536 için).
        let d = 15 ^ (31 ^ rng.leading_zeros());
        let cnt = self.cnt;
        self.dif = dif << d;
        self.rng = rng << d;
        self.cnt = cnt - d as i32;
        // C'deki `(unsigned)cnt < (unsigned)d`: negatif cnt dolgu yapmaz.
        if (cnt as u32) < d {
            self.ctx_refill();
        }
    }

    /// `dav1d_msac_decode_bool_equi` (`msac.c:99`, `msac.asm:406`): `f = 1/2`.
    ///
    /// `f = 16384 >> EC_PROB_SHIFT` olduğu için çarpma yerine kaydırma
    /// kullanılır (`(r >> 8) << 7`). CDF güncellenmez.
    #[inline(always)]
    pub fn decode_bool_equi(&mut self) -> u32 {
        #[cfg(target_arch = "x86_64")]
        let out = match self.isa {
            MsacIsa::Scalar => self.decode_bool_equi_scalar(),
            MsacIsa::Sse2(t) | MsacIsa::Avx2(t, _) => {
                crate::dsp::x86::msac::decode_bool_equi_sse2(t, self)
            }
        };
        #[cfg(target_arch = "aarch64")]
        let out = match self.isa {
            MsacIsa::Scalar => self.decode_bool_equi_scalar(),
            MsacIsa::Neon(t) => crate::dsp::arm::giris::msac_decode_bool_equi(t, self),
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let out = self.decode_bool_equi_scalar();
        #[cfg(feature = "msac-trace")]
        trace::sym("bool_equi", 1, i64::from(out), self.rng, self.cnt, 0, 0, -1);
        out
    }

    #[cfg_attr(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        cold,
        inline(never)
    )]
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        inline(always)
    )]
    pub(crate) fn decode_bool_equi_scalar(&mut self) -> u32 {
        let r = self.rng;
        let mut dif = self.dif;
        debug_assert!((dif >> EC_CODE_SHIFT) < r as u64);
        let mut v = ((r >> 8) << 7) + EC_MIN_PROB;
        let vw = (v as u64) << EC_CODE_SHIFT;
        let ret = u32::from(dif >= vw);
        if ret != 0 {
            dif = dif.wrapping_sub(vw);
        }
        v = v.wrapping_add(ret.wrapping_mul(r.wrapping_sub(2u32.wrapping_mul(v))));
        self.ctx_norm(dif, v);
        1 - ret
    }

    /// `dav1d_msac_decode_bool` (`msac.c:117`, `msac.asm:433`): `f` olasılığıyla tek bit.
    #[inline(always)]
    pub fn decode_bool(&mut self, f: u32) -> u32 {
        #[cfg(target_arch = "x86_64")]
        let out = match self.isa {
            MsacIsa::Scalar => self.decode_bool_scalar(f),
            MsacIsa::Sse2(t) | MsacIsa::Avx2(t, _) => {
                crate::dsp::x86::msac::decode_bool_sse2(t, self, f)
            }
        };
        #[cfg(target_arch = "aarch64")]
        let out = match self.isa {
            MsacIsa::Scalar => self.decode_bool_scalar(f),
            MsacIsa::Neon(t) => crate::dsp::arm::giris::msac_decode_bool(t, self, f),
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let out = self.decode_bool_scalar(f);
        #[cfg(feature = "msac-trace")]
        trace::sym("bool", 1, i64::from(out), self.rng, self.cnt, 0, 0, -1);
        out
    }

    #[cfg_attr(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        cold,
        inline(never)
    )]
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        inline(always)
    )]
    fn decode_bool_scalar(&mut self, f: u32) -> u32 {
        let r = self.rng;
        let mut dif = self.dif;
        debug_assert!((dif >> EC_CODE_SHIFT) < r as u64);
        let mut v =
            ((r >> 8).wrapping_mul(f >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB;
        let vw = (v as u64) << EC_CODE_SHIFT;
        let ret = u32::from(dif >= vw);
        if ret != 0 {
            dif = dif.wrapping_sub(vw);
        }
        v = v.wrapping_add(ret.wrapping_mul(r.wrapping_sub(2u32.wrapping_mul(v))));
        self.ctx_norm(dif, v);
        1 - ret
    }

    /// `dav1d_msac_decode_symbol_adapt4` (`msac.h:33`, `msac.asm:87`):
    /// `1 <= n_symbols <= 3` ve `cdf.len() >= 4` için 4-şeritli hızlı yol.
    #[inline(always)]
    pub fn decode_symbol_adapt4(&mut self, cdf: &mut [u16], n_symbols: usize) -> usize {
        debug_assert!((1..=3).contains(&n_symbols));
        #[cfg(feature = "msac-trace")]
        let (trace_c0, trace_cn) = (u32::from(cdf[0]), u32::from(cdf[n_symbols]));
        #[cfg(target_arch = "x86_64")]
        let val = match (self.isa, cdf.first_chunk_mut::<4>()) {
            (MsacIsa::Avx2(_, t_avx2), Some(c4)) => {
                crate::dsp::x86::msac_avx2::decode_symbol_adapt4_avx2(t_avx2, self, c4, n_symbols)
            }
            (MsacIsa::Sse2(t), Some(c4)) => {
                crate::dsp::x86::msac::decode_symbol_adapt4_sse2(t, self, c4, n_symbols)
            }
            _ => self.decode_symbol_adapt_scalar(cdf, n_symbols),
        };
        #[cfg(target_arch = "aarch64")]
        let val = match (self.isa, cdf.first_chunk_mut::<4>()) {
            (MsacIsa::Neon(t), Some(c4)) => {
                crate::dsp::arm::giris::msac_decode_symbol_adapt4(t, self, c4, n_symbols)
            }
            _ => self.decode_symbol_adapt_scalar(cdf, n_symbols),
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let val = self.decode_symbol_adapt_scalar(cdf, n_symbols);
        #[cfg(feature = "msac-trace")]
        trace::sym(
            "adapt", n_symbols, val as i64, self.rng, self.cnt, trace_c0, trace_cn, -1,
        );
        val
    }

    /// `dav1d_msac_decode_symbol_adapt8` (`msac.h:35`, `msac.asm:221`):
    /// `1 <= n_symbols <= 7` ve `cdf.len() >= 8` için 8-şeritli hızlı yol.
    #[inline(always)]
    pub fn decode_symbol_adapt8(&mut self, cdf: &mut [u16], n_symbols: usize) -> usize {
        debug_assert!((1..=7).contains(&n_symbols));
        #[cfg(feature = "msac-trace")]
        let (trace_c0, trace_cn) = (u32::from(cdf[0]), u32::from(cdf[n_symbols]));
        #[cfg(target_arch = "x86_64")]
        let val = match (self.isa, cdf.first_chunk_mut::<8>()) {
            (MsacIsa::Avx2(_, t_avx2), Some(c8)) => {
                crate::dsp::x86::msac_avx2::decode_symbol_adapt8_avx2(t_avx2, self, c8, n_symbols)
            }
            (MsacIsa::Sse2(t), Some(c8)) => {
                crate::dsp::x86::msac::decode_symbol_adapt8_sse2(t, self, c8, n_symbols)
            }
            _ => self.decode_symbol_adapt_scalar(cdf, n_symbols),
        };
        #[cfg(target_arch = "aarch64")]
        let val = match (self.isa, cdf.first_chunk_mut::<8>()) {
            (MsacIsa::Neon(t), Some(c8)) => {
                crate::dsp::arm::giris::msac_decode_symbol_adapt8(t, self, c8, n_symbols)
            }
            _ => self.decode_symbol_adapt_scalar(cdf, n_symbols),
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let val = self.decode_symbol_adapt_scalar(cdf, n_symbols);
        #[cfg(feature = "msac-trace")]
        trace::sym(
            "adapt", n_symbols, val as i64, self.rng, self.cnt, trace_c0, trace_cn, -1,
        );
        val
    }

    /// `dav1d_msac_decode_symbol_adapt16` (`msac.h:37-40`, `msac.asm:267,617`):
    /// `1 <= n_symbols <= 15` ve `cdf.len() >= 16` için 16-şeritli SSE2/AVX2/NEON hızlı yolu.
    #[inline(always)]
    pub fn decode_symbol_adapt16(&mut self, cdf: &mut [u16], n_symbols: usize) -> usize {
        debug_assert!((1..=15).contains(&n_symbols));
        #[cfg(feature = "msac-trace")]
        let (trace_c0, trace_cn) = (u32::from(cdf[0]), u32::from(cdf[n_symbols]));
        #[cfg(target_arch = "x86_64")]
        let val = match (self.isa, cdf.first_chunk_mut::<16>()) {
            (MsacIsa::Avx2(_, t_avx2), Some(c16)) => {
                crate::dsp::x86::msac_avx2::decode_symbol_adapt16_avx2(t_avx2, self, c16, n_symbols)
            }
            (MsacIsa::Sse2(t_sse2), Some(c16)) => {
                crate::dsp::x86::msac::decode_symbol_adapt16_sse2(t_sse2, self, c16, n_symbols)
            }
            _ => self.decode_symbol_adapt_scalar(cdf, n_symbols),
        };
        #[cfg(target_arch = "aarch64")]
        let val = match (self.isa, cdf.first_chunk_mut::<16>()) {
            (MsacIsa::Neon(t), Some(c16)) => {
                crate::dsp::arm::giris::msac_decode_symbol_adapt16(t, self, c16, n_symbols)
            }
            _ => self.decode_symbol_adapt_scalar(cdf, n_symbols),
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let val = self.decode_symbol_adapt_scalar(cdf, n_symbols);
        #[cfg(feature = "msac-trace")]
        trace::sym(
            "adapt", n_symbols, val as i64, self.rng, self.cnt, trace_c0, trace_cn, -1,
        );
        val
    }

    /// `dav1d_msac_decode_symbol_adapt_c` (`msac.c:132`): `n_symbols + 1`
    /// sembollü CDF'den sembol çözer ve (izinliyse) CDF'yi adapte eder.
    ///
    /// `cdf[n_symbols]` sayaç hücresidir (dav1d düzeni). Bozuk akışta
    /// dönen değer `n_symbols` olabilir (dav1d ile aynı davranış).
    #[inline(always)]
    pub fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> usize {
        #[cfg(feature = "msac-trace")]
        let (trace_c0, trace_cn) = (u32::from(cdf[0]), u32::from(cdf[n_symbols]));
        debug_assert!((1..=15).contains(&n_symbols));
        debug_assert!(cdf.len() > n_symbols);
        debug_assert!(cdf[n_symbols] <= 32);

        #[cfg(target_arch = "x86_64")]
        let val = match self.isa {
            MsacIsa::Scalar => self.decode_symbol_adapt_scalar(cdf, n_symbols),
            MsacIsa::Sse2(t_sse2) => {
                if n_symbols <= 3
                    && let Some(c4) = cdf.first_chunk_mut::<4>()
                {
                    crate::dsp::x86::msac::decode_symbol_adapt4_sse2(t_sse2, self, c4, n_symbols)
                } else if n_symbols <= 7
                    && let Some(c8) = cdf.first_chunk_mut::<8>()
                {
                    crate::dsp::x86::msac::decode_symbol_adapt8_sse2(t_sse2, self, c8, n_symbols)
                } else if let Some(c16) = cdf.first_chunk_mut::<16>() {
                    crate::dsp::x86::msac::decode_symbol_adapt16_sse2(t_sse2, self, c16, n_symbols)
                } else {
                    self.decode_symbol_adapt_scalar(cdf, n_symbols)
                }
            }
            MsacIsa::Avx2(_, t_avx2) => {
                if n_symbols <= 3
                    && let Some(c4) = cdf.first_chunk_mut::<4>()
                {
                    crate::dsp::x86::msac_avx2::decode_symbol_adapt4_avx2(
                        t_avx2, self, c4, n_symbols,
                    )
                } else if n_symbols <= 7
                    && let Some(c8) = cdf.first_chunk_mut::<8>()
                {
                    crate::dsp::x86::msac_avx2::decode_symbol_adapt8_avx2(
                        t_avx2, self, c8, n_symbols,
                    )
                } else if let Some(c16) = cdf.first_chunk_mut::<16>() {
                    crate::dsp::x86::msac_avx2::decode_symbol_adapt16_avx2(
                        t_avx2, self, c16, n_symbols,
                    )
                } else {
                    self.decode_symbol_adapt_scalar(cdf, n_symbols)
                }
            }
        };
        #[cfg(target_arch = "aarch64")]
        let val = match self.isa {
            MsacIsa::Scalar => self.decode_symbol_adapt_scalar(cdf, n_symbols),
            MsacIsa::Neon(t_neon) => {
                if n_symbols <= 3
                    && let Some(c4) = cdf.first_chunk_mut::<4>()
                {
                    crate::dsp::arm::giris::msac_decode_symbol_adapt4(t_neon, self, c4, n_symbols)
                } else if n_symbols <= 7
                    && let Some(c8) = cdf.first_chunk_mut::<8>()
                {
                    crate::dsp::arm::giris::msac_decode_symbol_adapt8(t_neon, self, c8, n_symbols)
                } else if let Some(c16) = cdf.first_chunk_mut::<16>() {
                    crate::dsp::arm::giris::msac_decode_symbol_adapt16(t_neon, self, c16, n_symbols)
                } else {
                    self.decode_symbol_adapt_scalar(cdf, n_symbols)
                }
            }
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let val = self.decode_symbol_adapt_scalar(cdf, n_symbols);

        #[cfg(feature = "msac-trace")]
        trace::sym(
            "adapt", n_symbols, val as i64, self.rng, self.cnt, trace_c0, trace_cn, -1,
        );
        val
    }

    #[cfg_attr(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        cold,
        inline(never)
    )]
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        inline(always)
    )]
    pub(crate) fn decode_symbol_adapt_scalar(
        &mut self,
        cdf: &mut [u16],
        n_symbols: usize,
    ) -> usize {
        let c = (self.dif >> EC_CODE_SHIFT) as u32;
        let r = self.rng >> 8;
        let mut v = self.rng;
        let mut u;
        let mut val = 0usize;
        loop {
            u = v;
            v = (r.wrapping_mul((cdf[val] >> EC_PROB_SHIFT) as u32) >> (7 - EC_PROB_SHIFT))
                .wrapping_add(EC_MIN_PROB * (n_symbols - val) as u32);
            if c >= v {
                break;
            }
            val += 1;
        }

        self.ctx_norm(
            self.dif.wrapping_sub((v as u64) << EC_CODE_SHIFT),
            u.wrapping_sub(v),
        );

        if self.allow_update_cdf {
            let count = cdf[n_symbols] as u32;
            let rate = 4 + (count >> 4) + u32::from(n_symbols > 2);
            let mut i = 0;
            while i < val {
                cdf[i] = (cdf[i] as i32 + ((32768 - cdf[i] as i32) >> rate)) as u16;
                i += 1;
            }
            while i < n_symbols {
                cdf[i] = (cdf[i] as i32 - ((cdf[i] as i32) >> rate)) as u16;
                i += 1;
            }
            cdf[n_symbols] = (count + u32::from(count < 32)) as u16;
        }
        val
    }

    /// `dav1d_msac_decode_bool_adapt` (`msac.c:168`, `msac.asm:340`): uyarlanan boolean.
    #[inline(always)]
    pub fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> u32 {
        #[cfg(feature = "msac-trace")]
        let (trace_c0, trace_cn) = (u32::from(cdf[0]), u32::from(cdf[1]));
        debug_assert!(cdf.len() >= 2);
        debug_assert!(cdf[1] <= 32);
        #[cfg(target_arch = "x86_64")]
        let bit = match (self.isa, cdf.first_chunk_mut::<2>()) {
            (MsacIsa::Sse2(t) | MsacIsa::Avx2(t, _), Some(c2)) => {
                crate::dsp::x86::msac::decode_bool_adapt_sse2(t, self, c2)
            }
            _ => self.decode_bool_adapt_scalar(cdf),
        };
        #[cfg(target_arch = "aarch64")]
        let bit = match (self.isa, cdf.first_chunk_mut::<2>()) {
            (MsacIsa::Neon(t), Some(c2)) => {
                crate::dsp::arm::giris::msac_decode_bool_adapt(t, self, c2)
            }
            _ => self.decode_bool_adapt_scalar(cdf),
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let bit = self.decode_bool_adapt_scalar(cdf);
        #[cfg(feature = "msac-trace")]
        trace::sym(
            "bool_adapt",
            1,
            i64::from(bit),
            self.rng,
            self.cnt,
            trace_c0,
            trace_cn,
            -1,
        );
        bit
    }

    #[cfg_attr(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        cold,
        inline(never)
    )]
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        inline(always)
    )]
    fn decode_bool_adapt_scalar(&mut self, cdf: &mut [u16]) -> u32 {
        let bit = self.decode_bool_scalar(cdf[0] as u32);
        if self.allow_update_cdf {
            let count = cdf[1] as u32;
            let rate = 4 + (count >> 4);
            if bit != 0 {
                cdf[0] = (cdf[0] as i32 + ((32768 - cdf[0] as i32) >> rate)) as u16;
            } else {
                cdf[0] = (cdf[0] as i32 - ((cdf[0] as i32) >> rate)) as u16;
            }
            cdf[1] = (count + u32::from(count < 32)) as u16;
        }
        bit
    }

    /// `dav1d_msac_decode_hi_tok` (`msac.c:187`, `msac.asm:589`): 3–15 arası üst token.
    ///
    /// Dört aşamaya kadar `decode_symbol_adapt(cdf, 3)` çağırır; hepsi **aynı**
    /// CDF dizisini kullanır.
    #[inline(always)]
    pub fn decode_hi_tok(&mut self, cdf: &mut [u16]) -> usize {
        #[cfg(feature = "msac-trace")]
        let (trace_c0, trace_cn) = (u32::from(cdf[0]), u32::from(cdf[3]));
        #[cfg(target_arch = "x86_64")]
        let tok = match (self.isa, cdf.first_chunk_mut::<4>()) {
            (MsacIsa::Avx2(_, t_avx2), Some(c4)) => {
                crate::dsp::x86::msac_avx2::decode_hi_tok_avx2(t_avx2, self, c4)
            }
            (MsacIsa::Sse2(t), Some(c4)) => crate::dsp::x86::msac::decode_hi_tok_sse2(t, self, c4),
            _ => self.decode_hi_tok_scalar(cdf),
        };
        #[cfg(target_arch = "aarch64")]
        let tok = match (self.isa, cdf.first_chunk_mut::<4>()) {
            (MsacIsa::Neon(t), Some(c4)) => crate::dsp::arm::giris::msac_decode_hi_tok(t, self, c4),
            _ => self.decode_hi_tok_scalar(cdf),
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let tok = self.decode_hi_tok_scalar(cdf);
        #[cfg(feature = "msac-trace")]
        trace::sym(
            "hi_tok", 1, tok as i64, self.rng, self.cnt, trace_c0, trace_cn, -1,
        );
        tok
    }

    #[cfg_attr(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        cold,
        inline(never)
    )]
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        inline(always)
    )]
    pub(crate) fn decode_hi_tok_scalar(&mut self, cdf: &mut [u16]) -> usize {
        let mut tok_br = self.decode_symbol_adapt_scalar(cdf, 3);
        let mut tok = 3 + tok_br;
        if tok_br == 3 {
            tok_br = self.decode_symbol_adapt_scalar(cdf, 3);
            tok = 6 + tok_br;
            if tok_br == 3 {
                tok_br = self.decode_symbol_adapt_scalar(cdf, 3);
                tok = 9 + tok_br;
                if tok_br == 3 {
                    tok = 12 + self.decode_hi_tok_tail_scalar(cdf);
                }
            }
        }
        tok
    }

    #[inline(never)]
    fn decode_hi_tok_tail_scalar(&mut self, cdf: &mut [u16]) -> usize {
        self.decode_symbol_adapt_scalar(cdf, 3)
    }

    /// `dav1d_msac_decode_subexp` (`msac.c:60`): `n >> k == 8` ön koşuluyla
    /// alt-üssel (subexp) değer; `inv_recenter` ile işaretlenir.
    pub fn decode_subexp(&mut self, r: i32, n: i32, mut k: u32) -> i32 {
        debug_assert_eq!(n >> k, 8);
        let out = {
            #[cfg(feature = "msac-trace")]
            let _suppress = trace::suppress();
            let mut a: u32 = 0;
            if self.decode_bool_equi() != 0 {
                if self.decode_bool_equi() != 0 {
                    k += self.decode_bool_equi() + 1;
                }
                a = 1 << k;
            }
            let v = self.decode_bools(k) + a;
            if r * 2 <= n {
                inv_recenter(r as u32, v) as i32
            } else {
                n - 1 - inv_recenter((n - 1 - r) as u32, v) as i32
            }
        };
        #[cfg(feature = "msac-trace")]
        trace::sym(
            "subexp",
            usize::try_from(n).unwrap_or(0),
            i64::from(out),
            self.rng,
            self.cnt,
            0,
            0,
            -1,
        );
        out
    }

    /// `dav1d_msac_decode_bools` (`msac.h:94`): `n` bit, MSB önce.
    pub fn decode_bools(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = v.wrapping_shl(1) | self.decode_bool_equi();
        }
        v
    }

    /// `dav1d_msac_decode_uniform` (`msac.h:101`): `[0, n)` aralığında düzgün
    /// değer. Sonlu kod (sonlu ağaç) yöntemi.
    pub fn decode_uniform(&mut self, n: u32) -> u32 {
        debug_assert!(n > 1);
        let l = ulog2(n) + 1;
        let m = (1u32 << l) - n;
        let v = self.decode_bools(l - 1);
        if v < m {
            v
        } else {
            v.wrapping_shl(1)
                .wrapping_sub(m)
                .wrapping_add(self.decode_bool_equi())
        }
    }
}

/// `ulog2` (`intops.h:67`): `31 ^ clz(v)`.
#[inline]
pub fn ulog2(v: u32) -> u32 {
    31 ^ v.leading_zeros()
}

/// `inv_recenter` (`intops.h:75`).
#[inline]
pub fn inv_recenter(r: u32, v: u32) -> u32 {
    if v > r.wrapping_shl(1) {
        v
    } else if v & 1 == 0 {
        (v >> 1) + r
    } else {
        r - ((v + 1) >> 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Boş tamponda `ctx_refill`: kalan bütün bitler 1, sayaç `-15`.
    #[test]
    fn bos_tampon_durumu_dav1d_ile_ayni() {
        let s = MsacContext::init(&[], false);
        assert_eq!(s.dif(), !(u64::MAX << 63));
        assert_eq!(s.rng(), 0x8000);
        assert_eq!(s.cnt(), -15);
        assert_eq!(s.buf_pos(), 0);
        assert!(s.allow_update_cdf());
    }

    /// Tek baytlık tampon: ilk bayt tümleyeniyle 55. yuvaya, sonra veri sonu
    /// dolgusu 47. yuvaya yazılır; sayaç `-7`.
    #[test]
    fn tek_bayt_ve_veri_sonu_dolgusu() {
        let s = MsacContext::init(&[0x00], true);
        // 0x00 baytı tümleyeniyle 55. yuvaya, veri sonu dolgusu [0, 55)'e yazılır.
        assert_eq!(s.dif(), (0xffu64 << 55) | !(u64::MAX << 55));
        assert_eq!(s.cnt(), -7);
        assert!(!s.allow_update_cdf());
    }

    /// Yedi bayt tam dolgu: sayaç `41` (dav1d `cnt = 64 - (-1) - 24`).
    #[test]
    fn yedi_bayt_tam_dolgu() {
        let s = MsacContext::init(&[0u8; 7], false);
        assert_eq!(s.cnt(), 41);
        assert_eq!(s.buf_pos(), 7);
    }

    /// `decode_bool_equi` her zaman 0/1 döner ve veri sonunda paniklemez.
    #[test]
    fn veri_sonunda_bool_equi_sonlu() {
        let mut s = MsacContext::init(&[], false);
        for _ in 0..1000 {
            assert!(s.decode_bool_equi() <= 1);
        }
    }

    /// `decode_bools(n)` `n` bitten oluşur; `decode_uniform(n)` `[0, n)`.
    #[test]
    fn bools_ve_uniform_araligi() {
        let mut s = MsacContext::init(&[0x5a, 0xa5, 0x3c, 0x0f, 0xf0], false);
        for n in 1..=16 {
            assert!(s.decode_bools(n) < (1u32 << n));
        }
        let mut s = MsacContext::init(&[0x11, 0x22, 0x33, 0x44], false);
        for n in 2..=64 {
            assert!(s.decode_uniform(n) < n);
        }
    }

    /// `decode_symbol_adapt` bütün sembolleri kapsar; adaptasyon sayacı artar.
    #[test]
    fn sembol_adapt_sayaci_ve_araligi() {
        let mut cdf = [16384u16, 8192, 4096, 0];
        let mut s = MsacContext::init(&[0x00; 16], false);
        let mut seen = [false; 4];
        for _ in 0..64 {
            let val = s.decode_symbol_adapt(&mut cdf, 3);
            assert!(val <= 3, "val={val}");
            seen[val] = true;
        }
        assert_eq!(cdf[3], 32, "sayaç 32'de doygun olmalı");
        assert!(seen.iter().any(|&x| x));
    }

    /// Adaptasyon kapalıyken CDF hiç değişmez.
    #[test]
    fn adaptasyon_kapali_cdf_degismez() {
        let mut sym = [20000u16, 10000, 4000, 5];
        let mut boolean = [18000u16, 7];
        let sym_once = sym;
        let bool_once = boolean;
        let mut s = MsacContext::init(&[0x99; 16], true);
        for _ in 0..64 {
            let _ = s.decode_symbol_adapt(&mut sym, 3);
            let _ = s.decode_bool_adapt(&mut boolean);
            let _ = s.decode_bool_adapt(&mut boolean);
        }
        assert_eq!(sym, sym_once);
        assert_eq!(boolean, bool_once);
    }

    /// `decode_hi_tok` 3–15 aralığında kalır ve dört adapt çağrısı yapar.
    #[test]
    fn hi_tok_araligi() {
        let mut cdf = [16384u16, 16384, 16384, 0];
        let mut s = MsacContext::init(&[0xde, 0xad, 0xbe, 0xef, 0x00, 0x11], false);
        for _ in 0..32 {
            let tok = s.decode_hi_tok(&mut cdf);
            assert!((3..=15).contains(&tok), "tok={tok}");
        }
    }

    /// `decode_subexp`: `n >> k == 8`; sonuç `inv_recenter` sözleşmesine uyar.
    #[test]
    fn subexp_sozlesmesi() {
        let mut s = MsacContext::init(&[0x37; 24], false);
        for k in 0..=4u32 {
            let n = 8i32 << k;
            for r in [0, 1, n / 2, n - 1] {
                let v = s.decode_subexp(r, n, k);
                assert!((-8 * n..=8 * n).contains(&v), "v={v}");
            }
        }
    }

    /// `inv_recenter` dav1d `intops.h` ile aynı dal yapısı.
    #[test]
    fn inv_recenter_dallari() {
        assert_eq!(inv_recenter(5, 11), 11); // v > 2r
        assert_eq!(inv_recenter(5, 4), 7); // çift
        assert_eq!(inv_recenter(5, 3), 3); // tek: 5 - 2
        assert_eq!(inv_recenter(0, 0), 0);
    }

    /// `check_trailing_bits` (`dav1d` `check_trailing_bits_after_symbol_coder`, `decode.c:2620-2635`)
    /// bütün 8 bit hizalamalarında `1` işaret bitini ve izleyen `0` dolgusunu doğrular.
    #[test]
    fn check_trailing_bits_tum_hizalamalar_ve_bozuk_dolgular() {
        for consumed_in_byte in 0..8i32 {
            // `n_bits = -(cnt + 14)`: `buf[pos - 1]` içinde `consumed_in_byte` (0..8) bit
            // tüketilmişse kalan `1..=8` bit için `n_bits = -7 + consumed_in_byte` (`-7..=0`),
            // yani `cnt = -7 - consumed_in_byte` (`-14..=-7`) ve `n_bytes = 0` (`p_idx = pos`).
            let cnt = -7 - consumed_in_byte;
            let marker = 128u8 >> (consumed_in_byte as u32);
            let buf_ok = [0xAAu8, marker, 0x00, 0x00];
            let s_ok = MsacContext {
                buf: &buf_ok,
                pos: 2,
                dif: 0,
                rng: 0x8000,
                cnt,
                allow_update_cdf: true,
                isa: MsacIsa::Scalar,
            };
            assert!(
                s_ok.check_trailing_bits(),
                "consumed={consumed_in_byte} marker=0x{marker:02x} geçerli olmalı"
            );

            // İşaret biti eksik (`0x00`): reddedilmeli.
            let buf_zero = [0xAAu8, 0x00, 0x00];
            let s_zero = MsacContext {
                buf: &buf_zero,
                pos: 2,
                ..s_ok
            };
            assert!(
                !s_zero.check_trailing_bits(),
                "consumed={consumed_in_byte} eksik işaret biti reddedilmeli"
            );

            // İşaret bitinden sonra aynı baytta fazladan `1` biti varsa: reddedilmeli.
            if marker > 1 {
                let buf_low_bit = [0xAAu8, marker | 1, 0x00];
                let s_low = MsacContext {
                    buf: &buf_low_bit,
                    pos: 2,
                    ..s_ok
                };
                assert!(
                    !s_low.check_trailing_bits(),
                    "consumed={consumed_in_byte} düşük bitteki 1 reddedilmeli"
                );
            }

            // Sonraki dolgu baytlarında sıfır dışı bayt varsa: reddedilmeli.
            let buf_bad_tail = [0xAAu8, marker, 0x00, 0x01];
            let s_tail = MsacContext {
                buf: &buf_bad_tail,
                pos: 2,
                ..s_ok
            };
            assert!(
                !s_tail.check_trailing_bits(),
                "consumed={consumed_in_byte} sıfır dışı kuyruk baytı reddedilmeli"
            );
        }

        // `cnt <= -15` (aşım): her zaman `false`.
        let buf = [0x80u8, 0x00];
        let s_over = MsacContext {
            buf: &buf,
            pos: 1,
            dif: 0,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: true,
            isa: MsacIsa::Scalar,
        };
        assert!(!s_over.check_trailing_bits());
    }
}
