//! Sınır denetimli AV1 bit okuyucu (F2a / S2a.1).
//!
//! ## Sorumluluk
//!
//! AV1 bit akışının temel okuma ilkelleri: `f(n)` (bit alanı), `s(n)` (işaretli
//! alan), `leb128()`, `uvlc()`, `ns(max)` (düzgün dağılım), `su(ref, n)`
//! (bağımlı alt-üstel), bayt hizalama ve `trailing_bits`.
//!
//! ## Sözleşme
//!
//! - **Sınır denetimli (GUV-01):** Her okuma, tamponun bit uzunluğuna karşı
//!   denetlenir. Yetersiz bit [`BitstreamErrorKind::UnexpectedEnd`] üretir;
//!   tampon sonuna yazan, taşan veya negatif indeksli bir yol **yoktur**.
//! - **Tanımlı aritmetik (SMD-01):** `f(n)` için `n ∈ 0..=32`; `u32` sonuç
//!   `u64` ara değerle kurulur, kaydırma miktarı hiçbir yerde 32'yi aşmaz.
//!   `leb128` en fazla 8 bayt okur ve `u32` aralığını aşan değeri reddeder.
//! - **Hata konumu:** Her hata [`BitstreamError::bit_offset`] ile tamponun
//!   başından itibaren **bit** konumunu taşır (dav1d `dav1d_get_bits_pos`).
//!   Başarısız okuma konumu **ilerletmez**; çağıran hatayı yukarı taşır.
//! - **Panik yok:** `assert!`, `unwrap`, dilim indeksleme taşması yoktur;
//!   taşma/kesik veri `Err` döner (BLK-05).
//!
//! ## Upstream karşılığı (dav1d `99f3354`, salt-okunur)
//!
//! | Rust | dav1d kaynağı | AV1 spesifikasyonu |
//! |---|---|---|
//! | [`BitReader::read_bit`] | `src/getbits.c` 47–63 (`dav1d_get_bit`) | §5.3.2 (bit alanları) |
//! | [`BitReader::read_bits`] | `src/getbits.c` 65–93 (`refill` + `GET_BITS` 80–93) | §4.10 (temel ayrıştırma: `f(n)`) |
//! | [`BitReader::read_sbits`] | `src/getbits.c` 93 (`dav1d_get_sbits`) | §4.10 (temel ayrıştırma: `s(n)`) |
//! | [`BitReader::read_uleb128`] | `src/getbits.c` 95–112 | §5.3.3 (`obu_size`) + §4.10 (`leb128()`) |
//! | [`BitReader::read_uvlc`] | `src/getbits.c` 125–136 | §4.10 (`uvlc()`) |
//! | [`BitReader::read_uniform`] | `src/getbits.c` 114–123 (`dav1d_get_uniform`) | §4.10 (`ns()`) |
//! | [`BitReader::read_subexp`] | `src/getbits.c` 138–165 (`dav1d_get_bits_subexp`) | §4.10 (`su()`) |
//! | [`BitReader::read_trailing_bits`] | `src/obu.c` 48–70 (`check_trailing_bits`) | §5.3.5 `trailing_bits()` / §6.4.4 |
//!
//! > Not: `f(n)`/`su()`/`ns()`/`uvlc()`/`leb128()` AV1 spesifikasyonunun
//! > **§4.10 (Parsing process)** ailesinde tanımlıdır; sequence header bu adımda
//! > yalnız `uvlc()` kullanır (timing info, `num_ticks_per_picture`).

#![forbid(unsafe_code)]

use crate::error::{BitstreamError, BitstreamErrorKind, Result};

/// Bit düzeyinde, sınır denetimli okuyucu.
///
/// Tamponu ödünç alır; sahiplik veya tahsis yoktur. Bit sırası AV1/MSB-first'dir:
/// ilk okunan bit, baytın **en anlamlı** bitidir (dav1d `getbits.c` 51–67).
#[derive(Clone, Copy, Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Okunacak sonraki bitin mutlak konumu.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Tamponun tamamı üzerinde, konum 0'dan başlayan okuyucu.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Geçerli bit konumu (tampon başından itibaren).
    #[must_use]
    pub const fn bit_position(&self) -> usize {
        self.pos
    }

    /// Geçerli bayt konumu (yalnız bayt hizalıyken anlamlıdır).
    #[must_use]
    pub const fn byte_position(&self) -> usize {
        self.pos / 8
    }

    /// Kalan bit sayısı.
    #[must_use]
    pub const fn remaining_bits(&self) -> usize {
        self.data.len() * 8 - self.pos
    }

    /// Bayt hizalı mı?
    #[must_use]
    pub const fn is_byte_aligned(&self) -> bool {
        self.pos.is_multiple_of(8)
    }

    /// Konumu sonraki bayt sınırına taşır (dav1d `dav1d_bytealign_get_bits`).
    pub fn align_to_byte(&mut self) {
        let rem = self.pos % 8;
        if rem != 0 {
            self.pos += 8 - rem;
        }
    }

    fn error(&self, kind: BitstreamErrorKind, offset: usize) -> BitstreamError {
        BitstreamError::new(kind, offset)
    }

    /// Tek bit okur (`0` veya `1`).
    ///
    /// # Hatalar
    ///
    /// [`BitstreamErrorKind::UnexpectedEnd`]: tamponda bit kalmadı.
    pub fn read_bit(&mut self) -> Result<u32> {
        if self.pos >= self.data.len() * 8 {
            return Err(self
                .error(BitstreamErrorKind::UnexpectedEnd, self.pos)
                .into());
        }
        let bit = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(u32::from(bit))
    }

    /// `n` bit okur (`0 <= n <= 32`); sonuç sağa hizalı `u32`.
    ///
    /// `n == 0` tanımlıdır ve `0` döner (hiç bit tüketmez); dav1d `get_bits`
    /// `n > 0` beklediği için bu durum yalnız yardımcı söz dizimlerinde kullanılır.
    ///
    /// # Hatalar
    ///
    /// - [`BitstreamErrorKind::InvalidBitCount`]: `n > 32`.
    /// - [`BitstreamErrorKind::UnexpectedEnd`]: `n` bit kalmadı.
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return Err(self
                .error(BitstreamErrorKind::InvalidBitCount(n), self.pos)
                .into());
        }
        if n == 0 {
            return Ok(0);
        }
        let needed = n as usize;
        if needed > self.remaining_bits() {
            return Err(self
                .error(BitstreamErrorKind::UnexpectedEnd, self.pos)
                .into());
        }
        let mut acc: u64 = 0;
        let mut index = self.pos;
        let end = self.pos + needed;
        while index < end {
            let bit = (self.data[index >> 3] >> (7 - (index & 7))) & 1;
            acc = (acc << 1) | u64::from(bit);
            index += 1;
        }
        self.pos = end;
        Ok(acc as u32)
    }

    /// `n` bit işaretli okur (ikiye tümleyen; dav1d `dav1d_get_sbits`).
    ///
    /// # Hatalar
    ///
    /// [`BitstreamErrorKind::InvalidBitCount`] / [`BitstreamErrorKind::UnexpectedEnd`].
    pub fn read_sbits(&mut self, n: u32) -> Result<i32> {
        if n > 32 {
            return Err(self
                .error(BitstreamErrorKind::InvalidBitCount(n), self.pos)
                .into());
        }
        if n == 0 {
            return Ok(0);
        }
        let value = self.read_bits(n)?;
        if n == 32 {
            return Ok(value as i32);
        }
        let sign_bit = 1u32 << (n - 1);
        if value & sign_bit == 0 {
            Ok(value as i32)
        } else {
            Ok((i64::from(value) - (1i64 << n)) as i32)
        }
    }

    /// `leb128()`: en fazla 8 bayt, `u32` sonuç (dav1d `dav1d_get_uleb128`).
    ///
    /// # Hatalar
    ///
    /// - [`BitstreamErrorKind::UnexpectedEnd`]: bayt kalmadı.
    /// - [`BitstreamErrorKind::Leb128Overflow`]: değer `u32` aralığını aştı.
    /// - [`BitstreamErrorKind::Leb128TooLong`]: 8 bayttan sonra devam biti 1.
    pub fn read_uleb128(&mut self) -> Result<u32> {
        let start = self.pos;
        let mut value: u64 = 0;
        let mut shift: u32 = 0;
        let mut more: u32;
        loop {
            let byte = u64::from(self.read_bits(8)?);
            more = (byte & 0x80) as u32;
            value |= (byte & 0x7f) << shift;
            shift += 7;
            if more == 0 || shift >= 56 {
                break;
            }
        }
        if more != 0 {
            return Err(self.error(BitstreamErrorKind::Leb128TooLong, start).into());
        }
        if value > u64::from(u32::MAX) {
            return Err(self.error(BitstreamErrorKind::Leb128Overflow, start).into());
        }
        Ok(value as u32)
    }

    /// `uvlc()`: alt-üstel (Exp-Golomb) değer; 32 sıfır biti taşma sayılır.
    ///
    /// dav1d `dav1d_get_vlc` (getbits.c 125–136) ile aynı: ilk bit 1 ise `0`;
    /// değilse `n_bits` adet sıfır sayılır, `1` bitini `n_bits` bit izler.
    ///
    /// # Hatalar
    ///
    /// - [`BitstreamErrorKind::UvlcOverflow`]: 32 sıfır biti (dav1d `UINT32_MAX`).
    /// - [`BitstreamErrorKind::UnexpectedEnd`]: bit kalmadı.
    pub fn read_uvlc(&mut self) -> Result<u32> {
        let start = self.pos;
        if self.read_bit()? != 0 {
            return Ok(0);
        }
        let mut n_bits: u32 = 0;
        loop {
            n_bits += 1;
            if n_bits == 32 {
                return Err(self.error(BitstreamErrorKind::UvlcOverflow, start).into());
            }
            if self.read_bit()? != 0 {
                break;
            }
        }
        let rest = self.read_bits(n_bits)?;
        Ok(((1u32 << n_bits) - 1) + rest)
    }

    /// `ns(max)`: `0..max-1` aralığında düzgün dağılım (dav1d `dav1d_get_uniform`).
    ///
    /// # Hatalar
    ///
    /// - [`BitstreamErrorKind::InvalidUniformMax`]: `max < 2`.
    /// - [`BitstreamErrorKind::UnexpectedEnd`]: bit kalmadı.
    pub fn read_uniform(&mut self, max: u32) -> Result<u32> {
        if max < 2 {
            return Err(self
                .error(BitstreamErrorKind::InvalidUniformMax(max), self.pos)
                .into());
        }
        // l = floor(log2(max)) + 1; u64 ara değerler taşmayı önler.
        let l = 32 - max.leading_zeros();
        let m = (1u64 << l) - u64::from(max);
        let v = u64::from(self.read_bits(l - 1)?);
        let value = if v < m {
            v
        } else {
            (v << 1) - m + u64::from(self.read_bit()?)
        };
        // 0 <= value < max <= u32::MAX olduğundan dönüşüm kayıpsızdır.
        Ok(value as u32)
    }

    /// `su(ref, n)`: bağımlı alt-üstel değer (dav1d `dav1d_get_bits_subexp`).
    ///
    /// dav1d formülü: `get_bits_subexp_u(ref + (1 << n), 2 << n) - (1 << n)`.
    /// Pratikte `n <= 12` kullanılır (warped motion matrisi, obu.c 1037–1057).
    ///
    /// # Hatalar
    ///
    /// - [`BitstreamErrorKind::InvalidSubexp`]: `n > 30` veya `ref` aralık dışı.
    /// - [`BitstreamErrorKind::InvalidUniformMax`] / [`BitstreamErrorKind::UnexpectedEnd`].
    pub fn read_subexp(&mut self, reference: i32, n: u32) -> Result<i32> {
        if n > 30 {
            return Err(self
                .error(BitstreamErrorKind::InvalidSubexp, self.pos)
                .into());
        }
        let start = self.pos;
        let shift = 1i64 << n;
        let ref_u = i64::from(reference) + shift;
        let range = 2i64 << n;
        if ref_u < 0 || ref_u > i64::from(u32::MAX) || range > i64::from(u32::MAX) {
            return Err(self.error(BitstreamErrorKind::InvalidSubexp, start).into());
        }
        let value = self.read_subexp_u(ref_u as u32, range as u32, start)?;
        Ok((i64::from(value) - shift) as i32)
    }

    /// `get_bits_subexp_u`: `read_subexp`'in işaretsiz çekirdeği (getbits.c 158–174).
    fn read_subexp_u(&mut self, reference: u32, n: u32, start: usize) -> Result<u32> {
        let mut v: u64 = 0;
        let mut i: u32 = 0;
        loop {
            // dav1d: b = i ? 3 + i - 1 : 3
            let b = if i == 0 { 3 } else { 3 + i - 1 };
            if b > 30 {
                return Err(self.error(BitstreamErrorKind::InvalidSubexp, start).into());
            }
            let step = 3u64 << b;
            if u64::from(n) < v + step {
                let v32 = u32::try_from(v)
                    .map_err(|_| BitstreamError::new(BitstreamErrorKind::InvalidSubexp, start))?;
                let max = n
                    .checked_sub(v32)
                    .and_then(|diff| diff.checked_add(1))
                    .ok_or_else(|| BitstreamError::new(BitstreamErrorKind::InvalidSubexp, start))?;
                v += u64::from(self.read_uniform(max)?);
                break;
            }
            if self.read_bit()? == 0 {
                v += u64::from(self.read_bits(b)?);
                break;
            }
            v += 1u64 << b;
            i += 1;
        }
        let v = u32::try_from(v)
            .map_err(|_| BitstreamError::new(BitstreamErrorKind::InvalidSubexp, start))?;
        // dav1d: ref * 2 <= n ? inv_recenter(ref, v) : n - inv_recenter(n - ref, v)
        let recentered = if u64::from(reference) * 2 <= u64::from(n) {
            inv_recenter(reference, v)
        } else {
            n - inv_recenter(n - reference, v)
        };
        Ok(recentered)
    }

    /// `trailing_bits()` okur (dav1d `check_trailing_bits`, obu.c 48–70).
    ///
    /// Bayrağın kendisi her zaman okunur; tampon bittiyse hata döner. `strict`
    /// iken dav1d'nin `strict_std_compliance` yoluna uyulur: `trailing_one_bit`
    /// `1` olmalı, kalan bitler ve sonraki baytlar `0` olmalıdır.
    ///
    /// # Hatalar
    ///
    /// - [`BitstreamErrorKind::UnexpectedEnd`]: trailing bit yok.
    /// - [`BitstreamErrorKind::TrailingBits`]: katı kipte kural ihlali.
    pub fn read_trailing_bits(&mut self, strict: bool) -> Result<()> {
        let start = self.pos;
        let trailing_one_bit = self.read_bit()?;
        if !strict {
            return Ok(());
        }
        if trailing_one_bit != 1 {
            return Err(self.error(BitstreamErrorKind::TrailingBits, start).into());
        }
        while !self.is_byte_aligned() {
            if self.read_bit()? != 0 {
                return Err(self.error(BitstreamErrorKind::TrailingBits, start).into());
            }
        }
        if self.data[self.byte_position()..].iter().any(|&b| b != 0) {
            return Err(self.error(BitstreamErrorKind::TrailingBits, start).into());
        }
        self.pos = self.data.len() * 8;
        Ok(())
    }
}

/// dav1d `inv_recenter` (include/common/intops.h 75–82).
fn inv_recenter(r: u32, v: u32) -> u32 {
    let r = u64::from(r);
    let v = u64::from(v);
    if v > r * 2 {
        v as u32
    } else if v & 1 == 0 {
        (v >> 1) as u32 + r as u32
    } else {
        (r - ((v + 1) >> 1)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitstreamErrorKind, inv_recenter};
    use crate::error::Error;

    fn kind(err: Error) -> BitstreamErrorKind {
        match err {
            Error::Bitstream(err) => err.kind,
            other => panic!("bit akışı hatası bekleniyordu: {other:?}"),
        }
    }

    #[test]
    fn f_n_sinirlari_ve_msb_sirasi() {
        // 0b1011_0010, 0b0100_0000
        let data = [0xB2, 0x40];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bit().expect("bit"), 1);
        assert_eq!(br.read_bits(3).expect("3 bit"), 0b011);
        assert_eq!(br.read_bits(4).expect("4 bit"), 0b0010);
        assert_eq!(br.read_bits(8).expect("8 bit"), 0x40);
        assert_eq!(br.remaining_bits(), 0);
        assert_eq!(
            kind(br.read_bit().expect_err("veri bitti")),
            BitstreamErrorKind::UnexpectedEnd
        );
    }

    #[test]
    fn f_32_ve_sifir_bit_tasmasiz() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(32).expect("32 bit"), u32::MAX);
        assert_eq!(br.read_bits(0).expect("0 bit"), 0);
        assert!(matches!(
            br.read_bits(33).expect_err("33 bit geçersiz"),
            Error::Bitstream(err) if err.kind == BitstreamErrorKind::InvalidBitCount(33)
        ));
    }

    #[test]
    fn işaretli_okuma_ikiye_tumleyen() {
        let data = [0b1111_1000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_sbits(4).expect("4 bit"), -1);
        assert_eq!(br.read_sbits(4).expect("4 bit"), -8);
        let mut br = BitReader::new(&[0x7F]);
        assert_eq!(br.read_sbits(8).expect("8 bit"), 127);
        let mut br = BitReader::new(&[0x80, 0x00, 0x00, 0x00]);
        assert_eq!(br.read_sbits(32).expect("32 bit"), i32::MIN);
    }

    #[test]
    fn leb128_gecerli_ve_en_buyuk() {
        // 0 -> 0x00
        assert_eq!(BitReader::new(&[0x00]).read_uleb128().expect("0"), 0);
        // 300 -> 0xAC 0x02
        assert_eq!(
            BitReader::new(&[0xAC, 0x02]).read_uleb128().expect("300"),
            300
        );
        // u32::MAX -> 0xFF 0xFF 0xFF 0xFF 0x0F
        assert_eq!(
            BitReader::new(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F])
                .read_uleb128()
                .expect("u32 max"),
            u32::MAX
        );
        // 8 baytlık, değeri sıfır olan biçim geçerlidir (dav1d de kabul eder).
        assert_eq!(
            BitReader::new(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00])
                .read_uleb128()
                .expect("8 bayt sıfır"),
            0
        );
    }

    #[test]
    fn leb128_gecersiz_ve_kesik() {
        // 5. bayttan sonra u32 taşması: 0xFF x4 + 0x1F (35 bit)
        let overflow = BitReader::new(&[0xFF, 0xFF, 0xFF, 0xFF, 0x1F])
            .read_uleb128()
            .expect_err("taşma");
        assert_eq!(kind(overflow), BitstreamErrorKind::Leb128Overflow);
        // 8 bayt, son bayt 0x80 -> devam biti hâlâ 1
        let too_long = BitReader::new(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80])
            .read_uleb128()
            .expect_err("uzun");
        assert_eq!(kind(too_long), BitstreamErrorKind::Leb128TooLong);
        // Kesik: devam biti 1 ama bayt yok
        let truncated = BitReader::new(&[0x80]).read_uleb128().expect_err("kesik");
        assert_eq!(kind(truncated), BitstreamErrorKind::UnexpectedEnd);
    }

    /// Test için `uvlc()` kodlayıcısı (spec §4.10, `uvlc()`).
    fn encode_uvlc(mut value: u32, bits: &mut Vec<u8>) {
        if value == 0 {
            bits.push(1);
            return;
        }
        value += 1;
        let n_bits = 31 - value.leading_zeros();
        for _ in 0..n_bits {
            bits.push(0);
        }
        bits.push(1);
        for i in (0..n_bits).rev() {
            bits.push(((value >> i) & 1) as u8);
        }
    }

    fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, bit) in bits.iter().enumerate() {
            if *bit != 0 {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }

    #[test]
    fn uvlc_sinirlari() {
        for value in [0u32, 1, 2, 3, 7, 8, 1000, u32::MAX - 1] {
            let mut bits = Vec::new();
            encode_uvlc(value, &mut bits);
            let bytes = bits_to_bytes(&bits);
            let mut br = BitReader::new(&bytes);
            let got = br.read_uvlc().expect("uvlc");
            assert_eq!(got, value, "uvlc değeri {value}");
            let trailing = br.remaining_bits();
            assert!(trailing < 8, "uvlc {value} için {trailing} bit arttı");
        }
        // 1 (değer 0) tek bittir.
        assert_eq!(BitReader::new(&[0x80]).read_uvlc().expect("0"), 0);
        // 32 sıfır biti taşma sayılır.
        let zeros = [0u8; 4];
        let mut br = BitReader::new(&zeros);
        let err = br.read_uvlc().expect_err("taşma");
        assert_eq!(kind(err), BitstreamErrorKind::UvlcOverflow);
        // Kesik veri: yedi sıfırdan sonra 1, ardından gereken 7 bit yok.
        let err = BitReader::new(&[0x01]).read_uvlc().expect_err("kesik");
        assert_eq!(kind(err), BitstreamErrorKind::UnexpectedEnd);
    }

    #[test]
    fn ns_dortgen_aralik_ve_gecersiz_max() {
        // max=5: l=3, m=3. '01' -> v=1 (<3) => 1.
        assert_eq!(
            BitReader::new(&[0b0100_0000]).read_uniform(5).expect("ns"),
            1
        );
        // '110' -> v=3 (>=3), ek bit 0 => (6-3)+0 = 3.
        assert_eq!(
            BitReader::new(&[0b1100_0000]).read_uniform(5).expect("ns"),
            3
        );
        // '111' -> v=3, ek bit 1 => 4.
        assert_eq!(
            BitReader::new(&[0b1110_0000]).read_uniform(5).expect("ns"),
            4
        );
        assert_eq!(
            kind(BitReader::new(&[0xFF]).read_uniform(1).expect_err("max")),
            BitstreamErrorKind::InvalidUniformMax(1)
        );
        // max=2 (uç durum): l=2, m=2, tek bit değerin kendisidir.
        assert_eq!(BitReader::new(&[0x80]).read_uniform(2).expect("ns"), 1);
        assert_eq!(BitReader::new(&[0x00]).read_uniform(2).expect("ns"), 0);
    }

    #[test]
    fn su_sinirli_ve_kesik_veride_hata() {
        // Tüm bitler 1: v küçük kalır, sonuç aralıkta olmalı.
        let data = [0xFFu8; 16];
        let mut br = BitReader::new(&data);
        let value = br.read_subexp(0, 4).expect("su");
        assert!(
            (-16..=16).contains(&value),
            "su sonucu aralık dışı: {value}"
        );
        // n > 30 reddedilir.
        let err = BitReader::new(&[0x00]).read_subexp(0, 31).expect_err("n");
        assert_eq!(kind(err), BitstreamErrorKind::InvalidSubexp);
        // Kesik veri hata verir (panik yok).
        let mut br = BitReader::new(&[]);
        assert!(br.read_subexp(0, 2).is_err());
    }

    #[test]
    fn inv_recenter_dav1d_ornekleri() {
        // dav1d intops.h 75-82: v > 2r -> v; çift -> r + v/2; tek -> r - (v+1)/2.
        assert_eq!(inv_recenter(3, 7), 7);
        assert_eq!(inv_recenter(3, 4), 5);
        assert_eq!(inv_recenter(3, 3), 1);
        assert_eq!(inv_recenter(0, 0), 0);
        // v > 2r ise v aynen döner.
        assert_eq!(inv_recenter(0, 1), 1);
        assert_eq!(inv_recenter(0, 2), 2);
    }

    #[test]
    fn trailing_bits_gevsek_ve_kati() {
        let mut br = BitReader::new(&[0x80]);
        br.read_trailing_bits(false).expect("gevşek");
        assert_eq!(br.bit_position(), 1);
        let mut br = BitReader::new(&[0x80]);
        br.read_trailing_bits(true).expect("katı");
        assert_eq!(br.bit_position(), 8);

        // İlk bit 0: katı kipte hata, gevşek kipte kabul.
        assert!(BitReader::new(&[0x00]).read_trailing_bits(true).is_err());
        BitReader::new(&[0x00])
            .read_trailing_bits(false)
            .expect("gevşek");
        // Takip eden bit kirli.
        let err = BitReader::new(&[0x81])
            .read_trailing_bits(true)
            .expect_err("kirli");
        assert_eq!(kind(err), BitstreamErrorKind::TrailingBits);
        // Takip eden bayt kirli.
        let err = BitReader::new(&[0x80, 0x01])
            .read_trailing_bits(true)
            .expect_err("bayt");
        assert_eq!(kind(err), BitstreamErrorKind::TrailingBits);
        // Hiç bit yok: her iki kipte de hata.
        assert!(BitReader::new(&[]).read_trailing_bits(false).is_err());
    }

    #[test]
    fn hizalama_ve_konum() {
        let mut br = BitReader::new(&[0xF0, 0x0F]);
        br.read_bits(3).expect("3 bit");
        assert_eq!(br.bit_position(), 3);
        assert!(!br.is_byte_aligned());
        br.align_to_byte();
        assert_eq!(br.bit_position(), 8);
        assert!(br.is_byte_aligned());
        assert_eq!(br.remaining_bits(), 8);
        assert_eq!(br.read_bits(8).expect("8 bit"), 0x0F);
    }
}
