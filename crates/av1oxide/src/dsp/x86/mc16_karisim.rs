//! MC karışım AVX2 çekirdekleri (16 bpc, 10/12 bit) — F4-MC c (P34):
//! `blend`, `blend_v`, `blend_h` (inter-intra / maske ve OBMC).
//!
//! ## Sözleşme
//!
//! Girişte O(1) denetim ([`crate::dsp::mc::karisim16_sozlesmesi`]): bit
//! derinliği 10/12 ve `bitdepth_max` tutarlı; `w`, `h` 2'nin kuvveti —
//! `blend` `w` 4…32, `h` 4…32 (`w·h ≥ 16`); `blend_v` `w` 2…32, `h` 2…128;
//! `blend_h` `w` 2…128, `h` 2…32 (OBMC maske tablosu); `tmp` en az `w·h`
//! piksel (satırlar bitişik), `blend`'de `mask` en az `w·h` bayt; hedef satır
//! adımı `≥ 2w` bayt, `dst` yazılan son satırın sonuna kadar. Sözleşme dışı
//! girdi skalere devreder ve sayılır. Okuma/yazma yalnız `w` piksellik
//! satırlar (`blend_h` yalnız ilk `(h·3) >> 2` satır).
//!
//! **Değer tanım kümesi:** hedef ve `tmp` pikselleri `≤ bitdepth_max ≤ 4095`
//! (çözme hattının piksel tamponları; skaler gövde de bu kümeyi `debug_assert`
//! ile ister), maske 0…64 (dav1d maske tabloları). Bu kümede `i16` çarpımı ve
//! `packusdw` skaler formülle bit-exact'tir.
//!
//! ## Yöntem
//!
//! `(d·(64−m) + t·m + 32) >> 6`: hedef ve `tmp` pikselleri `punpck{l,h}wd` ile
//! ikili, ağırlıklar `(64−m, m)` ikilisi, `pmaddwd` → `i32`, `+ 32`, `psrad 6`,
//! `packusdw` sırayı geri kurar (bir adım 16 piksel). `w ≥ 16` satır başına
//! 16'lık parçalar; `w` 4 ve 8'de bir ymm'e `16/w` satır toplanır (kısmi
//! grupta yalnız yazılan satırlar okunur); `w = 2` girişte doğrudan tamsayı formülüyle
//! (`#[target_feature]` çağrısı olmadan).
//! `blend_v`'de maske sütuna (`OBMC_MASKS[w + x]`; son çeyrek 0, hedef aynı
//! kalır), `blend_h`'de satıra (`OBMC_MASKS[h + y]`) bağlıdır.
//!
//! `unsafe` yalnız belirteçli girişlerdeki `#[target_feature]` çağrısıdır;
//! dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır. Yükleme/saklama
//! [`super::mem`] yardımcılarıyla yapılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, load_u8x32, store_u8x8, store_u8x16, store_u8x32};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{BlendDirFn, BlendFn, bd16_tutarli, karisim16_sozlesmesi, skaler_devri_say};
use crate::dsp::mc_tables::OBMC_MASKS;
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m256i, _mm_set1_epi16, _mm_setzero_si128, _mm_srli_si128, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_packus_epi32, _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setr_epi64x, _mm256_srai_epi32, _mm256_sub_epi16, _mm256_unpackhi_epi16,
    _mm256_unpacklo_epi16,
};

/// 16 pikselin karışımı `(d·(64−m) + t·m + 32) >> 6`: `d`, `t` sıralı `u16`,
/// `m` sıralı `i16` maske.
#[target_feature(enable = "avx2")]
#[inline]
fn harmanla(d: __m256i, t: __m256i, m: __m256i) -> __m256i {
    let n = _mm256_sub_epi16(_mm256_set1_epi16(64), m);
    let k = _mm256_set1_epi32(32);
    let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(d, t), _mm256_unpacklo_epi16(n, m));
    let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(d, t), _mm256_unpackhi_epi16(n, m));
    let lo = _mm256_srai_epi32::<6>(_mm256_add_epi32(lo, k));
    let hi = _mm256_srai_epi32::<6>(_mm256_add_epi32(hi, k));
    _mm256_packus_epi32(lo, hi)
}

/// 16 maske baytı `s[o..o+16]` → sıralı `i16`.
#[target_feature(enable = "avx2")]
#[inline]
fn maske16(s: &[u8], o: usize) -> __m256i {
    _mm256_cvtepu8_epi16(load_u8x16(chunk(s, o)))
}

/// `W` (4, 8) sütunlu `satir ≤ 16/W` satırı (`y`'den) bir ymm'e toplar;
/// eksik satırlar 0 (okunmaz).
#[target_feature(enable = "avx2")]
#[inline]
fn topla<const W: usize>(d: &[u8], ds: usize, y: usize, satir: usize) -> __m256i {
    if W == 4 {
        let mut r = [0i64; 4];
        for (i, v) in r.iter_mut().enumerate().take(satir) {
            *v = i64::from_le_bytes(*chunk(d, (y + i) * ds));
        }
        _mm256_setr_epi64x(r[0], r[1], r[2], r[3])
    } else {
        let r0 = load_u8x16(chunk(d, y * ds));
        let r1 = if satir > 1 {
            load_u8x16(chunk(d, (y + 1) * ds))
        } else {
            _mm_setzero_si128()
        };
        _mm256_set_m128i(r1, r0)
    }
}

/// [`topla`]'nın tersi: yalnız ilk `satir` satırı yazar.
#[target_feature(enable = "avx2")]
#[inline]
fn dagit<const W: usize>(d: &mut [u8], ds: usize, y: usize, satir: usize, v: __m256i) {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    if W == 4 {
        store_u8x8(chunk_mut(d, y * ds), lo);
        if satir > 1 {
            store_u8x8(chunk_mut(d, (y + 1) * ds), _mm_srli_si128::<8>(lo));
        }
        if satir > 2 {
            store_u8x8(chunk_mut(d, (y + 2) * ds), hi);
        }
        if satir > 3 {
            store_u8x8(chunk_mut(d, (y + 3) * ds), _mm_srli_si128::<8>(hi));
        }
    } else {
        store_u8x16(chunk_mut(d, y * ds), lo);
        if satir > 1 {
            store_u8x16(chunk_mut(d, (y + 1) * ds), hi);
        }
    }
}

/// `s`'nin `o`'dan `n ≤ 32` baytı; dilimde 32 bayt varsa doğrudan yüklenir
/// (fazla baytlar `tmp`'nin kendi bölgesindedir ve yazılmayan satırlara gider),
/// yoksa eksik baytlar 0 (dilim dışı okunmaz).
#[target_feature(enable = "avx2")]
#[inline]
fn bitisik32(s: &[u8], o: usize, n: usize) -> __m256i {
    if n == 32 || o + 32 <= s.len() {
        load_u8x32(chunk(s, o))
    } else {
        let mut b = [0u8; 32];
        b[..n].copy_from_slice(&s[o..o + n]);
        load_u8x32(&b)
    }
}

/// `blend` gövdesi, `W` sütun: `W ≥ 16` satır başına 16'lık parçalar, `W ≤ 8`
/// ymm başına `16/W` satır (`h` 2'nin kuvveti ≥ 4 olduğundan tam gruplar).
#[target_feature(enable = "avx2")]
fn blend_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize, mask: &[u8]) {
    if W >= 16 {
        for y in 0..h {
            for x in (0..W).step_by(16) {
                let (o, od) = (y * W + x, y * ds + 2 * x);
                let d = load_u8x32(chunk(dst, od));
                let r = harmanla(d, load_u8x32(chunk(tmp, 2 * o)), maske16(mask, o));
                store_u8x32(chunk_mut(dst, od), r);
            }
        }
        return;
    }
    let rpx = 16 / W;
    for y in (0..h).step_by(rpx) {
        let o = y * W;
        let r = harmanla(
            topla::<W>(dst, ds, y, rpx),
            load_u8x32(chunk(tmp, 2 * o)),
            maske16(mask, o),
        );
        dagit::<W>(dst, ds, y, rpx, r);
    }
}

/// Tek piksel karışımı `(d·(64−m) + t·m + 32) >> 6` (`w = 2`).
#[inline(always)]
fn px(dst: &mut [u8], od: usize, tmp: &[u8], ot: usize, m: i32) {
    let d = i32::from(u16::from_ne_bytes(*chunk(dst, od)));
    let t = i32::from(u16::from_ne_bytes(*chunk(tmp, ot)));
    let v = ((d * (64 - m) + t * m + 32) >> 6) as u16;
    *chunk_mut(dst, od) = v.to_ne_bytes();
}

/// `blend_v`, `w = 2`: yalnız sütun 0 karışır (`(2·3) >> 2 = 1`; sütun 1'in
/// maskesi 0). Girişte doğrudan tamsayı formülü: 2 piksellik satırda vektör
/// kurulumu ve `#[target_feature]` çağrısı işten pahalı.
// P36: P35'te tek çağırıcılı olduğu için girişe satır içi alınıyordu; AVX512ICL
// girişi de çağırınca aynı kod yapısı için zorunlu.
#[inline(always)]
fn blend_v_iki(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let m = i32::from(OBMC_MASKS[2]);
    for y in 0..h {
        px(dst, y * ds, tmp, y * 4, m);
    }
}

/// `blend_h`, `w = 2`: ilk `(h·3) >> 2` satır, satır maskesi `OBMC_MASKS[h + y]`
/// (girişte doğrudan tamsayı formülü; bkz. [`blend_v_iki`]).
// P36: P35'te tek çağırıcılı olduğu için girişe satır içi alınıyordu; AVX512ICL
// girişi de çağırınca aynı kod yapısı için zorunlu.
#[inline(always)]
fn blend_h_iki(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    for (y, &m) in OBMC_MASKS[h..].iter().enumerate().take((h * 3) >> 2) {
        // Satırın iki pikseli tek 4 baytlık yükleme/saklamayla.
        let m = u32::from(m);
        let d = u32::from_ne_bytes(*chunk(dst, y * ds));
        let t = u32::from_ne_bytes(*chunk(tmp, y * 4));
        let k = |d: u32, t: u32| (d * (64 - m) + t * m + 32) >> 6;
        let v0 = k(d & 0xffff, t & 0xffff);
        let v1 = k(d >> 16, t >> 16);
        *chunk_mut(dst, y * ds) = (v0 | (v1 << 16)).to_ne_bytes();
    }
}

/// `blend_v` gövdesi: sütun maskesi `OBMC_MASKS[W + x]`, satırlara yayılı.
#[target_feature(enable = "avx2")]
fn blend_v_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let tablo = &BLEND_V_MASKE[W.trailing_zeros() as usize - 2];
    let m0 = maske16(tablo, 0);
    if W >= 16 {
        let m1 = maske16(tablo, 16);
        for y in 0..h {
            for x in (0..W).step_by(16) {
                let od = y * ds + 2 * x;
                let m = if x == 0 { m0 } else { m1 };
                let d = load_u8x32(chunk(dst, od));
                let r = harmanla(d, load_u8x32(chunk(tmp, 2 * (y * W + x))), m);
                store_u8x32(chunk_mut(dst, od), r);
            }
        }
        return;
    }
    let rpx = 16 / W;
    let mut y = 0;
    while y < h {
        // `W = 4`, `h = 2`'de tek kısmi grup.
        let satir = rpx.min(h - y);
        let t = bitisik32(tmp, 2 * y * W, 2 * satir * W);
        let r = harmanla(topla::<W>(dst, ds, y, satir), t, m0);
        dagit::<W>(dst, ds, y, satir, r);
        y += satir;
    }
}

/// `blend_h` gövdesi: satır maskesi `OBMC_MASKS[h + y]`, ilk `(h·3) >> 2` satır.
#[target_feature(enable = "avx2")]
fn blend_h_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let satirlar = (h * 3) >> 2;
    let tablo = &OBMC_MASKS[h..];
    if W >= 16 {
        for (y, &my) in tablo.iter().enumerate().take(satirlar) {
            let m = _mm256_set1_epi16(i16::from(my));
            for x in (0..W).step_by(16) {
                let od = y * ds + 2 * x;
                let d = load_u8x32(chunk(dst, od));
                let r = harmanla(d, load_u8x32(chunk(tmp, 2 * (y * W + x))), m);
                store_u8x32(chunk_mut(dst, od), r);
            }
        }
        return;
    }
    let rpx = 16 / W;
    let mut y = 0;
    while y < satirlar {
        let satir = rpx.min(satirlar - y);
        // Satır maskesi yayını: satır `r` için `W` öğe `tablo[y + r]` (eksik satır 0).
        let mut m = [0u16; 4];
        for (r, v) in m.iter_mut().enumerate().take(satir) {
            *v = u16::from(tablo[y + r]);
        }
        let mb = if W == 4 {
            let k = |r: usize| (u64::from(m[r]) * 0x0001_0001_0001_0001) as i64;
            _mm256_setr_epi64x(k(0), k(1), k(2), k(3))
        } else {
            _mm256_set_m128i(_mm_set1_epi16(m[1] as i16), _mm_set1_epi16(m[0] as i16))
        };
        let t = bitisik32(tmp, 2 * y * W, 2 * satir * W);
        let r = harmanla(topla::<W>(dst, ds, y, satir), t, mb);
        dagit::<W>(dst, ds, y, satir, r);
        y += satir;
    }
}

/// `blend_v` sütun maskesi `OBMC_MASKS[W + x]`, 32 bayta yayılı (`W` = 4, 8, 16, 32; 8 bpc
/// `mc_karisim` tablosuyla aynı, kendi kopyası).
static BLEND_V_MASKE: [[u8; 32]; 4] = [
    blend_v_maske(4),
    blend_v_maske(8),
    blend_v_maske(16),
    blend_v_maske(32),
];

const fn blend_v_maske(w: usize) -> [u8; 32] {
    let mut b = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        b[i] = OBMC_MASKS[w + i % w];
        i += 1;
    }
    b
}

macro_rules! skaler {
    ($bd:expr, $f:ident ( $($a:expr),* )) => {
        match $bd {
            BitDepth::Eight => scalar::$f::<8>($($a),*),
            BitDepth::Ten => scalar::$f::<10>($($a),*),
            BitDepth::Twelve => scalar::$f::<12>($($a),*),
        }
    };
}

fn blend16_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    if mask.len() < w * h
        || w * h < 16
        || !karisim16_sozlesmesi(
            w,
            (4, 32),
            h,
            (4, 32),
            h,
            tmp.len(),
            dst.len(),
            ds,
            bd,
            bdmax,
        )
    {
        skaler_devri_say();
        return skaler!(bd, blend(dst, ds, tmp, w, h, mask, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`blend_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve hedefte `h` satır × `2w`
    // baytı, `mask.len() ≥ w·h` maskeyi kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, girdiler `&`; ödünç kuralları çakışmayı dışlar.
    unsafe {
        match w {
            4 => blend_govde::<4>(dst, ds, tmp, h, mask),
            8 => blend_govde::<8>(dst, ds, tmp, h, mask),
            16 => blend_govde::<16>(dst, ds, tmp, h, mask),
            _ => blend_govde::<32>(dst, ds, tmp, h, mask),
        }
    }
}

fn blend_v16_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    // `w = 2`: girişte doğrudan tamsayı formülü (güvenli kod; skalerle aynı sırada
    // aynı erişimler, sözleşme denetimine gerek yok — yalnız bit derinliği).
    if w == 2 && bd16_tutarli(bd, bdmax) {
        return blend_v_iki(dst, ds, tmp, h);
    }
    if !karisim16_sozlesmesi(
        w,
        (2, 32),
        h,
        (2, 128),
        h,
        tmp.len(),
        dst.len(),
        ds,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return skaler!(bd, blend_v(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`blend_v_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve hedefte `h` satır × `2w`
    // baytı kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek
    // `&mut`, `tmp` `&`.
    unsafe {
        match w {
            4 => blend_v_govde::<4>(dst, ds, tmp, h),
            8 => blend_v_govde::<8>(dst, ds, tmp, h),
            16 => blend_v_govde::<16>(dst, ds, tmp, h),
            _ => blend_v_govde::<32>(dst, ds, tmp, h),
        }
    }
}

fn blend_h16_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    // `w = 2`: girişte doğrudan tamsayı formülü (güvenli kod; skalerle aynı sırada
    // aynı erişimler, sözleşme denetimine gerek yok — yalnız bit derinliği).
    if w == 2 && bd16_tutarli(bd, bdmax) {
        return blend_h_iki(dst, ds, tmp, h);
    }
    if !karisim16_sozlesmesi(
        w,
        (2, 128),
        h,
        (2, 32),
        (h * 3) >> 2,
        tmp.len(),
        dst.len(),
        ds,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return skaler!(bd, blend_h(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`blend_h_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve hedefte yazılan
    // `(h·3) >> 2` satır × `2w` baytı kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `tmp` `&`.
    unsafe {
        match w {
            4 => blend_h_govde::<4>(dst, ds, tmp, h),
            8 => blend_h_govde::<8>(dst, ds, tmp, h),
            16 => blend_h_govde::<16>(dst, ds, tmp, h),
            32 => blend_h_govde::<32>(dst, ds, tmp, h),
            64 => blend_h_govde::<64>(dst, ds, tmp, h),
            _ => blend_h_govde::<128>(dst, ds, tmp, h),
        }
    }
}

/// 16 bpc `blend` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_fn(_avx2: Avx2Token) -> BlendFn {
    blend16_giris
}

/// 16 bpc `blend_v` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_v_fn(_avx2: Avx2Token) -> BlendDirFn {
    blend_v16_giris
}

/// 16 bpc `blend_h` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_h_fn(_avx2: Avx2Token) -> BlendDirFn {
    blend_h16_giris
}

// ---------------------------------------------------------------------------
// P36: AVX512ICL girişleri (adres yalnız [`super::mc16_karisim_avx512`] belirteçli işaretçileriyle)
// ---------------------------------------------------------------------------
//
// Girişler bu modüldedir: dar bloklar bu modülün AVX2 gövdelerini, geniş bloklar
// `mc16_karisim_avx512`'in gövdelerini çağırır. Paylaşılan AVX2 gövdelerinin bütün çağırıcıları
// aynı kod üretim biriminde kalır ve iç bağlantıları korunur (başka birimden
// çağrılan tek kopya dış bağlantıya geçip LLVM'nin yerel işlev iyileştirmelerini
// yitiriyordu: AVX2 hücrelerinde +%10–20, P36 hız bütçesi tanı koşusu).

/// `blend` girişi: AVX2 girişiyle aynı sözleşme denetimi (aksi hâlde skaler
/// devir); `w < 16` AVX2 gövdeleri, aksi hâlde AVX512ICL gövdeleri. Tek giriş, tek
/// `#[target_feature]` çağrısı (P36 hücre ölçümü: AVX2 girişine devretmek çağrı
/// başına sabit maliyet ekliyordu; `w = 8`'de dört satırı zmm'e toplamak AVX2
/// yolundan yavaştı).
pub(super) fn blend16_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    if mask.len() < w * h
        || w * h < 16
        || !karisim16_sozlesmesi(
            w,
            (4, 32),
            h,
            (4, 32),
            h,
            tmp.len(),
            dst.len(),
            ds,
            bd,
            bdmax,
        )
    {
        skaler_devri_say();
        return skaler!(bd, blend(dst, ds, tmp, w, h, mask, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc16_karisim_avx512::blend_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespiti + maske izninin
    // kanıtıdır. (2) sınır: `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve
    // hedefte `h` satır × `2w` baytı, `mask.len() ≥ w·h` maskeyi kapsar; `h` 2'nin
    // kuvveti ≥ 4 olduğundan satır çiftleri tamdır; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe {
        match w {
            4 => blend_govde::<4>(dst, ds, tmp, h, mask),
            8 => blend_govde::<8>(dst, ds, tmp, h, mask),
            16 => super::mc16_karisim_avx512::blend_govde::<16>(dst, ds, tmp, h, mask),
            _ => super::mc16_karisim_avx512::blend_govde::<32>(dst, ds, tmp, h, mask),
        }
    }
}

/// `blend_v` girişi: [`blend16_giris`] gibi; `w = 2` AVX2 girişinin tamsayı yolu.
pub(super) fn blend_v16_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if w == 2 && bd16_tutarli(bd, bdmax) {
        return blend_v_iki(dst, ds, tmp, h);
    }
    if !karisim16_sozlesmesi(
        w,
        (2, 32),
        h,
        (2, 128),
        h,
        tmp.len(),
        dst.len(),
        ds,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return skaler!(bd, blend_v(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc16_karisim_avx512::blend_v_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır.
    // (2) sınır: `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve hedefte `h`
    // satır × `2w` baytı kapsar; `h` 2'nin kuvveti olduğundan satır çiftleri
    // tamdır; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`,
    // `tmp` `&`.
    unsafe {
        match w {
            4 => blend_v_govde::<4>(dst, ds, tmp, h),
            8 => blend_v_govde::<8>(dst, ds, tmp, h),
            16 => super::mc16_karisim_avx512::blend_v_govde::<16>(dst, ds, tmp, h),
            _ => super::mc16_karisim_avx512::blend_v_govde::<32>(dst, ds, tmp, h),
        }
    }
}

/// `blend_h` girişi: [`blend16_giris`] gibi; `w = 2` AVX2 girişinin tamsayı yolu.
pub(super) fn blend_h16_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if w == 2 && bd16_tutarli(bd, bdmax) {
        return blend_h_iki(dst, ds, tmp, h);
    }
    if !karisim16_sozlesmesi(
        w,
        (2, 128),
        h,
        (2, 32),
        (h * 3) >> 2,
        tmp.len(),
        dst.len(),
        ds,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return skaler!(bd, blend_h(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc16_karisim_avx512::blend_h_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır.
    // (2) sınır: `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve hedefte
    // yazılan `(h·3) >> 2` satır × `2w` baytı kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `tmp` `&`.
    unsafe {
        match w {
            4 => blend_h_govde::<4>(dst, ds, tmp, h),
            8 => blend_h_govde::<8>(dst, ds, tmp, h),
            16 => super::mc16_karisim_avx512::blend_h_govde::<16>(dst, ds, tmp, h),
            32 => super::mc16_karisim_avx512::blend_h_govde::<32>(dst, ds, tmp, h),
            64 => super::mc16_karisim_avx512::blend_h_govde::<64>(dst, ds, tmp, h),
            _ => super::mc16_karisim_avx512::blend_h_govde::<128>(dst, ds, tmp, h),
        }
    }
}
