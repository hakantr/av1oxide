//! MC 16 bpc (10/12 bit) SSSE3 `put`/`prep` çekirdekleri — F4-MC d (P35):
//! `mc[0..9]`, `mct[0..9]` (9 adet 8-tap birleşimi + bilinear).
//!
//! ## Sözleşme
//!
//! AVX2 gövdesiyle ([`super::mc16`]) aynı: girişte O(1)
//! [`crate::dsp::mc::simd16_sozlesmesi`] (10/12 bit, tutarlı `bitdepth_max`,
//! kaynak `(h+6)·ss + (w+7)·2 + K` bayt); sözleşme dışı girdi skalere devreder
//! ve sayılır. **Okuma bölgesi** satır `0..=h+6`, sütun `0..=w+6`; `w ≥ 16`
//! yolunun 8 piksellik yüklemeleri bölgenin içinde biter, `w ≤ 8` yolu 8 çıktı
//! hesaplar ve satır bölgesini en çok 6 piksel (12 bayt ≤ `K`) aşar.
//!
//! ## Yöntem
//!
//! AVX2 gövdesinin xmm yolu (tap ikilileri `pmaddwd`, `i32` toplam, çift/tek
//! çıktılar `punpck{l,h}dq`), `w ≥ 16` 8 piksellik parçalarla; iki boyutlu yolda
//! 8 satırlık ara halka xmm dizisinde. SSE4.1 komutu yok: `put` kırpması
//! `packssdw` + `pmaxsw(0)` + `pminsw(bitdepth_max)` (pikseller ≤ 4095, `i16`'ya
//! sığar; doyurmalı paketleme taşmayı sınıra çeker). Yuvarlama dav1d
//! `mc_tmpl.c` (AVX2 gövdesiyle aynı; `ib` 10 bitte 4, 12 bitte 2).
//!
//! `unsafe` yalnız belirteçli girişlerdeki `#[target_feature]` çağrısıdır;
//! dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc::{BILIN, prep_skaler, put_skaler};
use super::mem::{chunk, chunk_mut, load_u8x16, store_i16x8, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{McPrepFn, McPutFn, simd16_sozlesmesi, skaler_devri_say};
use crate::dsp::mc_tables::MC_SUBPEL_FILTERS;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_cvtsi128_si64,
    _mm_madd_epi16, _mm_max_epi16, _mm_min_epi16, _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setzero_si128, _mm_sll_epi16, _mm_sra_epi32, _mm_sub_epi16, _mm_sub_epi32,
    _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Hedef: `put` için bayt düzlemi (adım bayt), `prep` için `tmp` (adım `w`).
struct Hedef<'a> {
    dst: &'a mut [u8],
    ds: usize,
    tmp: &'a mut [i16],
}

/// Bir geçişin yuvarlaması: `(Σ + r) >> sh`.
#[derive(Clone, Copy)]
struct Yuv {
    r: i32,
    sh: i32,
}

impl Yuv {
    const fn yeni(sh: i32) -> Self {
        Yuv {
            r: (1 << sh) >> 1,
            sh,
        }
    }
}

/// Çıktıya özgü değerler: `bias` (`prep`), `max` (`put` kırpma sınırı).
#[derive(Clone, Copy)]
struct Cikti {
    bias: i32,
    max: i32,
}

/// Filtre satırı (dav1d `GET_H_FILTER`/`GET_V_FILTER`).
fn filtre_satiri(t: usize, boyut: usize, m: u8) -> &'static [i8; 8] {
    let tablo = if boyut > 4 { t } else { 3 + (t & 1) };
    &MC_SUBPEL_FILTERS[tablo][usize::from(m) - 1]
}

/// Tap ikilisi katsayıları `(f[S+2i], f[S+2i+1])` → `pmaddwd` sözcükleri.
fn ikililer<const P: usize, const S: usize>(f: &[i8; 8]) -> [i32; 4] {
    let mut k = [0i32; 4];
    for (i, ki) in k.iter_mut().enumerate().take(P) {
        let a = i32::from(f[S + 2 * i]);
        let b = i32::from(f[S + 2 * i + 1]);
        *ki = (a & 0xffff) | (b << 16);
    }
    k
}

/// Filtrenin sıfır olmayan taplarına göre tap ikilisi sayısı `P`: 8 tap → 4
/// (`S = 0`), dış taplar 0 → 3 (`S = 1`, 6 tap), ayrıca 1. ve 6. taplar 0 → 2
/// (`S = 2`, 4 tap). Sıfır tapın katkısı 0 olduğundan sonuç 8 tapla aynıdır
/// (8 bpc gövdesinin `tap` seçimi gibi; dav1d'nin 6 tap yolu).
fn tap_p(f: &[i8; 8]) -> usize {
    if f[0] != 0 || f[7] != 0 {
        4
    } else if f[1] != 0 || f[6] != 0 {
        3
    } else {
        2
    }
}

/// Bilinear katsayı ikilisi `(16−m, m)`.
const fn bilin_ikili(m: u8) -> [i32; 4] {
    let m = m as i32;
    [((16 - m) & 0xffff) | (m << 16), 0, 0, 0]
}

// ---------------------------------------------------------------------------
// Giriş
// ---------------------------------------------------------------------------

fn put16_giris<const FH: usize, const FV: usize>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let ok = dst_stride >= 2 * w
        && h >= 1
        && simd16_sozlesmesi(
            src.len(),
            src_stride,
            w,
            h,
            mx,
            my,
            bit_depth,
            bitdepth_max,
            dst.len(),
            (h - 1) * dst_stride + 2 * w,
        );
    if !ok {
        skaler_devri_say();
        put_skaler::<FH, FV>(
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
        return;
    }
    let mut hd = Hedef {
        dst,
        ds: dst_stride,
        tmp: &mut [],
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`put16_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `simd16_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve hedefte `h` satır ×
    // `2w` baytı kapsar; gövde erişimleri dilim/dizi denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `src` `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { govde::<FH, FV, false>(&mut hd, src, src_stride, w, h, mx, my, bitdepth_max) }
}

fn prep16_giris<const FH: usize, const FV: usize>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let ok = simd16_sozlesmesi(
        src.len(),
        src_stride,
        w,
        h,
        mx,
        my,
        bit_depth,
        bitdepth_max,
        tmp.len(),
        w * h,
    );
    if !ok {
        skaler_devri_say();
        prep_skaler::<FH, FV>(tmp, src, src_stride, w, h, mx, my, bit_depth, bitdepth_max);
        return;
    }
    let mut hd = Hedef {
        dst: &mut [],
        ds: 0,
        tmp,
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`prep16_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `simd16_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'de `w·h` öğeyi
    // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`,
    // `src` `&`.
    unsafe { govde::<FH, FV, true>(&mut hd, src, src_stride, w, h, mx, my, bitdepth_max) }
}

/// Yol seçimi: kopya / yalnız yatay / yalnız dikey / iki boyutlu.
#[target_feature(enable = "ssse3")]
fn govde<const FH: usize, const FV: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bitdepth_max: u16,
) {
    let max = i32::from(bitdepth_max);
    let ib = if max == 4095 { 2 } else { 4 };
    let cikti = Cikti {
        bias: PREP_BIAS,
        max,
    };
    let bil = FH == BILIN;
    let sh_ara = if bil { 4 - ib } else { 6 - ib };
    let tam = if bil { 4 } else { 6 };
    match (mx != 0, my != 0) {
        (false, false) => kopya::<PREP>(hd, src, ss, w, h, ib),
        (true, false) => {
            let yuv = if PREP {
                Yuv::yeni(sh_ara)
            } else {
                let ara = Yuv::yeni(sh_ara);
                Yuv {
                    r: ara.r + (((1 << ib) >> 1) << sh_ara),
                    sh: tam,
                }
            };
            if bil {
                yol_h::<PREP, 1, 3>(hd, src, ss, w, h, bilin_ikili(mx), yuv, cikti);
            } else {
                let f = filtre_satiri(FH, w, mx);
                match tap_p(f) {
                    2 => yol_h::<PREP, 2, 2>(hd, src, ss, w, h, ikililer::<2, 2>(f), yuv, cikti),
                    3 => yol_h::<PREP, 3, 1>(hd, src, ss, w, h, ikililer::<3, 1>(f), yuv, cikti),
                    _ => yol_h::<PREP, 4, 0>(hd, src, ss, w, h, ikililer::<4, 0>(f), yuv, cikti),
                }
            }
        }
        (false, true) => {
            let yuv = if PREP {
                Yuv::yeni(sh_ara)
            } else {
                Yuv::yeni(tam)
            };
            if bil {
                yol_v::<PREP, 1, 3>(hd, src, ss, w, h, bilin_ikili(my), yuv, cikti);
            } else {
                let f = filtre_satiri(FV, h, my);
                match tap_p(f) {
                    2 => yol_v::<PREP, 2, 2>(hd, src, ss, w, h, ikililer::<2, 2>(f), yuv, cikti),
                    3 => yol_v::<PREP, 3, 1>(hd, src, ss, w, h, ikililer::<3, 1>(f), yuv, cikti),
                    _ => yol_v::<PREP, 4, 0>(hd, src, ss, w, h, ikililer::<4, 0>(f), yuv, cikti),
                }
            }
        }
        (true, true) => {
            let ara = Yuv::yeni(sh_ara);
            let son = if PREP {
                Yuv::yeni(tam)
            } else {
                Yuv::yeni(tam + ib)
            };
            // Yatay ve dikey aynı tap sınıfıyla (ikisinin büyüğü); bilinearde filtre
            // tablosu kullanılmaz.
            const SIFIR: [i8; 8] = [0; 8];
            let (fh, fv) = if bil {
                (&SIFIR, &SIFIR)
            } else {
                (filtre_satiri(FH, w, mx), filtre_satiri(FV, h, my))
            };
            macro_rules! hv {
                ($p:literal, $s:literal) => {{
                    let (kh, kv) = if bil {
                        (bilin_ikili(mx), bilin_ikili(my))
                    } else {
                        (ikililer::<$p, $s>(fh), ikililer::<$p, $s>(fv))
                    };
                    match w {
                        ..=8 => {
                            hv_yol::<PREP, $p, $s, 1>(hd, src, ss, w, h, kh, kv, ara, son, cikti)
                        }
                        16 => hv_yol::<PREP, $p, $s, 2>(hd, src, ss, w, h, kh, kv, ara, son, cikti),
                        32 => hv_yol::<PREP, $p, $s, 4>(hd, src, ss, w, h, kh, kv, ara, son, cikti),
                        64 => hv_yol::<PREP, $p, $s, 8>(hd, src, ss, w, h, kh, kv, ara, son, cikti),
                        _ => hv_yol::<PREP, $p, $s, 16>(hd, src, ss, w, h, kh, kv, ara, son, cikti),
                    }
                }};
            }
            if bil {
                hv!(1, 3)
            } else {
                match tap_p(fh).max(tap_p(fv)) {
                    2 => hv!(2, 2),
                    3 => hv!(3, 1),
                    _ => hv!(4, 0),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ortak yardımcılar
// ---------------------------------------------------------------------------

/// 8 çıktının yatay tap toplamları (`lo` çıktı 0..3, `hi` 4..7); `o` dolgu
/// satırı + sütun `x`'in bayt ofseti.
#[target_feature(enable = "ssse3")]
#[inline]
fn yatay8<const P: usize, const S: usize>(
    src: &[u8],
    o: usize,
    k: &[__m128i; 4],
) -> (__m128i, __m128i) {
    // Okunan pencere `2·(2P − 1) + 16 = 4P + 14` bayt (8 tap: 30 bayt, değişmedi);
    // bir kez ayrılır, yüklemeler derleme zamanı ofsetiyle (tek sınır denetimi).
    match P {
        4 => yatay8_n::<P, S, 30>(src, o, k),
        3 => yatay8_n::<P, S, 26>(src, o, k),
        2 => yatay8_n::<P, S, 22>(src, o, k),
        _ => yatay8_n::<P, S, 18>(src, o, k),
    }
}

/// [`yatay8`]'in pencere boyu `N = 4P + 14` monomorfu.
#[target_feature(enable = "ssse3")]
#[inline]
fn yatay8_n<const P: usize, const S: usize, const N: usize>(
    src: &[u8],
    o: usize,
    k: &[__m128i; 4],
) -> (__m128i, __m128i) {
    let p: &[u8; N] = chunk(src, o + 2 * S);
    let mut e = _mm_setzero_si128();
    let mut t = _mm_setzero_si128();
    for (i, ki) in k.iter().enumerate().take(P) {
        let a = load_u8x16(chunk(p, 4 * i));
        let b = load_u8x16(chunk(p, 4 * i + 2));
        e = _mm_add_epi32(e, _mm_madd_epi16(a, *ki));
        t = _mm_add_epi32(t, _mm_madd_epi16(b, *ki));
    }
    (_mm_unpacklo_epi32(e, t), _mm_unpackhi_epi32(e, t))
}

/// `(v + r) >> sh`.
#[target_feature(enable = "ssse3")]
#[inline]
fn yuv(v: __m128i, r: __m128i, sh: __m128i) -> __m128i {
    _mm_sra_epi32(_mm_add_epi32(v, r), sh)
}

/// `put` kırpması `[0, max]`: `packssdw` + `pmaxsw(0)` + `pminsw(max)`.
#[target_feature(enable = "ssse3")]
#[inline]
fn kirp(lo: __m128i, hi: __m128i, max: __m128i) -> __m128i {
    _mm_min_epi16(
        _mm_max_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128()),
        max,
    )
}

/// 8 çıktıdan `w` (2/4/8) tanesini satır `y` sütun `x`'e yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn yaz8<const PREP: bool>(
    hd: &mut Hedef<'_>,
    w: usize,
    y: usize,
    x: usize,
    lo: __m128i,
    hi: __m128i,
    c: (__m128i, __m128i),
) {
    let n = w.min(8);
    if PREP {
        let v = _mm_packs_epi32(_mm_sub_epi32(lo, c.0), _mm_sub_epi32(hi, c.0));
        let o = y * w + x;
        match n {
            8 => store_i16x8(chunk_mut(hd.tmp, o), v),
            4 => {
                let b = _mm_cvtsi128_si64(v).to_le_bytes();
                for (k, t) in hd.tmp[o..o + 4].iter_mut().enumerate() {
                    *t = i16::from_le_bytes([b[2 * k], b[2 * k + 1]]);
                }
            }
            _ => {
                let b = _mm_cvtsi128_si32(v).to_le_bytes();
                hd.tmp[o] = i16::from_le_bytes([b[0], b[1]]);
                hd.tmp[o + 1] = i16::from_le_bytes([b[2], b[3]]);
            }
        }
    } else {
        let v = kirp(lo, hi, c.1);
        let o = y * hd.ds + 2 * x;
        match n {
            8 => store_u8x16(chunk_mut(hd.dst, o), v),
            4 => store_u8x8(chunk_mut(hd.dst, o), v),
            _ => store_u8x4(chunk_mut(hd.dst, o), v),
        }
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn kat(k: [i32; 4]) -> [__m128i; 4] {
    [
        _mm_set1_epi32(k[0]),
        _mm_set1_epi32(k[1]),
        _mm_set1_epi32(k[2]),
        _mm_set1_epi32(k[3]),
    ]
}

// ---------------------------------------------------------------------------
// Yollar
// ---------------------------------------------------------------------------

/// Kopya: `put` satır kopyası, `prep` `(p << ib) − 8192`.
#[target_feature(enable = "ssse3")]
fn kopya<const PREP: bool>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize, ib: i32) {
    let ibv = _mm_cvtsi32_si128(ib);
    let b8 = _mm_set1_epi16(PREP_BIAS as i16);
    for y in 0..h {
        let s0 = (y + 3) * ss + 6;
        if PREP {
            let o = y * w;
            if w >= 8 {
                // Satır dilimleri bir kez ayrılır; 8 piksellik parçalarda sınır denetimi yok.
                let sr = &src[s0..s0 + 2 * w];
                let tr = &mut hd.tmp[o..o + w];
                for (sp, tp) in sr.as_chunks::<16>().0.iter().zip(tr.as_chunks_mut::<8>().0) {
                    store_i16x8(tp, _mm_sub_epi16(_mm_sll_epi16(load_u8x16(sp), ibv), b8));
                }
            } else {
                let v = _mm_sub_epi16(_mm_sll_epi16(load_u8x16(chunk(src, s0)), ibv), b8);
                if w == 4 {
                    let b = _mm_cvtsi128_si64(v).to_le_bytes();
                    for (k, t) in hd.tmp[o..o + 4].iter_mut().enumerate() {
                        *t = i16::from_le_bytes([b[2 * k], b[2 * k + 1]]);
                    }
                } else {
                    let b = _mm_cvtsi128_si32(v).to_le_bytes();
                    hd.tmp[o] = i16::from_le_bytes([b[0], b[1]]);
                    hd.tmp[o + 1] = i16::from_le_bytes([b[2], b[3]]);
                }
            }
        } else {
            let d0 = y * hd.ds;
            match w {
                2 => *chunk_mut::<u8, 4>(hd.dst, d0) = *chunk(src, s0),
                4 => *chunk_mut::<u8, 8>(hd.dst, d0) = *chunk(src, s0),
                8 => kopya_satir::<16>(hd.dst, d0, src, s0),
                16 => kopya_satir::<32>(hd.dst, d0, src, s0),
                32 => kopya_satir::<64>(hd.dst, d0, src, s0),
                64 => kopya_satir::<128>(hd.dst, d0, src, s0),
                _ => kopya_satir::<256>(hd.dst, d0, src, s0),
            }
        }
    }
}

/// `put` kopyasının bir satırı: `B` bayt (16'nın katı), satır başına tek sınır
/// denetimi, içte sabit ofsetli 16 baytlık parçalar.
#[target_feature(enable = "ssse3")]
#[inline]
fn kopya_satir<const B: usize>(dst: &mut [u8], d0: usize, src: &[u8], s0: usize) {
    let d: &mut [u8; B] = chunk_mut(dst, d0);
    let s: &[u8; B] = chunk(src, s0);
    let mut i = 0;
    while i < B {
        store_u8x16(chunk_mut(d, i), load_u8x16(chunk(s, i)));
        i += 16;
    }
}

/// Yalnız yatay: dolgu satırı `y + 3`, 8 çıktılık parçalar.
#[target_feature(enable = "ssse3")]
fn yol_h<const PREP: bool, const P: usize, const S: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    k: [i32; 4],
    y_: Yuv,
    c: Cikti,
) {
    let sh = _mm_cvtsi32_si128(y_.sh);
    let k = kat(k);
    let r = _mm_set1_epi32(y_.r);
    let cv = (_mm_set1_epi32(c.bias), _mm_set1_epi16(c.max as i16));
    for y in 0..h {
        let o = (y + 3) * ss;
        let mut x = 0;
        loop {
            let (lo, hi) = yatay8::<P, S>(src, o + 2 * x, &k);
            yaz8::<PREP>(hd, w, y, x, yuv(lo, r, sh), yuv(hi, r, sh), cv);
            x += 8;
            if x >= w {
                break;
            }
        }
    }
}

/// Yalnız dikey: 8 sütunluk şeritler; şerit içinde ardışık satır ikililerinin
/// (`punpck{l,h}wd`) kayan penceresi — çıktı satırı `y` ikilileri `(y + S + 2i,
/// +1)`, bir sonraki çıktı bir kaydırılmışını kullanır; satır başına tek yükleme,
/// tek sınır denetimi ve tek ikili.
#[target_feature(enable = "ssse3")]
fn yol_v<const PREP: bool, const P: usize, const S: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    k: [i32; 4],
    y_: Yuv,
    c: Cikti,
) {
    let sh = _mm_cvtsi32_si128(y_.sh);
    let k = kat(k);
    let r = _mm_set1_epi32(y_.r);
    let cv = (_mm_set1_epi32(c.bias), _mm_set1_epi16(c.max as i16));
    let mut x = 0;
    loop {
        let o = S * ss + 2 * (x + 3);
        // `ik[t]` = satır `t` ile `t + 1`'in ikilisi (alt, üst yarı).
        let z = _mm_setzero_si128();
        let mut ik = [(z, z); 7];
        let mut onceki = load_u8x16(chunk(src, o));
        for (t, it) in ik.iter_mut().enumerate().take(2 * P - 2) {
            let r = load_u8x16(chunk(src, o + (t + 1) * ss));
            *it = (_mm_unpacklo_epi16(onceki, r), _mm_unpackhi_epi16(onceki, r));
            onceki = r;
        }
        for y in 0..h {
            let yeni = load_u8x16(chunk(src, o + (y + 2 * P - 1) * ss));
            ik[2 * P - 2] = (
                _mm_unpacklo_epi16(onceki, yeni),
                _mm_unpackhi_epi16(onceki, yeni),
            );
            onceki = yeni;
            let mut lo = r;
            let mut hi = r;
            for (i, ki) in k.iter().enumerate().take(P) {
                lo = _mm_add_epi32(lo, _mm_madd_epi16(ik[2 * i].0, *ki));
                hi = _mm_add_epi32(hi, _mm_madd_epi16(ik[2 * i].1, *ki));
            }
            let (lo, hi) = (_mm_sra_epi32(lo, sh), _mm_sra_epi32(hi, sh));
            yaz8::<PREP>(hd, w, y, x, lo, hi, cv);
            ik.copy_within(1..2 * P - 1, 0);
        }
        x += 8;
        if x >= w {
            break;
        }
    }
}

/// İki boyutlu: 8 satırlık ara halka, satır başına `NC` adet 8 sütunluk xmm
/// (`NC = max(w, 8)/8`); dolgu satırı `j` halka yuvası `j & 7`; çıktı satırı
/// `y` ara satırları `y + S + 2i`, `+1`.
#[target_feature(enable = "ssse3")]
fn hv_yol<const PREP: bool, const P: usize, const S: usize, const NC: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    kh: [i32; 4],
    kv: [i32; 4],
    ara: Yuv,
    son: Yuv,
    c: Cikti,
) {
    let (kh, kv) = (kat(kh), kat(kv));
    let (ra, sa) = (_mm_set1_epi32(ara.r), _mm_cvtsi32_si128(ara.sh));
    let (rs, ssh) = (_mm_set1_epi32(son.r), _mm_cvtsi32_si128(son.sh));
    let cv = (_mm_set1_epi32(c.bias), _mm_set1_epi16(c.max as i16));
    let mut halka = [[_mm_setzero_si128(); NC]; 8];
    let mut j = S;
    for y in 0..h {
        while j < y + S + 2 * P {
            let o = j * ss;
            let satir = &mut halka[j & 7];
            for (q, hq) in satir.iter_mut().enumerate() {
                let (lo, hi) = yatay8::<P, S>(src, o + 16 * q, &kh);
                *hq = _mm_packs_epi32(yuv(lo, ra, sa), yuv(hi, ra, sa));
            }
            j += 1;
        }
        // Aynı `q` sütunu birden çok halka satırından okunur.
        #[allow(clippy::needless_range_loop)]
        for q in 0..NC {
            let mut lo = rs;
            let mut hi = rs;
            for (i, ki) in kv.iter().enumerate().take(P) {
                let a = halka[(y + S + 2 * i) & 7][q];
                let b = halka[(y + S + 2 * i + 1) & 7][q];
                lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *ki));
                hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), *ki));
            }
            let (lo, hi) = (_mm_sra_epi32(lo, ssh), _mm_sra_epi32(hi, ssh));
            yaz8::<PREP>(hd, w, y, 8 * q, lo, hi, cv);
        }
    }
}

// ---------------------------------------------------------------------------
// Yuvalar
// ---------------------------------------------------------------------------

/// 16 bpc `mc[f]` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn put16_fn(_ssse3: Ssse3Token, f: crate::dsp::mc::Filter2d) -> McPutFn {
    use crate::dsp::mc::Filter2d as F;
    match f {
        F::Regular => put16_giris::<0, 0>,
        F::RegularSmooth => put16_giris::<0, 1>,
        F::RegularSharp => put16_giris::<0, 2>,
        F::SharpRegular => put16_giris::<2, 0>,
        F::SharpSmooth => put16_giris::<2, 1>,
        F::Sharp => put16_giris::<2, 2>,
        F::SmoothRegular => put16_giris::<1, 0>,
        F::Smooth => put16_giris::<1, 1>,
        F::SmoothSharp => put16_giris::<1, 2>,
        F::Bilinear => put16_giris::<BILIN, BILIN>,
    }
}

/// 16 bpc `mct[f]` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn prep16_fn(_ssse3: Ssse3Token, f: crate::dsp::mc::Filter2d) -> McPrepFn {
    use crate::dsp::mc::Filter2d as F;
    match f {
        F::Regular => prep16_giris::<0, 0>,
        F::RegularSmooth => prep16_giris::<0, 1>,
        F::RegularSharp => prep16_giris::<0, 2>,
        F::SharpRegular => prep16_giris::<2, 0>,
        F::SharpSmooth => prep16_giris::<2, 1>,
        F::Sharp => prep16_giris::<2, 2>,
        F::SmoothRegular => prep16_giris::<1, 0>,
        F::Smooth => prep16_giris::<1, 1>,
        F::SmoothSharp => prep16_giris::<1, 2>,
        F::Bilinear => prep16_giris::<BILIN, BILIN>,
    }
}
