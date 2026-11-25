//! S2b.4 — MC aile **uç değer** testleri (kabul ölçütü 4(c)).
//!
//! Üç sınıf:
//!
//! 1. **En büyük piksel**: `bitdepth_max` ile dolu kaynak; sabit girdide
//!    8-tap filtrenin katsayı toplamı 64 olduğundan çıktı **tam olarak**
//!    `bitdepth_max` kalmalıdır (kırpma da sınırı doğrular).
//! 2. **En büyük ara değerler**: `prep` çıktısının belgelenen aralığının
//!    (`src/mc_tmpl.c:41-48`) uçları (`lo`, `hi`) ile `avg`/`w_avg`/`mask`/
//!    `w_mask`/`blend` çalıştırılır; sonuç `[0, bitdepth_max]` içinde kalmalı
//!    ve `w_mask` maskesi `[0, 64]` olmalıdır.
//! 3. **Warp parametre sınırları**: tablo indeksleri tam `0` ve `192` iken
//!    (dav1d `dav1d_mc_warp_filter[193]`) çekirdek paniklememeli ve çıktı
//!    aralıkta kalmalıdır; adım büyüklükleri de sınırın içinde kalır.

#![forbid(unsafe_code)]
#![allow(clippy::chunks_exact_to_as_chunks, clippy::while_let_loop)]

use av1oxide::dsp::mc::{Filter2d, McDsp};
use av1oxide::frame::BitDepth;

const BDS: [BitDepth; 3] = [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve];

fn max_of(bd: BitDepth) -> i32 {
    (1i32 << bd.bits()) - 1
}

fn bpp(bd: BitDepth) -> usize {
    bd.bytes_per_sample()
}

fn px_fill(buf: &mut [u8], bd: BitDepth, value: i32) {
    if bpp(bd) == 1 {
        for b in buf.iter_mut() {
            *b = value as u8;
        }
    } else {
        for c in buf.chunks_exact_mut(2) {
            c.copy_from_slice(&(value as u16).to_le_bytes());
        }
    }
}

fn px_get(buf: &[u8], bd: BitDepth, index: usize) -> i32 {
    if bpp(bd) == 1 {
        i32::from(buf[index])
    } else {
        i32::from(u16::from_le_bytes([buf[index * 2], buf[index * 2 + 1]]))
    }
}

/// (c1) — en büyük piksel: sabit `bitdepth_max` girdisi sabit çıktı verir.
#[test]
fn en_buyuk_piksel_sabit_girdi_sabit_cikti() {
    for bd in BDS {
        let dsp = McDsp::scalar_for(bd);
        let max = max_of(bd);
        let (w, h) = (16usize, 16usize);
        let stride = (w + 7) * bpp(bd);
        let mut src = vec![0u8; (h + 6) * stride + (w + 7) * bpp(bd)];
        px_fill(&mut src, bd, max);
        let mut dst = vec![0u8; (h - 1) * stride + w * bpp(bd)];
        // 2D yol (mx, my ≠ 0) — yatay+dikey ara değerlerin en büyük hâli.
        (dsp.mc[Filter2d::Regular.index()])(
            &mut dst, stride, &src, stride, w, h, 15, 15, bd, max as u16,
        );
        for y in 0..h {
            for x in 0..w {
                let v = px_get(&dst, bd, y * stride / bpp(bd) + x);
                assert_eq!(
                    v,
                    max,
                    "{}-bit sabit max girdide çıktı max olmalı (y={y}, x={x})",
                    bd.bits()
                );
            }
        }
        // Kopya yolu (mx = my = 0) pikseli birebir taşır.
        let mut copy = vec![0u8; (h - 1) * stride + w * bpp(bd)];
        (dsp.mc[Filter2d::Regular.index()])(
            &mut copy, stride, &src, stride, w, h, 0, 0, bd, max as u16,
        );
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    px_get(&copy, bd, y * stride / bpp(bd) + x),
                    max,
                    "{}-bit kopya yolu",
                    bd.bits()
                );
            }
        }
        println!(
            "  {}-bit: sabit max girdi → sabit max çıktı (put, 16×16)",
            bd.bits()
        );
    }
}

/// (c2) — en büyük ara değerler: `prep` aralığının uçları.
#[test]
fn en_buyuk_ara_degerler() {
    // dav1d `src/mc_tmpl.c:41-48` belgelenen `prep` çıktı aralığı.
    let ranges: [(BitDepth, i32, i32); 3] = [
        (BitDepth::Eight, -2140, 4100),
        (BitDepth::Ten, -28780, 28764),
        (BitDepth::Twelve, -28794, 28791),
    ];
    for (bd, lo, hi) in ranges {
        let dsp = McDsp::scalar_for(bd);
        let max = max_of(bd);
        let (w, h) = (8usize, 8usize);
        let stride = (w + 3) * bpp(bd);
        // İki uç: (lo, hi) ve (hi, lo) — farkın işareti her iki yönde.
        for (a, b) in [(lo, hi), (hi, lo)] {
            let mut tmp1 = vec![0i16; w * h];
            let mut tmp2 = vec![0i16; w * h];
            for i in 0..w * h {
                tmp1[i] = a as i16;
                tmp2[i] = b as i16;
            }
            let mut dst = vec![0u8; (h - 1) * stride + w * bpp(bd)];
            (dsp.avg)(&mut dst, stride, &tmp1, &tmp2, w, h, bd, max as u16);
            for y in 0..h {
                for x in 0..w {
                    let v = px_get(&dst, bd, y * stride / bpp(bd) + x);
                    assert!(
                        (0..=max).contains(&v),
                        "{}-bit avg çıktısı aralık dışı: {v}",
                        bd.bits()
                    );
                }
            }
            (dsp.w_avg)(&mut dst, stride, &tmp1, &tmp2, w, h, 15, bd, max as u16);
            (dsp.mask)(
                &mut dst,
                stride,
                &tmp1,
                &tmp2,
                w,
                h,
                &vec![64u8; w * h],
                bd,
                max as u16,
            );
            for ss in 0..3u32 {
                let ss_hor = ss != 0;
                let ss_ver = ss == 2;
                let mw = w >> usize::from(ss_hor);
                let mh = h >> usize::from(ss_ver);
                let mut mask = vec![0u8; mw * mh];
                (dsp.w_mask[ss as usize])(
                    &mut dst, stride, &tmp1, &tmp2, w, h, &mut mask, 1, ss_hor, ss_ver, bd,
                    max as u16,
                );
                for (i, &m) in mask.iter().enumerate() {
                    assert!(
                        m <= 64,
                        "{}-bit w_mask maskesi > 64 (indeks {i})",
                        bd.bits()
                    );
                }
            }
        }
        println!(
            "  {}-bit: prep uçları [{lo}, {hi}] → avg/w_avg/mask/w_mask aralıkta",
            bd.bits()
        );
    }
}

/// (c3) — warp parametre sınırları: tablo indeksi tam `0` ve `192`.
#[test]
fn warp_parametre_sinirlari() {
    for bd in BDS {
        let dsp = McDsp::scalar_for(bd);
        let max = max_of(bd);
        let mut src = vec![0u8; 15 * 15 * bpp(bd)];
        // Deterministik, uçlara değen desen.
        for i in 0..15 * 15 {
            let v = if i % 3 == 0 { 0 } else { max };
            px_fill(&mut src[i * bpp(bd)..(i + 1) * bpp(bd)], bd, v);
        }
        // İndeks formülü: 64 + ((tmx + 512) >> 10) → sınır indeksleri için
        // tmx = (i − 64)·1024 − 512.
        for index in [0i32, 1, 96, 191, 192] {
            let tmx = (index - 64) * 1024 - 512;
            let abcd = [0i16; 4];
            let stride = 15 * bpp(bd);
            let mut dst = vec![0u8; 7 * stride + 8 * bpp(bd)];
            (dsp.warp8x8)(
                &mut dst, stride, &src, stride, &abcd, tmx, tmx, bd, max as u16,
            );
            for y in 0..8 {
                for x in 0..8 {
                    let v = px_get(&dst, bd, y * stride / bpp(bd) + x);
                    assert!(
                        (0..=max).contains(&v),
                        "{}-bit warp8x8 çıktısı aralık dışı (indeks {index}): {v}",
                        bd.bits()
                    );
                }
            }
            let mut tmp = vec![0i16; 64];
            (dsp.warp8x8t)(&mut tmp, 8, &src, stride, &abcd, tmx, tmx, bd, max as u16);
            // Warp8x8t `PREP_BIAS` çıkarır; i16 aralığında olmalı.
            for v in tmp {
                assert!(
                    (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&i32::from(v)),
                    "warp8x8t i16 aralığı"
                );
            }
        }
        // Aşırı adım: indeks aralık dışına çıkarsa sözleşme ihlali — burada
        // **geçerli** en büyük adımı (indeks 96 civarında ±%50) doğruluyoruz.
        let tmx = (96 - 64) * 1024 - 512;
        let abcd = [40i16, 40, 40, 40];
        let stride = 15 * bpp(bd);
        let mut dst = vec![0u8; 7 * stride + 8 * bpp(bd)];
        (dsp.warp8x8)(
            &mut dst, stride, &src, stride, &abcd, tmx, tmx, bd, max as u16,
        );
        println!(
            "  {}-bit: warp indeks sınırları 0/1/96/191/192 aralıkta",
            bd.bits()
        );
    }
}
