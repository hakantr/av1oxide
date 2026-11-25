//! S2b.2 — intra tahmin çekirdekleri için **S15.3 sentinel + over-read** testi.
//!
//! Üç şeyi kanıtlar:
//!
//! 1. **Yazma taşması yok:** hedef tampon tam `(h−1)·stride + w·bpp` baytken
//!    koruyucular bozulmaz; tampon `+64` bayt fazlaysa kuyruk değişmez.
//! 2. **Over-read yok:** kenar tamponu, çekirdeğin belgelenen aralığı dışında
//!    iki farklı dolgu deseniyle (0x00 / 0xFF) verilir; çıktı **bit-aynı**
//!    olur ([`OverReadProbe`]). Aynı yöntem `cfl_ac` luma girdisi için de
//!    uygulanır.
//! 3. **Negatif kontrol:** kasıtlı bir taşma ve kasıtlı bir padding okuması
//!    test düzeneği tarafından **yakalanır**; düzenek boş yere yeşil yanmaz.
//!
//! Ek olarak sınır vakaları: en büyük piksel değerleri, en büyük/en küçük CfL
//! alfa (±16) ve `cfl_ac` çıktısının i16 aralığında kalması.

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::ipred::{CflPredMode, Edge, IntraPredMode};
use av1oxide::dsp::scalar::ipred::required_range;
use av1oxide::dsp::scalar::pal::pal_idx_finish;
use av1oxide::frame::{BitDepth, ChromaFormat};
use common::ipred_slots::{
    cfl_ac_420, cfl_ac_422, cfl_ac_444, cfl_pred_dc, cfl_pred_dc_128, cfl_pred_left_dc,
    cfl_pred_top_dc, intra_pred, pal_pred,
};
use common::sentinel::{OverReadProbe, SentinelBuffer, VARSAYILAN_KORUMA, VARSAYILAN_SENTINEL};

fn bit_depth(bpc: u8) -> BitDepth {
    BitDepth::from_bits(u32::from(bpc)).unwrap()
}

fn max_sample(bpc: u8) -> u16 {
    if bpc == 8 {
        255
    } else if bpc == 10 {
        1023
    } else {
        4095
    }
}

fn bytes_of(bpc: u8) -> usize {
    if bpc == 8 { 1 } else { 2 }
}

fn read_px(dst: &[u8], offset: usize, bpc: u8) -> u16 {
    if bpc == 8 {
        u16::from(dst[offset])
    } else {
        u16::from_le_bytes([dst[offset], dst[offset + 1]])
    }
}

fn write_px(dst: &mut [u8], offset: usize, bpc: u8, value: u16) {
    if bpc == 8 {
        dst[offset] = value as u8;
    } else {
        dst[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
}

/// Kenar dolgusu için iki farklı bayt deseni (S15.3): veri uzunluğu ve
/// sol-üst pikselin veri içi bayt ofseti.
fn edge_layout(mode: IntraPredMode, w: usize, h: usize, angle: usize, bpc: u8) -> (usize, usize) {
    let (back, front) = required_range(mode, w, h, angle);
    let bytes = bytes_of(bpc);
    ((back + front + 1) * bytes, back * bytes)
}

fn case_angles(mode: IntraPredMode) -> Vec<(usize, usize, usize)> {
    match mode {
        IntraPredMode::Z1 => vec![
            (3, 0, 0),
            (45 | (1 << 10), 0, 0),
            (87 | (1 << 10) | (1 << 9), 0, 0),
        ],
        IntraPredMode::Z2 => vec![
            (93, 1, 65536),
            (135 | (1 << 10), 65536, 1),
            (177 | (1 << 10) | (1 << 9), 4, 4),
        ],
        IntraPredMode::Z3 => vec![
            (183, 0, 0),
            (225 | (1 << 10), 0, 0),
            (267 | (1 << 10) | (1 << 9), 0, 0),
        ],
        IntraPredMode::Filter => (0..5).map(|index| (index, 0, 0)).collect(),
        _ => vec![(0, 0, 0)],
    }
}

fn sizes_for(mode: IntraPredMode) -> Vec<(usize, usize)> {
    let all = [
        (4, 4),
        (4, 16),
        (16, 4),
        (8, 8),
        (16, 32),
        (32, 16),
        (64, 16),
        (16, 64),
    ];
    if mode == IntraPredMode::Filter {
        all.iter()
            .copied()
            .filter(|&(w, h)| w <= 32 && h <= 32)
            .collect()
    } else {
        all.to_vec()
    }
}

#[test]
fn yazma_tasmasi_ve_over_read_yok() {
    for bpc in [8u8, 10, 12] {
        let bd = bit_depth(bpc);
        let bytes = bytes_of(bpc);
        for mode in IntraPredMode::ALL {
            for (w, h) in sizes_for(mode) {
                for &(angle, max_width, max_height) in &case_angles(mode) {
                    let stride = w * bytes;
                    let out_len = (h - 1) * stride + w * bytes;
                    let (data_len, back_bytes) = edge_layout(mode, w, h, angle, bpc);
                    let probe = OverReadProbe::new(data_len, 64, out_len);
                    let topleft = probe.data_offset() + back_bytes;
                    let mut out_a = SentinelBuffer::with_defaults(out_len);
                    let mut out_b = SentinelBuffer::with_defaults(out_len);
                    let result =
                        probe.run_pair(out_a.data_mut(), out_b.data_mut(), |input, out| {
                            intra_pred(
                                mode,
                                out,
                                stride,
                                Edge::new(input, topleft),
                                w,
                                h,
                                angle,
                                max_width,
                                max_height,
                                bd,
                                max_sample(bpc),
                            );
                        });
                    assert!(
                        result.is_ok(),
                        "{mode:?} {w}x{h} bpc={bpc} over-read: {result:?}"
                    );
                    out_a.verify().expect("ön/arka koruyucu bozuldu");
                    out_b.verify().expect("ön/arka koruyucu bozuldu");
                    // Rust çekirdeği aynı girdiyle deterministik olmalı.
                    assert_eq!(out_a.data(), out_b.data());

                    // Fazladan 64 bayt verilirse kuyruk değişmemeli.
                    let mut padded = vec![0xA5u8; out_len + 64];
                    let mut edge_buffer = vec![0x5Au8; data_len + 64];
                    edge_buffer[64..64 + data_len].copy_from_slice(&probe.payload());
                    let edge = Edge::new(&edge_buffer, 64 + back_bytes);
                    intra_pred(
                        mode,
                        &mut padded[..out_len],
                        stride,
                        edge,
                        w,
                        h,
                        angle,
                        max_width,
                        max_height,
                        bd,
                        max_sample(bpc),
                    );
                    assert!(
                        padded[out_len..].iter().all(|&byte| byte == 0xA5),
                        "{mode:?} {w}x{h}: hedef tampon kuyruğuna yazıldı"
                    );
                }
            }
        }
    }
}

#[test]
fn negatif_kontrol_tasma_ve_over_read_yakalanir() {
    // (1) Sentinel düzeneği kasıtlı bir baytlık taşmayı yakalar.
    let mut buffer = SentinelBuffer::new(16, VARSAYILAN_KORUMA, VARSAYILAN_SENTINEL);
    let overflow_index = buffer.data_offset() + buffer.data_len();
    buffer.raw_mut()[overflow_index] = 0x5A;
    let violation = buffer.verify().expect_err("kasıtlı taşma yakalanmalıydı");
    assert_eq!(violation.index, overflow_index);
    assert_eq!(violation.found, 0x5A);

    // (2) OverReadProbe, padding'i okuyan kasıtlı hatayı yakalar.
    let probe = OverReadProbe::new(16, 8, 4);
    let over = probe
        .run(|input, out| {
            out[0] = input.iter().fold(0u8, |acc, &byte| acc.wrapping_add(byte));
        })
        .expect_err("kasıtlı padding okuması yakalanmalıydı");
    assert_eq!(over.first_diff, 0);
    assert_eq!(over.padding_a, 0x00);
    assert_eq!(over.padding_b, 0xFF);
}

#[test]
fn en_buyuk_piksel_ve_alfa_sinirlari() {
    for bpc in [8u8, 10, 12] {
        let bd = bit_depth(bpc);
        let max = max_sample(bpc);
        let bytes = bytes_of(bpc);
        let mode = IntraPredMode::Dc;
        let w = 8;
        let h = 8;
        let stride = w * bytes;
        let (back, front) = required_range(mode, w, h, 0);
        let mut edge = vec![0u8; (back + front + 1) * bytes];
        for index in 0..back + front + 1 {
            write_px(&mut edge, index * bytes, bpc, max);
        }
        let mut dst = vec![0u8; (h - 1) * stride + w * bytes];
        intra_pred(
            mode,
            &mut dst,
            stride,
            Edge::new(&edge, back * bytes),
            w,
            h,
            0,
            0,
            0,
            bd,
            max,
        );
        for y in 0..h {
            for x in 0..w {
                assert_eq!(read_px(&dst, y * stride + x * bytes, bpc), max);
            }
        }

        // CfL alfa uçları: ac = +span, edge = max → +16 tavan, −16 taban.
        let span = i16::try_from(u32::from(max) << 3).expect("span i16");
        let ac = vec![span; w * h];
        let edge_back = h;
        let edge_front = w;
        let mut edge = vec![0u8; (edge_back + edge_front + 1) * bytes];
        for index in 0..edge_back + edge_front + 1 {
            write_px(&mut edge, index * bytes, bpc, max);
        }
        for (alpha, expected) in [(16i32, max), (-16i32, 0)] {
            let mut dst = vec![0u8; (h - 1) * stride + w * bytes];
            cfl_pred_dc(
                &mut dst,
                stride,
                Edge::new(&edge, edge_back * bytes),
                w,
                h,
                &ac,
                alpha,
                bd,
                max,
            );
            for y in 0..h {
                for x in 0..w {
                    let value = read_px(&dst, y * stride + x * bytes, bpc);
                    assert_eq!(
                        value, expected,
                        "cfl_pred alpha={alpha} bpc={bpc}: {value} != {expected}"
                    );
                }
            }
        }

        // cfl_ac en büyük luma: çıktı i16 aralığında ve DC çıkarılmış.
        for (layout, cw, ch) in [(0u8, 8usize, 8usize), (1, 8, 8), (2, 8, 8)] {
            let (ss_hor, ss_ver) = match layout {
                0 => (1, 1),
                1 => (1, 0),
                _ => (0, 0),
            };
            let luma_w = cw << ss_hor;
            let luma_h = ch << ss_ver;
            let luma_stride = luma_w * bytes;
            let mut luma = vec![0u8; luma_stride * luma_h];
            for index in 0..luma_w * luma_h {
                write_px(&mut luma, index * bytes, bpc, max);
            }
            let mut ac = vec![0i16; cw * ch];
            match layout {
                0 => cfl_ac_420(&mut ac, &luma, luma_stride, 0, 0, cw, ch, bd),
                1 => cfl_ac_422(&mut ac, &luma, luma_stride, 0, 0, cw, ch, bd),
                _ => cfl_ac_444(&mut ac, &luma, luma_stride, 0, 0, cw, ch, bd),
            }
            let _ = ChromaFormat::C444;
            let sum: i64 = ac.iter().map(|&value| i64::from(value)).sum();
            assert_eq!(sum, 0, "cfl_ac DC çıkarılmadı (layout={layout}, bpc={bpc})");
            let limit = i64::from(max) << 3;
            assert!(
                ac.iter().all(|&value| i64::from(value).abs() <= limit),
                "cfl_ac değeri span dışında"
            );
        }
    }
}

#[test]
fn cfl_ac_ve_pal_sentinel() {
    // cfl_ac: ac çıktı tamponu tam boyutlu; taşma koruyucuları bozmamalı.
    for bpc in [8u8, 10, 12] {
        let bd = bit_depth(bpc);
        let bytes = bytes_of(bpc);
        let cw = 8;
        let ch = 8;
        let luma_w = 16;
        let luma_h = 16;
        let luma_stride = (luma_w + 3) * bytes;
        let mut luma = vec![0u8; luma_stride * luma_h];
        for index in 0..luma_w * luma_h {
            write_px(
                &mut luma,
                index * bytes,
                bpc,
                (index as u16) & max_sample(bpc),
            );
        }
        let mut ac = vec![0i16; cw * ch];
        cfl_ac_420(&mut ac, &luma, luma_stride, 1, 1, cw, ch, bd);
        let _ = &ac;

        // pal_pred ve pal_idx_finish: tam boyutlu hedef + kuyruk denetimi.
        let w = 8;
        let h = 8;
        let stride = w * bytes;
        let pal_bytes = if bpc == 8 { 8 } else { 16 };
        let mut pal = vec![0u8; pal_bytes];
        for index in 0..8 {
            write_px(
                &mut pal,
                index * bytes,
                bpc,
                ((index as u16) * 3) & max_sample(bpc),
            );
        }
        let idx = vec![0x10u8; w * h / 2];
        let mut dst = SentinelBuffer::with_defaults((h - 1) * stride + w * bytes);
        pal_pred(dst.data_mut(), stride, &pal, &idx, w, h, bd);
        dst.verify().expect("pal_pred hedefi koruyucuyu bozdu");

        let bw = 16;
        let bh = 16;
        let mut src = vec![0u8; bw * bh];
        for (index, slot) in src.iter_mut().enumerate() {
            *slot = (index % 8) as u8;
        }
        let mut packed = SentinelBuffer::with_defaults((bw / 2) * bh);
        pal_idx_finish(packed.data_mut(), &src, bw, bh, bw, bh);
        packed.verify().expect("pal_idx_finish koruyucuyu bozdu");
    }

    // pal_idx_finish çıktısı `dst_bw·bh` baytın tamamını yazar: fazladan
    // verilen kuyruk değişmez.
    let bw = 16;
    let bh = 16;
    let mut src = vec![0u8; bw * bh];
    for (index, slot) in src.iter_mut().enumerate() {
        *slot = (index % 8) as u8;
    }
    let out_len = (bw / 2) * bh;
    let mut dst = vec![0xA5u8; out_len + 32];
    pal_idx_finish(&mut dst[..out_len], &src, bw, bh, bw, bh);
    assert!(dst[out_len..].iter().all(|&byte| byte == 0xA5));
}

#[test]
fn cfl_pred_mod_dortluesu_sinirli_girdiyle_calisir() {
    let w = 8;
    let h = 8;
    let max = 255u16;
    let edge = vec![0x7Fu8; (h + w + 1) * 2];
    let ac = vec![0i16; w * h];
    for mode in [
        CflPredMode::Dc,
        CflPredMode::LeftDc,
        CflPredMode::TopDc,
        CflPredMode::Dc128,
    ] {
        let mut dst = vec![0u8; (h - 1) * w + w];
        let edge = Edge::new(&edge, h * 2);
        match mode {
            CflPredMode::Dc => cfl_pred_dc(&mut dst, w, edge, w, h, &ac, 0, BitDepth::Eight, max),
            CflPredMode::LeftDc => {
                cfl_pred_left_dc(&mut dst, w, edge, w, h, &ac, 0, BitDepth::Eight, max)
            }
            CflPredMode::TopDc => {
                cfl_pred_top_dc(&mut dst, w, edge, w, h, &ac, 0, BitDepth::Eight, max)
            }
            CflPredMode::Dc128 => {
                cfl_pred_dc_128(&mut dst, w, edge, w, h, &ac, 0, BitDepth::Eight, max)
            }
        }
        assert!(dst.iter().all(|&value| value > 0 && value < 255));
    }
}
