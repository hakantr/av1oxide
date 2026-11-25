//! P53 (`F5c` — `F5-ITX`) `aarch64` NEON ters dönüşüm (`3 × 156 = 468` yuva) bit-birebir
//! doğruluk, C oracle (`small` `2 808` + `large` `468 000` + `extreme` `468 000` blok),
//! `SentinelBuffer`, `OverReadProbe` ve sözleşme dışı devir testleri.

#![forbid(unsafe_code)]

#[cfg(target_arch = "aarch64")]
mod common;

use av1oxide::dsp::ARM_ISA_CANDIDATES;

#[test]
fn p53_arm64_itx_aday_sayilari() {
    let n = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "itx" && c.implementation == "neon")
        .count();
    assert_eq!(n, 468, "itx neon adayı: 3 bit derinliği × 156 alan");
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::common::sentinel::{OverReadProbe, SentinelBuffer};
    use av1oxide::cpu::CpuFlags;
    use av1oxide::dsp::DspDispatch;
    use av1oxide::dsp::arm::itx_alanlari::ITX_NEON_ALANLARI;
    use av1oxide::dsp::itx::{
        ItxDsp, TX_SIZES, TX_TYPES, TxSize, TxType, is_supported, itx_skaler_devir_sayisi,
    };
    use av1oxide::frame::BitDepth;
    use std::fs::File;
    use std::io::{BufReader, Read};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static DEVRI_KILIT: Mutex<()> = Mutex::new(());

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn aralik(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i64
        }
    }

    fn hucreler() -> Vec<(TxSize, TxType)> {
        ITX_NEON_ALANLARI
            .iter()
            .map(|f| {
                let ic = f
                    .strip_prefix("itxfm_add[")
                    .unwrap()
                    .strip_suffix(']')
                    .unwrap();
                let (a, b) = ic.split_once("][").unwrap();
                let tx = *TX_SIZES.iter().find(|t| t.dav1d_name() == a).unwrap();
                let tt = TX_TYPES
                    .iter()
                    .copied()
                    .chain(core::iter::once(TxType::WhtWht))
                    .find(|t| t.dav1d_name() == b)
                    .unwrap();
                (tx, tt)
            })
            .collect()
    }

    fn cf_max(bd: BitDepth) -> i64 {
        (1i64
            << (if bd == BitDepth::Eight {
                15
            } else {
                7 + bd.bits()
            }))
            - 1
    }

    fn read_u8<R: Read>(r: &mut R) -> u8 {
        let mut b = [0u8; 1];
        r.read_exact(&mut b).expect("u8 okuma");
        b[0]
    }

    fn read_u16<R: Read>(r: &mut R) -> u16 {
        let mut b = [0u8; 2];
        r.read_exact(&mut b).expect("u16 okuma");
        u16::from_le_bytes(b)
    }

    fn read_u32<R: Read>(r: &mut R) -> u32 {
        let mut b = [0u8; 4];
        r.read_exact(&mut b).expect("u32 okuma");
        u32::from_le_bytes(b)
    }

    fn read_i16<R: Read>(r: &mut R) -> i16 {
        i16::from_le_bytes([read_u8(r), read_u8(r)])
    }

    fn read_i32<R: Read>(r: &mut R) -> i32 {
        i32::from_le_bytes([read_u8(r), read_u8(r), read_u8(r), read_u8(r)])
    }

    fn pix_seed(record_index: u64, bpc: u8) -> u64 {
        0x9E37_79B9_7F4A_7C15u64.wrapping_mul(record_index + 1) ^ (u64::from(bpc) << 51)
    }

    fn pix_next(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 33) as u32
    }

    fn tail_value(index: usize, record_index: u64, bpc: u8) -> i32 {
        let x = (index as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((record_index as u32).wrapping_mul(40_503))
            .wrapping_add(u32::from(bpc) * 97);
        let v = if bpc == 8 {
            i32::from(x as i16)
        } else {
            x as i32
        };
        if v == 0 { 13_579 } else { v }
    }

    fn dogrula_c_oracle_kumesi(
        dir: &Path,
        profile_name: &str,
        neon: &DspDispatch,
    ) -> Option<usize> {
        let mut toplam_kayit = 0usize;
        for (bpc, bd) in [
            (8u8, BitDepth::Eight),
            (10u8, BitDepth::Ten),
            (12u8, BitDepth::Twelve),
        ] {
            let in_path = dir.join(format!("itx-{profile_name}-{bpc}-input.bin"));
            let exp_path = dir.join(format!("itx-{profile_name}-{bpc}-expect.bin"));
            let (Ok(in_f), Ok(exp_f)) = (File::open(&in_path), File::open(&exp_path)) else {
                return None;
            };
            let mut inp = BufReader::new(in_f);
            let mut exp = BufReader::new(exp_f);

            let mut magic = [0u8; 4];
            inp.read_exact(&mut magic).expect("ITXI magic");
            assert_eq!(&magic, b"ITXI");
            assert_eq!(read_u32(&mut inp), 1);
            assert_eq!(read_u8(&mut inp), bpc);
            let profile = read_u8(&mut inp);
            assert_eq!(read_u16(&mut inp), 0);
            let count = read_u32(&mut inp);

            exp.read_exact(&mut magic).expect("ITXE magic");
            assert_eq!(&magic, b"ITXE");
            assert_eq!(read_u32(&mut exp), 1);
            assert_eq!(read_u8(&mut exp), bpc);
            let _ = read_u8(&mut exp);
            assert_eq!(read_u16(&mut exp), 0);
            assert_eq!(read_u32(&mut exp), count);

            let tablo = neon.for_bit_depth(bd).itx().itxfm_add;
            let bpp = if bpc == 8 { 1 } else { 2 };
            let max_sample = (1u32 << bpc) - 1;
            let mut dst = vec![0u8; 64 * 64 * 2];

            for rec in 0..u64::from(count) {
                let tx_u8 = read_u8(&mut inp);
                let txtp_u8 = read_u8(&mut inp);
                let rec_bpc = read_u8(&mut inp);
                let _ = read_u8(&mut inp);
                let eob = usize::from(read_u16(&mut inp));
                let _ = read_u16(&mut inp);
                assert_eq!(rec_bpc, bpc);

                let tx = TxSize::from_index(usize::from(tx_u8)).expect("tx");
                let txtp = TxType::from_index(usize::from(txtp_u8)).expect("txtp");
                assert!(is_supported(tx, txtp));
                let w = tx.width();
                let h = tx.height();
                let sw = w.min(32);
                let sh = h.min(32);
                let stride = w * bpp;

                let mut coeff_in = [0i32; 1024];
                for v in coeff_in.iter_mut().take(sw * sh) {
                    *v = if bpc == 8 {
                        i32::from(read_i16(&mut inp))
                    } else {
                        read_i32(&mut inp)
                    };
                }
                for (idx, v) in coeff_in.iter_mut().enumerate().skip(sw * sh) {
                    *v = tail_value(idx, rec, bpc);
                }

                let is_dconly = txtp == TxType::DctDct && eob == 0;
                if is_dconly {
                    for (idx, v) in coeff_in.iter_mut().enumerate().skip(1) {
                        *v = tail_value(idx, rec, bpc);
                    }
                }

                let e_tx = read_u8(&mut exp);
                let e_txtp = read_u8(&mut exp);
                let e_bpc = read_u8(&mut exp);
                let _ = read_u8(&mut exp);
                let e_eob = usize::from(read_u16(&mut exp));
                let _ = read_u16(&mut exp);
                assert_eq!((e_tx, e_txtp, e_bpc, e_eob), (tx_u8, txtp_u8, bpc, eob));

                let mut st = pix_seed(rec, bpc);
                let variant = if profile == 0 { (rec % 6) as u8 } else { 0xff };
                let d_slice = &mut dst[..w * h * bpp];
                for p in 0..w * h {
                    let v = pix_next(&mut st) & max_sample;
                    let val = if profile == 0 && (variant == 1 || variant == 5) {
                        max_sample
                    } else if profile == 0 && (variant == 0 || variant == 2) {
                        0
                    } else {
                        v
                    };
                    if bpc == 8 {
                        d_slice[p] = val as u8;
                    } else {
                        d_slice[2 * p..2 * p + 2].copy_from_slice(&(val as u16).to_le_bytes());
                    }
                }

                let mut coeff = coeff_in;
                let f = tablo[tx.index()][txtp.index()];
                f(d_slice, stride, &mut coeff, eob, tx, txtp, bd);

                for p in 0..w * h {
                    let want = read_u16(&mut exp);
                    let got = if bpc == 8 {
                        u16::from(d_slice[p])
                    } else {
                        u16::from_le_bytes([d_slice[2 * p], d_slice[2 * p + 1]])
                    };
                    assert_eq!(
                        got, want,
                        "{profile_name} bpc={bpc} rec={rec} {tx:?}/{txtp:?} eob={eob} piksel {p}"
                    );
                }

                if is_dconly {
                    assert_eq!(coeff[0], 0, "{profile_name} rec={rec} DC coeff[0] != 0");
                    assert_eq!(
                        &coeff[1..],
                        &coeff_in[1..],
                        "{profile_name} rec={rec} DC-only coeff[1..1024] değişti"
                    );
                } else {
                    assert!(
                        coeff[..sw * sh].iter().all(|&c| c == 0),
                        "{profile_name} rec={rec} {tx:?}/{txtp:?} coeff[..{}] sıfırlanmadı",
                        sw * sh
                    );
                    assert_eq!(
                        &coeff[sw * sh..],
                        &coeff_in[sw * sh..],
                        "{profile_name} rec={rec} {tx:?}/{txtp:?} kuyruk bozuldu"
                    );
                }
                toplam_kayit += 1;
            }
        }
        Some(toplam_kayit)
    }

    #[test]
    fn p53_itx_yuva_kurulumu() {
        let neon = DspDispatch::select(CpuFlags::detect());
        let skaler = DspDispatch::select(CpuFlags::NONE);
        let n = neon
            .slot_states()
            .iter()
            .filter(|s| s.key.family == "itx" && s.implementation == "neon")
            .count();
        assert_eq!(n, 468);
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            assert_eq!(
                neon.slot_implementation(bd, "itx", "itxfm_add[TX_4X4][WHT_WHT]"),
                "neon"
            );
            assert_eq!(
                skaler.slot_implementation(bd, "itx", "itxfm_add[TX_64X64][DCT_DCT]"),
                "skaler"
            );
        }
    }

    #[test]
    fn p53_itx_8_10_12bpc_neon_skaler_bit_exact_sentinel() {
        let _g = DEVRI_KILIT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let neon = DspDispatch::select(CpuFlags::detect());
        let skaler = DspDispatch::select(CpuFlags::NONE);
        let devir0 = itx_skaler_devir_sayisi();
        let mut r = Rng(0x1357_9BDF_2468_ACE0);
        let mut sayac = 0usize;
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            let bpp = if bd == BitDepth::Eight { 1 } else { 2 };
            let maxp = (1i64 << bd.bits()) - 1;
            let cm = cf_max(bd);
            let tn = neon.for_bit_depth(bd).itx().itxfm_add;
            let ts = skaler.for_bit_depth(bd).itx().itxfm_add;
            for (tx, tt) in hucreler() {
                let (w, h) = (tx.width(), tx.height());
                let n = w.min(32) * h.min(32);
                let fn_n = tn[tx.index()][tt.index()];
                let fn_s = ts[tx.index()][tt.index()];
                let tur_sayisi = if tt == TxType::WhtWht { 8 } else { 14 };
                for kip in 0..tur_sayisi {
                    let mut c = [0i32; 1024];
                    let (lo, hi) = if tt == TxType::WhtWht {
                        (-(1 << 15), (1 << 15) - 1)
                    } else {
                        (-cm - 1, cm)
                    };
                    for (i, v) in c.iter_mut().enumerate() {
                        *v = if i < n {
                            match kip {
                                0 => hi as i32,
                                1 => lo as i32,
                                2 => {
                                    if i & 1 == 0 {
                                        hi as i32
                                    } else {
                                        lo as i32
                                    }
                                }
                                3 => {
                                    if i % 3 == 0 {
                                        hi as i32
                                    } else {
                                        (lo / 3) as i32
                                    }
                                }
                                4..=7 => r.aralik(lo / 64, hi / 64) as i32,
                                _ => r.aralik(lo, hi) as i32,
                            }
                        } else {
                            0x2121_4343
                        };
                    }
                    let eob = match kip {
                        0..=3 => n - 1,
                        8 if tt == TxType::DctDct => 0,
                        _ => r.aralik(0, n as i64 - 1) as usize,
                    };
                    let ds = w * bpp + bpp * (r.next() % 4) as usize;
                    let dlen = (h - 1) * ds + w * bpp;
                    let mut taban = vec![0xA5u8; dlen];
                    for y in 0..h {
                        for x in 0..w {
                            let p = match (x + y + kip) & 3 {
                                0 => 0,
                                1 => maxp,
                                _ => r.aralik(0, maxp),
                            };
                            let o = y * ds + x * bpp;
                            if bpp == 1 {
                                taban[o] = p as u8;
                            } else {
                                taban[o..o + 2].copy_from_slice(&(p as u16).to_le_bytes());
                            }
                        }
                    }
                    let mut c_ref = c;
                    let mut d_ref = taban.clone();
                    fn_s(&mut d_ref, ds, &mut c_ref, eob, tx, tt, bd);
                    let mut sb = SentinelBuffer::with_defaults(dlen);
                    sb.write_data(&taban);
                    fn_n(sb.data_mut(), ds, &mut c, eob, tx, tt, bd);
                    sb.verify().expect("itx sentinel");
                    assert_eq!(
                        sb.data(),
                        &d_ref[..],
                        "itx bd={bd:?} {tx:?}/{tt:?} kip={kip} eob={eob}"
                    );
                    assert_eq!(
                        c[..],
                        c_ref[..],
                        "itx katsayı tamponu bd={bd:?} {tx:?}/{tt:?} kip={kip}"
                    );
                    sayac += 1;
                }
            }
        }
        assert_eq!(
            itx_skaler_devir_sayisi(),
            devir0,
            "sözleşme içi çağrıda skaler devir olmamalı"
        );
        assert!(sayac > 3 * 155 * 14);
    }

    #[test]
    fn p53_itx_c_oracle_small_large_extreme_bit_exact() {
        let _g = DEVRI_KILIT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let neon = DspDispatch::select(CpuFlags::detect());
        let devir0 = itx_skaler_devir_sayisi();

        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let small_dir = manifest.join("tests/fixtures/itx");
        let small_n = dogrula_c_oracle_kumesi(&small_dir, "small", &neon)
            .expect("tests/fixtures/itx small kümesi bulunmalı");
        assert_eq!(small_n, 2_808, "small kümesi 3 × 936 = 2 808 blok olmalı");

        if !cfg!(debug_assertions) || std::env::var_os("AV1OXIDE_ITX_FULL").is_some() {
            let cache_dir = manifest.join("../../.cache/itx-oracle");
            if let Some(large_n) = dogrula_c_oracle_kumesi(&cache_dir, "large", &neon) {
                assert_eq!(large_n, 468_000, "large kümesi 468 000 blok olmalı");
            }
            if let Some(ext_n) = dogrula_c_oracle_kumesi(&cache_dir, "extreme", &neon) {
                assert_eq!(ext_n, 468_000, "extreme kümesi 468 000 blok olmalı");
            }
        }

        assert_eq!(
            itx_skaler_devir_sayisi(),
            devir0,
            "C oracle bloklarında skaler devir 0 olmalı"
        );
    }

    #[test]
    fn p53_itx_overread_probe_ve_sozlesme_disi_devir() {
        let _g = DEVRI_KILIT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let neon = DspDispatch::select(CpuFlags::detect());
        let devir0 = itx_skaler_devir_sayisi();

        // 1) OverReadProbe: 468 yuvanın tamamında hem hedef satır sonu dolgusu hem de
        //    katsayı kuyruğu (`sw·sh..1024`) `0x00` vs `0xFF` dolgu deseniyle sınanır.
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            let bpp = if bd == BitDepth::Eight { 1 } else { 2 };
            let maskp = ((1u32 << bd.bits()) - 1) as u16;
            let tn = neon.for_bit_depth(bd).itx().itxfm_add;
            for (tx, tt) in hucreler() {
                let (w, h) = (tx.width(), tx.height());
                let (sw, sh) = (w.min(32), h.min(32));
                let eob = if tt == TxType::WhtWht { 0 } else { sw * sh - 1 };
                let fn_n = tn[tx.index()][tt.index()];
                let ds = w * bpp + 7 * bpp;
                let dlen = (h - 1) * ds + w * bpp;
                let probe = OverReadProbe::new(sw * sh * 4, 64, h * w * bpp);

                let sonuc = probe.run_pair(
                    &mut vec![0u8; h * w * bpp],
                    &mut vec![0u8; h * w * bpp],
                    |inp, out_px| {
                        let pad = inp[0];
                        let mut sb = SentinelBuffer::with_defaults(dlen);
                        sb.data_mut().fill(pad);
                        for y in 0..h {
                            for x in 0..w {
                                let v = (((x * 37 + y * 13 + 5) as u16) & maskp).to_le_bytes();
                                let o = y * ds + x * bpp;
                                if bpp == 1 {
                                    sb.data_mut()[o] = v[0];
                                } else {
                                    sb.data_mut()[o..o + 2].copy_from_slice(&v);
                                }
                            }
                        }
                        let mut coeff = [i32::from_le_bytes([pad; 4]); 1024];
                        let veri =
                            &inp[probe.data_offset()..probe.data_offset() + probe.data_len()];
                        for (i, slot) in coeff.iter_mut().take(sw * sh).enumerate() {
                            let raw = i16::from_le_bytes([veri[4 * i], veri[4 * i + 1]]);
                            *slot = i32::from(raw >> 2);
                        }
                        fn_n(sb.data_mut(), ds, &mut coeff, eob, tx, tt, bd);
                        sb.verify().expect("OverReadProbe sentinel");
                        for y in 0..h - 1 {
                            for b in w * bpp..ds {
                                assert_eq!(
                                    sb.data()[y * ds + b],
                                    pad,
                                    "stride dolgusu yazıldı: bd={bd:?} {tx:?}/{tt:?}"
                                );
                            }
                        }
                        for y in 0..h {
                            out_px[y * w * bpp..(y + 1) * w * bpp]
                                .copy_from_slice(&sb.data()[y * ds..y * ds + w * bpp]);
                        }
                    },
                );
                assert!(
                    sonuc.is_ok(),
                    "OverReadProbe fark yakaladı: bd={bd:?} {tx:?}/{tt:?} -> {sonuc:?}"
                );
            }
        }
        assert_eq!(
            itx_skaler_devir_sayisi(),
            devir0,
            "OverReadProbe sözleşme içi çağrılarında skaler devir 0 olmalı"
        );

        // 2) Sözleşme dışı girdilerde paniklemeden skaler devir sayacı artışı.
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            let bpp = if bd == BitDepth::Eight { 1 } else { 2 };
            let diger_bd = if bd == BitDepth::Eight {
                BitDepth::Ten
            } else {
                BitDepth::Eight
            };
            let f = neon.for_bit_depth(bd).itx().itxfm_add[TxSize::Tx8x8.index()]
                [TxType::DctDct.index()];
            let f_sk =
                ItxDsp::scalar_for(bd).itxfm_add[TxSize::Tx8x8.index()][TxType::DctDct.index()];
            let mut dst = vec![64u8; 8 * 8 * bpp];
            let mut dst_ref = dst.clone();
            let mut c = [0i32; 1024];
            c[0] = 120;
            let mut c_ref = c;

            // Yanlış BitDepth → skaler monomorfa devreder + sayaç +1.
            let d0 = itx_skaler_devir_sayisi();
            f(
                &mut dst,
                8 * bpp,
                &mut c,
                0,
                TxSize::Tx8x8,
                TxType::DctDct,
                diger_bd,
            );
            f_sk(
                &mut dst_ref,
                8 * bpp,
                &mut c_ref,
                0,
                TxSize::Tx8x8,
                TxType::DctDct,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d0 + 1);
            assert_eq!(dst, dst_ref);

            // eob >= sw * sh → sayaç +1, panik yok.
            let d1 = itx_skaler_devir_sayisi();
            f(
                &mut dst,
                8 * bpp,
                &mut c,
                64,
                TxSize::Tx8x8,
                TxType::DctDct,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d1 + 1);

            // Kısa stride → sayaç +1, panik yok.
            let d2 = itx_skaler_devir_sayisi();
            f(
                &mut dst,
                8 * bpp - 1,
                &mut c,
                0,
                TxSize::Tx8x8,
                TxType::DctDct,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d2 + 1);

            // Kısa dst → sayaç +1, panik yok.
            let mut kisa_dst = vec![0u8; 8 * 8 * bpp - 1];
            let d3 = itx_skaler_devir_sayisi();
            f(
                &mut kisa_dst,
                8 * bpp,
                &mut c,
                0,
                TxSize::Tx8x8,
                TxType::DctDct,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d3 + 1);

            // Kısa coeff (< 1024) → sayaç +1, panik yok.
            let mut kisa_c = [0i32; 512];
            let d4 = itx_skaler_devir_sayisi();
            f(
                &mut dst,
                8 * bpp,
                &mut kisa_c,
                0,
                TxSize::Tx8x8,
                TxType::DctDct,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d4 + 1);

            // Desteklenmeyen (Tx64x64, AdstAdst) → sayaç +1, panik yok.
            let mut dst64 = vec![0u8; 64 * 64 * bpp];
            let d5 = itx_skaler_devir_sayisi();
            f(
                &mut dst64,
                64 * bpp,
                &mut c,
                0,
                TxSize::Tx64x64,
                TxType::AdstAdst,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d5 + 1);
        }
    }
}
