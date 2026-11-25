//! P44 (Adım 2 / 3) — 10 bpc AVX512ICL ters dönüşüm (`75` kayıt: `itx16_avx512` `60` + `itx16_buyuk_avx512` `15`) doğrulama testleri.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::itx::{
    ItxDsp, ItxFn, N_RECT_TX_SIZES, N_TX_TYPES_PLUS_LL, TX_SIZES, TX_TYPES, TxSize, TxType,
    is_supported, itx_skaler_devir_sayisi,
};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::SentinelBuffer;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Rng(u64);

impl Rng {
    fn u64(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn aralik(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (i64::from(hi) - i64::from(lo) + 1) as u64;
        (i64::from(lo) + ((self.u64() >> 16) % span) as i64) as i32
    }
}

fn wht_disi_155_hucre() -> Vec<(TxSize, TxType)> {
    let mut v = Vec::with_capacity(155);
    for tx in TX_SIZES {
        for txtp in TX_TYPES {
            if is_supported(tx, txtp) {
                v.push((tx, txtp));
            }
        }
    }
    assert_eq!(v.len(), 155);
    v
}

fn avx512_destekli_75_hucre() -> Vec<(TxSize, TxType)> {
    let v: Vec<(TxSize, TxType)> = wht_disi_155_hucre()
        .into_iter()
        .filter(|&(tx, _)| tx.width() >= 8 && tx.height() >= 8)
        .collect();
    assert_eq!(v.len(), 75);
    v
}

fn avx512_tablo(
    test: &str,
) -> Option<(DspDispatch, [[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES])> {
    #[cfg(target_arch = "x86_64")]
    {
        let dsp = DspDispatch::select(CpuFlags::detect());
        if dsp.avx512icl_token().is_some() {
            let tablo = dsp.for_bit_depth(BitDepth::Ten).itx().itxfm_add;
            return Some((dsp, tablo));
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P44] 10 bpc ITX AVX512ICL gövdeleri x86_64 AVX512ICL ister",
    );
    None
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

fn read_i32<R: Read>(r: &mut R) -> i32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).expect("i32 okuma");
    i32::from_le_bytes(b)
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
    if bpc == 8 {
        i32::from(x as i16)
    } else {
        x as i32
    }
}

fn buyuk_oracle_dizini(manifest: &Path) -> Option<PathBuf> {
    let yerel = manifest.join("../../.cache/itx-oracle");
    if yerel.join("itx-large-10-input.bin").exists()
        && yerel.join("itx-large-10-expect.bin").exists()
    {
        return Some(yerel);
    }
    if let Ok(env_dir) = std::env::var("ITX_ORACLE_DIR") {
        let p = PathBuf::from(env_dir);
        if p.join("itx-large-10-input.bin").exists() && p.join("itx-large-10-expect.bin").exists() {
            return Some(p);
        }
    }
    None
}

fn dogrula_oracle_10bpc(
    dir: &Path,
    profile_name: &str,
    tablo: &[[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES],
) -> (usize, usize, usize) {
    let in_path = dir.join(format!("itx-{profile_name}-10-input.bin"));
    let exp_path = dir.join(format!("itx-{profile_name}-10-expect.bin"));
    let in_f =
        File::open(&in_path).unwrap_or_else(|e| panic!("{} açılamadı: {e}", in_path.display()));
    let exp_f =
        File::open(&exp_path).unwrap_or_else(|e| panic!("{} açılamadı: {e}", exp_path.display()));
    let mut inp = BufReader::new(in_f);
    let mut exp = BufReader::new(exp_f);

    let mut magic = [0u8; 4];
    inp.read_exact(&mut magic).expect("inp magic");
    assert_eq!(&magic, b"ITXI");
    assert_eq!(read_u32(&mut inp), 1);
    assert_eq!(read_u8(&mut inp), 10);
    let profile = read_u8(&mut inp);
    assert_eq!(read_u16(&mut inp), 0);
    let count = read_u32(&mut inp);

    exp.read_exact(&mut magic).expect("exp magic");
    assert_eq!(&magic, b"ITXE");
    assert_eq!(read_u32(&mut exp), 1);
    assert_eq!(read_u8(&mut exp), 10);
    assert_eq!(read_u8(&mut exp), profile);
    assert_eq!(read_u16(&mut exp), 0);
    assert_eq!(read_u32(&mut exp), count);

    let mut toplam = 0usize;
    let mut wht_disi = 0usize;
    let mut avx512_blok = 0usize;
    let mut coeff = [0i32; 1024];

    for rec in 0..u64::from(count) {
        let tx_u8 = read_u8(&mut inp);
        let txtp_u8 = read_u8(&mut inp);
        let rec_bpc = read_u8(&mut inp);
        assert_eq!(read_u8(&mut inp), 0);
        let eob = usize::from(read_u16(&mut inp));
        assert_eq!(read_u16(&mut inp), 0);
        assert_eq!(rec_bpc, 10);

        let tx = TxSize::from_index(usize::from(tx_u8)).expect("tx");
        let txtp = TxType::from_index(usize::from(txtp_u8)).expect("txtp");
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);

        for v in coeff.iter_mut().take(sw * sh) {
            *v = read_i32(&mut inp);
        }
        for (idx, v) in coeff.iter_mut().enumerate().skip(sw * sh) {
            *v = tail_value(idx, rec, 10);
        }

        let e_tx = read_u8(&mut exp);
        let e_txtp = read_u8(&mut exp);
        let e_bpc = read_u8(&mut exp);
        assert_eq!(read_u8(&mut exp), 0);
        let e_eob = usize::from(read_u16(&mut exp));
        assert_eq!(read_u16(&mut exp), 0);
        assert_eq!((e_tx, e_txtp, e_bpc, e_eob), (tx_u8, txtp_u8, 10, eob));

        let mut exp_dst = vec![0u8; w * h * 2];
        for chunk in exp_dst.as_chunks_mut::<2>().0 {
            let v = read_u16(&mut exp);
            assert!(v <= 1023);
            *chunk = v.to_le_bytes();
        }

        let mut dst = vec![0u8; w * h * 2];
        let mut st = pix_seed(rec, 10);
        let variant = if profile == 0 { (rec % 6) as u8 } else { 0xff };
        for chunk in dst.as_chunks_mut::<2>().0 {
            let v = pix_next(&mut st) & 1023;
            let px = if profile == 0 && (variant == 1 || variant == 5) {
                1023u16
            } else if profile == 0 && (variant == 0 || variant == 2) {
                0u16
            } else {
                v as u16
            };
            *chunk = px.to_le_bytes();
        }

        let f = tablo[tx.index()][txtp.index()];
        f(&mut dst, w * 2, &mut coeff, eob, tx, txtp, BitDepth::Ten);

        for (i, &c) in coeff[..sw * sh].iter().enumerate() {
            assert_eq!(
                c, 0,
                "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) coeff[{i}] != 0"
            );
        }
        for (i, &c) in coeff.iter().enumerate().skip(sw * sh) {
            assert_eq!(
                c,
                tail_value(i, rec, 10),
                "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) kuyruk coeff[{i}] bozuldu"
            );
        }
        assert_eq!(
            dst, exp_dst,
            "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) 10 bpc piksel çıktısı oracle'dan farklı"
        );

        toplam += 1;
        if txtp != TxType::WhtWht {
            wht_disi += 1;
            if w >= 8 && h >= 8 {
                avx512_blok += 1;
            }
        }
    }
    (toplam, wht_disi, avx512_blok)
}

#[test]
fn test_itx10_avx512_slot_install_and_12bpc_isolation() {
    const T: &str = "itx16_avx512::test_itx10_avx512_slot_install_and_12bpc_isolation";
    let Some(_) = avx512_tablo(T) else {
        return;
    };

    let all_x86 = CpuFlags::X86_DAV1D;
    let fast_gather = all_x86.difference(CpuFlags::SLOW_GATHER);

    for (etiket, bayraklar) in [
        ("ALL_X86 & !SLOW_GATHER", fast_gather),
        ("ALL_X86 (SLOW_GATHER)", all_x86),
    ] {
        let dsp = DspDispatch::select(bayraklar);
        assert!(
            dsp.avx512icl_token().is_some(),
            "{etiket} altında Avx512IclToken üretilmeli"
        );
        let tablo10 = dsp.for_bit_depth(BitDepth::Ten).itx().itxfm_add;
        let tablo12 = dsp.for_bit_depth(BitDepth::Twelve).itx().itxfm_add;

        let mut say_avx512_10 = 0usize;
        let mut say_avx2_10 = 0usize;

        for (tx, txtp) in wht_disi_155_hucre() {
            let alan = format!("itxfm_add[{}][{}]", tx.dav1d_name(), txtp.dav1d_name());
            let impl10 = dsp.slot_implementation(BitDepth::Ten, "itx", &alan);
            let impl12 = dsp.slot_implementation(BitDepth::Twelve, "itx", &alan);
            if tx.width() >= 8 && tx.height() >= 8 {
                assert_eq!(
                    impl10, "avx512icl",
                    "{etiket}: 10 bpc {alan} (w>=8, h>=8) avx512icl olmalı"
                );
                say_avx512_10 += 1;
            } else {
                assert_eq!(
                    impl10, "avx2",
                    "{etiket}: 10 bpc {alan} (w==4 || h==4) avx2 olmalı"
                );
                say_avx2_10 += 1;
            }
            assert_ne!(
                impl12, "avx512icl",
                "{etiket}: 12 bpc {alan} asla avx512icl olmamalı"
            );
            let f10 = tablo10[tx.index()][txtp.index()] as usize;
            let f12 = tablo12[tx.index()][txtp.index()] as usize;
            assert_ne!(
                f10, f12,
                "{etiket}: 12 bpc {alan} yuvası 10 bpc işlev işaretçisini paylaşmamalı"
            );
        }

        let wht_10 = dsp.slot_implementation(BitDepth::Ten, "itx", "itxfm_add[TX_4X4][WHT_WHT]");
        let wht_12 = dsp.slot_implementation(BitDepth::Twelve, "itx", "itxfm_add[TX_4X4][WHT_WHT]");
        assert_eq!(wht_10, "avx2", "{etiket}: 10 bpc WHT_WHT avx2 olmalı");
        assert_eq!(wht_12, "avx2", "{etiket}: 12 bpc WHT_WHT avx2 olmalı");
        say_avx2_10 += 1;

        assert_eq!(
            say_avx512_10, 75,
            "{etiket}: 10 bpc'de tam 75 yuva avx512icl olmalı"
        );
        assert_eq!(
            say_avx2_10, 81,
            "{etiket}: 10 bpc'de kalan 81 yuva (80 dar + 1 WHT_WHT) avx2 olmalı"
        );
    }
}

#[test]
fn test_itx10_avx512_c_oracle_small_vectors() {
    const T: &str = "itx16_avx512::test_itx10_avx512_c_oracle_small_vectors";
    let Some((_, tablo)) = avx512_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let small_dir = manifest.join("tests/fixtures/itx");
    let (small_toplam, small_wht_disi, small_avx512) =
        dogrula_oracle_10bpc(&small_dir, "small", &tablo);
    assert_eq!(small_toplam, 936);
    assert_eq!(small_wht_disi, 930);
    assert_eq!(small_avx512, 450);
    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "10 bpc AVX512ICL small oracle koşusunda skaler devir 0 olmalı"
    );
}

#[test]
fn test_itx10_avx512_c_oracle_large_vectors() {
    const T: &str = "itx16_avx512::test_itx10_avx512_c_oracle_large_vectors";
    let Some((_, tablo)) = avx512_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let large_dir =
        buyuk_oracle_dizini(&manifest).expect(".cache/itx-oracle/itx-large-10-*.bin bulunmalı");
    let (large_toplam, wht_disi, avx512_blok) = dogrula_oracle_10bpc(&large_dir, "large", &tablo);
    assert_eq!(large_toplam, 156_000);
    assert_eq!(wht_disi, 155_000);
    assert_eq!(avx512_blok, 75_000);
    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "10 bpc AVX512ICL large oracle koşusunda skaler devir 0 olmalı"
    );
    println!(
        "[P44 ITX10 AVX512ICL oracle] large={wht_disi}/155000 WHT-dışı blok ({avx512_blok} doğrudan avx512icl) %100 bit-exact"
    );
}

#[test]
fn test_itx10_avx512_sentinel_and_contract_fallback() {
    const T: &str = "itx16_avx512::test_itx10_avx512_sentinel_and_contract_fallback";
    let Some((_, tablo)) = avx512_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();
    let skaler8 = ItxDsp::scalar::<8>().itxfm_add;
    let skaler10 = ItxDsp::scalar::<10>().itxfm_add;
    let skaler12 = ItxDsp::scalar::<12>().itxfm_add;
    let mut rng = Rng(0x5044_0002_2026_1117);

    // A) 75 AVX512ICL hücresinde SentinelBuffer, çift bayt stride, eob alt-blok eşikleri ve coeff sıfırlama sözleşmesi.
    for (tx, txtp) in avx512_destekli_75_hucre() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let row_bytes = w * 2;
        let f = tablo[tx.index()][txtp.index()];
        let f_ref = skaler10[tx.index()][txtp.index()];

        for &stride in &[row_bytes, row_bytes + 2, row_bytes + 6, row_bytes + 14] {
            let data_len = (h - 1) * stride + row_bytes;
            for &eob in &[
                0usize,
                1.min(n - 1),
                35.min(n - 1),
                42.min(n - 1),
                106.min(n - 1),
                135.min(n - 1),
                150.min(n - 1),
                278.min(n - 1),
                n - 1,
            ] {
                let mut c_temel = [0i32; 1024];
                if eob <= 1 && eob < n - 1 {
                    c_temel[0] = rng.aralik(-16_384, 16_384);
                } else if eob == n - 1 {
                    let lim = (16_384 / (sw.max(sh) as i32)).max(256);
                    for x in 0..sw {
                        for y in 0..sh {
                            c_temel[y + x * sh] = rng.aralik(-lim, lim);
                        }
                    }
                } else {
                    let (yatay_id, dikey_id) = match txtp {
                        TxType::VDct | TxType::VAdst | TxType::VFlipadst => (true, false),
                        TxType::HDct | TxType::HAdst | TxType::HFlipadst => (false, true),
                        _ => (false, false),
                    };
                    let lim = 1_024;
                    for x in 0..sw {
                        for y in 0..sh {
                            let gecerli = if yatay_id {
                                x + y * sw <= eob
                            } else if dikey_id {
                                y + x * sh <= eob
                            } else if eob <= 42 {
                                x < 4 && y < 4 && x + y <= 4
                            } else {
                                x < 8 && y < 8 && x + y <= 8
                            };
                            if gecerli {
                                c_temel[y + x * sh] = rng.aralik(-lim, lim);
                            }
                        }
                    }
                }

                let mut s0 = SentinelBuffer::with_defaults(data_len);
                let mut s1 = SentinelBuffer::with_defaults(data_len);
                let mut s_ref = SentinelBuffer::with_defaults(data_len);
                s0.data_mut().fill(0x00);
                s1.data_mut().fill(0xFF);
                s_ref.data_mut().fill(0x00);

                for y in 0..h {
                    for x in 0..w {
                        let px = (rng.aralik(0, 1023) as u16).to_le_bytes();
                        let off = y * stride + x * 2;
                        s0.data_mut()[off..off + 2].copy_from_slice(&px);
                        s1.data_mut()[off..off + 2].copy_from_slice(&px);
                        s_ref.data_mut()[off..off + 2].copy_from_slice(&px);
                    }
                }

                let mut c0 = c_temel;
                let mut c1 = c_temel;
                let mut c_ref = c_temel;
                for i in n..1024 {
                    c0[i] = 0x1357_2468;
                    c1[i] = -0x2468_1357;
                }
                if txtp == TxType::DctDct && eob == 0 {
                    for i in 1..n {
                        c0[i] = 0x1122_3344;
                        c1[i] = -0x1122_3344;
                    }
                }

                f(s0.data_mut(), stride, &mut c0, eob, tx, txtp, BitDepth::Ten);
                f(s1.data_mut(), stride, &mut c1, eob, tx, txtp, BitDepth::Ten);
                f_ref(
                    s_ref.data_mut(),
                    stride,
                    &mut c_ref,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Ten,
                );

                assert_eq!(s0.verify(), Ok(()));
                assert_eq!(s1.verify(), Ok(()));
                assert_eq!(
                    s0.data(),
                    s_ref.data(),
                    "{tx:?}/{txtp:?} stride={stride} eob={eob} AVX512ICL != 10 bpc skaler"
                );
                for y in 0..h {
                    assert_eq!(
                        &s0.data()[y * stride..y * stride + row_bytes],
                        &s1.data()[y * stride..y * stride + row_bytes]
                    );
                    if y + 1 < h {
                        for b in row_bytes..stride {
                            assert_eq!(s0.data()[y * stride + b], 0x00);
                            assert_eq!(s1.data()[y * stride + b], 0xFF);
                        }
                    }
                }
                if txtp == TxType::DctDct && eob == 0 {
                    assert_eq!(c0[0], 0);
                    assert!(c0[1..n].iter().all(|&v| v == 0x1122_3344));
                } else {
                    assert!(c0[..n].iter().all(|&v| v == 0));
                }
                assert!(c0[n..].iter().all(|&v| v == 0x1357_2468));
            }
        }
    }

    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "sözleşme içi 10 bpc AVX512ICL çağrılarında skaler devir 0 olmalı"
    );

    // B) Sözleşme dışı girdilerde ve BitDepth::Eight / BitDepth::Twelve çağrılarında güvenli skaler devir.
    for &(tx, txtp) in &[
        (TxSize::Tx8x8, TxType::DctDct),
        (TxSize::Rtx8x16, TxType::FlipadstAdst),
        (TxSize::Rtx16x8, TxType::AdstDct),
        (TxSize::Tx16x16, TxType::AdstAdst),
        (TxSize::Rtx8x32, TxType::Idtx),
        (TxSize::Rtx32x8, TxType::DctDct),
        (TxSize::Rtx16x32, TxType::DctDct),
        (TxSize::Rtx32x16, TxType::Idtx),
        (TxSize::Tx32x32, TxType::DctDct),
        (TxSize::Tx64x64, TxType::DctDct),
    ] {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let stride = w * 2;
        let f = tablo[tx.index()][txtp.index()];

        // 1) Yanlış `tx`
        let baska_tx = if tx == TxSize::Tx8x8 {
            TxSize::Tx16x16
        } else {
            TxSize::Tx8x8
        };
        let bw = baska_tx.width();
        let bh = baska_tx.height();
        let mut dst_simd = vec![0u8; bw * bh * 2];
        for chunk in dst_simd.as_chunks_mut::<2>().0 {
            *chunk = 300u16.to_le_bytes();
        }
        let mut dst_ref = dst_simd.clone();
        let mut c_simd = [0i32; 1024];
        c_simd[0] = 1_280;
        let mut c_ref = c_simd;
        let d0 = itx_skaler_devir_sayisi();
        f(
            &mut dst_simd,
            bw * 2,
            &mut c_simd,
            0,
            baska_tx,
            TxType::DctDct,
            BitDepth::Ten,
        );
        (skaler10[baska_tx.index()][TxType::DctDct.index()])(
            &mut dst_ref,
            bw * 2,
            &mut c_ref,
            0,
            baska_tx,
            TxType::DctDct,
            BitDepth::Ten,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d0 + 1);
        assert_eq!(dst_simd, dst_ref);

        // 2) Yanlış `txtp`
        let baska_txtp = if txtp == TxType::DctDct {
            TxType::Idtx
        } else {
            TxType::DctDct
        };
        if is_supported(tx, baska_txtp) {
            let mut dst_t = vec![0u8; w * h * 2];
            let mut c_t = [0i32; 1024];
            let d_t = itx_skaler_devir_sayisi();
            f(
                &mut dst_t,
                stride,
                &mut c_t,
                0,
                tx,
                baska_txtp,
                BitDepth::Ten,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d_t + 1);
        }

        // 3) `BitDepth::Twelve` ile doğrudan çağrı -> 12 bpc skaler devir
        let mut dst12_simd = vec![0u8; w * h * 2];
        for chunk in dst12_simd.as_chunks_mut::<2>().0 {
            *chunk = 2_500u16.to_le_bytes();
        }
        let mut dst12_ref = dst12_simd.clone();
        let mut c12_simd = [0i32; 1024];
        c12_simd[0] = 8_192;
        let mut c12_ref = c12_simd;
        let d12 = itx_skaler_devir_sayisi();
        f(
            &mut dst12_simd,
            stride,
            &mut c12_simd,
            0,
            tx,
            txtp,
            BitDepth::Twelve,
        );
        (skaler12[tx.index()][txtp.index()])(
            &mut dst12_ref,
            stride,
            &mut c12_ref,
            0,
            tx,
            txtp,
            BitDepth::Twelve,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d12 + 1);
        assert_eq!(dst12_simd, dst12_ref);

        // 4) `BitDepth::Eight` ile doğrudan çağrı -> 8 bpc skaler devir
        let mut dst8_simd = vec![120u8; w * h];
        let mut dst8_ref = dst8_simd.clone();
        let mut c8_simd = [0i32; 1024];
        c8_simd[0] = 512;
        let mut c8_ref = c8_simd;
        let d8 = itx_skaler_devir_sayisi();
        f(
            &mut dst8_simd,
            w,
            &mut c8_simd,
            0,
            tx,
            txtp,
            BitDepth::Eight,
        );
        (skaler8[tx.index()][txtp.index()])(
            &mut dst8_ref,
            w,
            &mut c8_ref,
            0,
            tx,
            txtp,
            BitDepth::Eight,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d8 + 1);
        assert_eq!(dst8_simd, dst8_ref);

        // 5) Tek sayı `dst_stride`, kısa `dst_stride`, geçersiz `eob`, kısa `dst` ve kısa `coeff`
        let tek_stride = stride + 1;
        let mut dst_tek = vec![0u8; (h - 1) * tek_stride + stride];
        let mut c_ok = [0i32; 1024];
        let d_tek = itx_skaler_devir_sayisi();
        f(
            &mut dst_tek,
            tek_stride,
            &mut c_ok,
            0,
            tx,
            txtp,
            BitDepth::Ten,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d_tek + 1);

        let mut dst_ok = vec![0u8; w * h * 2];
        let d_dar = itx_skaler_devir_sayisi();
        f(
            &mut dst_ok,
            stride - 2,
            &mut c_ok,
            0,
            tx,
            txtp,
            BitDepth::Ten,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d_dar + 1);

        for gecersiz_eob in [sw * sh, usize::MAX] {
            let d = itx_skaler_devir_sayisi();
            f(
                &mut dst_ok,
                stride,
                &mut c_ok,
                gecersiz_eob,
                tx,
                txtp,
                BitDepth::Ten,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d + 1);
        }

        let mut kisa_dst = vec![0u8; (h - 1) * stride + stride - 1];
        let d_kisa_dst = itx_skaler_devir_sayisi();
        f(&mut kisa_dst, stride, &mut c_ok, 0, tx, txtp, BitDepth::Ten);
        assert_eq!(itx_skaler_devir_sayisi(), d_kisa_dst + 1);

        let mut kisa_coeff = [0i32; 512];
        let d_kisa_c = itx_skaler_devir_sayisi();
        f(
            &mut dst_ok,
            stride,
            &mut kisa_coeff,
            0,
            tx,
            txtp,
            BitDepth::Ten,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d_kisa_c + 1);
    }
}

#[test]
fn test_itx10_avx512_extreme_saturation_parity() {
    const T: &str = "itx16_avx512::test_itx10_avx512_extreme_saturation_parity";
    let Some((_, tablo_avx512)) = avx512_tablo(T) else {
        return;
    };
    let detected = CpuFlags::detect();
    let tablo_avx2 = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()))
        .for_bit_depth(BitDepth::Ten)
        .itx()
        .itxfm_add;
    let tablo_skaler = ItxDsp::scalar::<10>().itxfm_add;
    let mut rng = Rng(0xDEAD_BEEF_2026_5120);

    for (tx, txtp) in avx512_destekli_75_hucre() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let row_bytes = w * 2;
        let stride = row_bytes + 6;
        let data_len = (h - 1) * stride + row_bytes;
        let f_avx512 = tablo_avx512[tx.index()][txtp.index()];
        let f_avx2 = tablo_avx2[tx.index()][txtp.index()];
        let f_skaler = tablo_skaler[tx.index()][txtp.index()];

        // 1) Tekil/seyrek uç [-131072, +131071] doyurma darbeleri, dama tahtası ve rastgele katsayılar:
        //    `itxfm_add_rust::<10>` (`ItxDsp::scalar::<10>()`) ile %100 bit-birebir eşitlik!
        for kip in 0..9u8 {
            let mut c_base = [0i32; 1024];
            let eob = match kip {
                0 => {
                    c_base[0] = 131_071;
                    0
                }
                1 => {
                    c_base[0] = -131_072;
                    0
                }
                2 => {
                    c_base[0] = 131_071;
                    n - 1
                }
                3 => {
                    c_base[0] = -131_072;
                    n - 1
                }
                4 => {
                    let pos = (sw / 2) * sh + (sh / 2);
                    c_base[pos] = 131_071;
                    n - 1
                }
                5 => {
                    let pos = (sw - 1) * sh + (sh - 1);
                    c_base[pos] = -131_072;
                    n - 1
                }
                6 => {
                    let lim = (16_384 / (sw.max(sh) as i32)).max(256);
                    for x in 0..sw {
                        for y in 0..sh {
                            c_base[y + x * sh] = if (x + y) & 1 == 0 { lim } else { -lim };
                        }
                    }
                    n - 1
                }
                _ => {
                    let lim = (24_576 / (sw.max(sh) as i32)).max(384);
                    for v in c_base.iter_mut().take(n) {
                        *v = rng.aralik(-lim, lim);
                    }
                    n - 1
                }
            };
            for v in c_base.iter_mut().skip(n) {
                *v = 0x4242_1010;
            }

            let mut sb_avx512 = SentinelBuffer::with_defaults(data_len);
            let mut sb_ref = SentinelBuffer::with_defaults(data_len);
            sb_avx512.data_mut().fill(0x5A);
            sb_ref.data_mut().fill(0x5A);
            for y in 0..h {
                for x in 0..w {
                    let px = match (x + y + usize::from(kip)) % 3 {
                        0 => 0u16,
                        1 => 1023u16,
                        _ => 512u16,
                    };
                    let off = y * stride + x * 2;
                    sb_avx512.data_mut()[off..off + 2].copy_from_slice(&px.to_le_bytes());
                    sb_ref.data_mut()[off..off + 2].copy_from_slice(&px.to_le_bytes());
                }
            }
            let mut c_avx512 = c_base;
            let mut c_ref = c_base;
            f_avx512(
                sb_avx512.data_mut(),
                stride,
                &mut c_avx512,
                eob,
                tx,
                txtp,
                BitDepth::Ten,
            );
            f_skaler(
                sb_ref.data_mut(),
                stride,
                &mut c_ref,
                eob,
                tx,
                txtp,
                BitDepth::Ten,
            );

            assert_eq!(sb_avx512.verify(), Ok(()));
            assert_eq!(
                sb_avx512.data(),
                sb_ref.data(),
                "{tx:?}/{txtp:?} kip={kip}: uç doyurma darbesinde AVX512ICL != skaler 10 bpc"
            );
            assert_eq!(&c_avx512[..], &c_ref[..]);
        }

        // 2) Tam blok [-131072, +131071] ve i32::MIN/MAX taşma taraması:
        //    paniksiz, SentinelBuffer temiz, [0, 1023] aralığında ve AVX2 10-bpc ile %100 bit-exact!
        for kip in 0..6u8 {
            let mut c_base = [0i32; 1024];
            for (i, v) in c_base.iter_mut().enumerate().take(n) {
                *v = match kip {
                    0 => 131_071,
                    1 => -131_072,
                    2 => {
                        if i & 1 == 0 {
                            131_071
                        } else {
                            -131_072
                        }
                    }
                    3 => i32::MAX,
                    4 => i32::MIN,
                    _ => rng.aralik(-131_072, 131_071),
                };
            }
            for v in c_base.iter_mut().skip(n) {
                *v = 0x4242_1010;
            }
            let eob = if kip == 0 { 0 } else { n - 1 };

            let mut sb_avx512 = SentinelBuffer::with_defaults(data_len);
            let mut sb_avx2 = SentinelBuffer::with_defaults(data_len);
            sb_avx512.data_mut().fill(0x5A);
            sb_avx2.data_mut().fill(0x5A);
            for y in 0..h {
                for x in 0..w {
                    let px = if (x + y) & 1 == 0 { 0u16 } else { 1023u16 };
                    let off = y * stride + x * 2;
                    sb_avx512.data_mut()[off..off + 2].copy_from_slice(&px.to_le_bytes());
                    sb_avx2.data_mut()[off..off + 2].copy_from_slice(&px.to_le_bytes());
                }
            }
            let mut c_avx512 = c_base;
            let mut c_avx2 = c_base;
            f_avx512(
                sb_avx512.data_mut(),
                stride,
                &mut c_avx512,
                eob,
                tx,
                txtp,
                BitDepth::Ten,
            );
            f_avx2(
                sb_avx2.data_mut(),
                stride,
                &mut c_avx2,
                eob,
                tx,
                txtp,
                BitDepth::Ten,
            );

            assert_eq!(sb_avx512.verify(), Ok(()));
            assert_eq!(
                sb_avx512.data(),
                sb_avx2.data(),
                "{tx:?}/{txtp:?} tam blok uç kip={kip}: AVX512ICL != AVX2"
            );
            assert_eq!(&c_avx512[..], &c_avx2[..]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_itx10_avx512_isa_boundary_no_gather_and_stack_frame_objdump() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let mut kucuk_sembol = 0usize;
    let mut buyuk_sembol = 0usize;
    let mut kucuk_zmm = 0usize;
    let mut buyuk_zmm = 0usize;
    let mut max_rsp_sub = 0usize;
    let mut aktif: Option<&str> = None;
    let mut aktif_ad = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") {
            aktif_ad = satir.to_string();
            let uretim_x86 =
                satir.contains("av1oxide") && satir.contains("3dsp") && satir.contains("3x86");
            if uretim_x86
                && (satir.contains("itx16_buyuk_avx512") || satir.contains("itx10_buyuk_avx512_"))
            {
                aktif = Some("itx16_buyuk_avx512");
                buyuk_sembol += 1;
            } else if uretim_x86
                && (satir.contains("itx16_avx512") || satir.contains("itx10_avx512_"))
            {
                aktif = Some("itx16_avx512");
                kucuk_sembol += 1;
            } else {
                aktif = None;
            }
            continue;
        }
        if satir.is_empty() {
            aktif = None;
            continue;
        }
        if let Some(modul) = aktif
            && let Some((_, sag)) = satir.split_once('\t')
        {
            let sag = sag.trim();
            assert!(
                !sag.starts_with("vpgather") && !sag.starts_with("vpscatter"),
                "{modul} Zen 4 yasaklı gather/scatter yönergesi üretti: {aktif_ad} -> {satir}"
            );
            if sag.contains("%zmm") {
                if modul == "itx16_avx512" {
                    kucuk_zmm += 1;
                } else {
                    buyuk_zmm += 1;
                }
            }
            if !cfg!(debug_assertions)
                && let Some(kalan) = sag.strip_prefix("sub")
            {
                let kalan = kalan.trim_start();
                if let Some((imm, hedef)) = kalan.split_once(',')
                    && hedef.trim() == "%rsp"
                    && let Some(hex) = imm.trim().strip_prefix("$0x")
                    && let Ok(bytes) = usize::from_str_radix(hex, 16)
                {
                    max_rsp_sub = max_rsp_sub.max(bytes);
                    assert!(
                        bytes < 0x400,
                        "{modul} aşırı yığın çerçevesi ({bytes:#x} >= 0x400): {aktif_ad}"
                    );
                }
            }
        }
    }

    assert!(
        kucuk_sembol >= 60,
        "dsp::x86::itx16_avx512 içinde en az 60 sembol bulunmalı: {kucuk_sembol}"
    );
    assert!(
        buyuk_sembol >= 15,
        "dsp::x86::itx16_buyuk_avx512 içinde en az 15 sembol bulunmalı: {buyuk_sembol}"
    );
    assert!(
        buyuk_zmm > 0,
        "itx16_buyuk_avx512 içinde 512-bit %zmm yönergesi bulunmalı"
    );
    println!(
        "[P44 ITX10 AVX512ICL objdump] itx16_avx512={kucuk_sembol} sembol ({kucuk_zmm} %zmm), itx16_buyuk_avx512={buyuk_sembol} sembol ({buyuk_zmm} %zmm), gather/scatter=0, max_rsp_sub={max_rsp_sub:#x}"
    );
}
