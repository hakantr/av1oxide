//! P44 Adım 1 (Paket B) — 12 bpc WHT-dışı ters dönüşüm (`147` kayıt = `140` `w, h ≤ 16` +
//! `7` büyük, yalnız `BitDepth::Twelve`) AVX2 (`itx16_12.rs` + `itx16_12_buyuk.rs`)
//! doğrulama testleri:
//!
//! 1. `test_itx12_avx2_slot_install_and_8_large_dct_remain_scalar`: `CpuLevel::Avx2` altında
//!    `12 bpc` tablosunda `147` WHT-dışı ITX yuvasının + `1` WHT yuvasının `"avx2"` olarak
//!    kurulduğunu, en büyük `8` `DCT_DCT` dönüşümünün (`16x32`, `32x16`, `32x32`, `16x64`,
//!    `32x64`, `64x16`, `64x32`, `64x64` × `DCT_DCT`) `12 bpc`'de `None` dönerek `"skaler"`de
//!    kaldığını doğrular (`dav1d` `src/x86/itx.h:249-263` kapsam paritesi).
//! 2. `test_itx12_avx2_c_oracle_small_vectors`: `tests/fixtures/itx/itx-small-12-{input,expect}.bin`
//!    (`936` toplam = `930` WHT-dışı: `882` AVX2 + `48` skaler büyük `DCT_DCT`) üzerinde `%100`
//!    bit-exact (`coeff` sıfırlama sınırı dahil).
//! 3. `test_itx12_avx2_c_oracle_large_vectors`: `.cache/itx-oracle/itx-large-12-{input,expect}.bin`
//!    (`156 000` toplam = `155 000` WHT-dışı: `147 000` AVX2 + `8 000` skaler büyük `DCT_DCT`)
//!    üzerinde `%100` bit-exact.
//! 4. `test_itx12_avx2_sentinel_and_contract_fallback`: `SentinelBuffer` taşma `0`, sözleşme dışı
//!    çağrılarda (`8 bpc`, `10 bpc`, tek bayt `dst_stride`, kısa `dst`/`coeff`, taşan `eob`,
//!    yanlış `tx`/`txtp`) `itx_skaler_devir_sayisi` artışı ve skalerle birebirlik.
//! 5. `test_itx12_avx2_extreme_saturation_parity`: rastgele/uç `[-524288, +524287]` (`clip_20b`)
//!    ve `i32::MIN/MAX` katsayılarında skaler 12 bpc referansıyla `%100` bit-exact.
//! 6. `test_itx12_avx2_isa_boundary_and_stack_frame_objdump`: `objdump -d` ile `itx16_12` ve
//!    `itx16_12_buyuk` sembollerinde EVEX (`%zmm`, `{%k`) yönergesi `0` ve `sub $N, %rsp`
//!    yığın çerçevesi `< 0x400` (`1024` bayt).

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

mod common;

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::itx::{
    ItxDsp, ItxFn, N_RECT_TX_SIZES, N_TX_TYPES_PLUS_LL, TX_SIZES, TX_TYPES, TxSize, TxType,
    is_supported, itx_skaler_devir_sayisi,
};
#[cfg(target_arch = "x86_64")]
use av1oxide::dsp::x86::itx16_12::itx12_avx2_lookup;
#[cfg(target_arch = "x86_64")]
use av1oxide::dsp::x86::itx16_12_buyuk::itx12_buyuk_avx2_lookup;

/// `x86_64` dışı hedeflerde (`aarch64`, P52) AVX2 gövdesi yoktur; testler
/// `avx2_12_tablo` içinde `on_kosul::eksik` ile atlanır, bu yer tutucular
/// yalnız derlenebilirlik içindir ve hiçbir zaman `Some` dönmez.
#[cfg(not(target_arch = "x86_64"))]
const fn itx12_avx2_lookup(
    _token: av1oxide::cpu::Avx2Token,
    _tx: TxSize,
    _txtp: TxType,
) -> Option<ItxFn> {
    None
}
#[cfg(not(target_arch = "x86_64"))]
const fn itx12_buyuk_avx2_lookup(
    _token: av1oxide::cpu::Avx2Token,
    _tx: TxSize,
    _txtp: TxType,
) -> Option<ItxFn> {
    None
}
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};
use common::sentinel::SentinelBuffer;

static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());

const SKALER_BUYUK_8_DCT: [TxSize; 8] = [
    TxSize::Rtx16x32,
    TxSize::Rtx32x16,
    TxSize::Tx32x32,
    TxSize::Rtx16x64,
    TxSize::Rtx32x64,
    TxSize::Rtx64x16,
    TxSize::Rtx64x32,
    TxSize::Tx64x64,
];

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

fn avx2_12_tablo(
    test: &str,
) -> Option<(DspDispatch, [[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES])> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let dsp = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()));
        if dsp.avx2_token().is_some() {
            let tablo = dsp.for_bit_depth(BitDepth::Twelve).itx().itxfm_add;
            return Some((dsp, tablo));
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P44] 12 bpc ITX AVX2 gövdeleri x86_64 AVX2 ister",
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
    if yerel.join("itx-large-12-input.bin").exists()
        && yerel.join("itx-large-12-expect.bin").exists()
    {
        return Some(yerel);
    }
    if let Ok(env_dir) = std::env::var("ITX_ORACLE_DIR") {
        let p = PathBuf::from(env_dir);
        if p.join("itx-large-12-input.bin").exists() && p.join("itx-large-12-expect.bin").exists() {
            return Some(p);
        }
    }
    let ana_depo = PathBuf::from("/home/hakanbiris/github/av1oxide/.cache/itx-oracle");
    if ana_depo.join("itx-large-12-input.bin").exists()
        && ana_depo.join("itx-large-12-expect.bin").exists()
    {
        return Some(ana_depo);
    }
    None
}

fn dogrula_oracle_12bpc(
    dir: &Path,
    profile_name: &str,
    dsp: &DspDispatch,
    tablo: &[[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES],
) -> (usize, usize, usize, usize) {
    let in_path = dir.join(format!("itx-{profile_name}-12-input.bin"));
    let exp_path = dir.join(format!("itx-{profile_name}-12-expect.bin"));
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
    assert_eq!(read_u8(&mut inp), 12);
    let profile = read_u8(&mut inp);
    assert_eq!(read_u16(&mut inp), 0);
    let count = read_u32(&mut inp);

    exp.read_exact(&mut magic).expect("exp magic");
    assert_eq!(&magic, b"ITXE");
    assert_eq!(read_u32(&mut exp), 1);
    assert_eq!(read_u8(&mut exp), 12);
    assert_eq!(read_u8(&mut exp), profile);
    assert_eq!(read_u16(&mut exp), 0);
    assert_eq!(read_u32(&mut exp), count);

    let token = dsp.avx2_token().expect("Avx2Token olmalı");
    let mut toplam = 0usize;
    let mut wht_disi = 0usize;
    let mut avx2_sayisi = 0usize;
    let mut skaler_buyuk_sayisi = 0usize;
    let mut coeff = [0i32; 1024];

    for rec in 0..u64::from(count) {
        let tx_u8 = read_u8(&mut inp);
        let txtp_u8 = read_u8(&mut inp);
        let rec_bpc = read_u8(&mut inp);
        assert_eq!(read_u8(&mut inp), 0);
        let eob = usize::from(read_u16(&mut inp));
        assert_eq!(read_u16(&mut inp), 0);
        assert_eq!(rec_bpc, 12);

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
            *v = tail_value(idx, rec, 12);
        }

        let e_tx = read_u8(&mut exp);
        let e_txtp = read_u8(&mut exp);
        let e_bpc = read_u8(&mut exp);
        assert_eq!(read_u8(&mut exp), 0);
        let e_eob = usize::from(read_u16(&mut exp));
        assert_eq!(read_u16(&mut exp), 0);
        assert_eq!((e_tx, e_txtp, e_bpc, e_eob), (tx_u8, txtp_u8, 12, eob));

        let mut exp_dst = vec![0u8; w * h * 2];
        for chunk in exp_dst.as_chunks_mut::<2>().0 {
            let v = read_u16(&mut exp);
            assert!(v <= 4095);
            *chunk = v.to_le_bytes();
        }

        let mut dst = vec![0u8; w * h * 2];
        let mut st = pix_seed(rec, 12);
        let variant = if profile == 0 { (rec % 6) as u8 } else { 0xff };
        for chunk in dst.as_chunks_mut::<2>().0 {
            let v = pix_next(&mut st) & 4095;
            let px = if profile == 0 && (variant == 1 || variant == 5) {
                4095u16
            } else if profile == 0 && (variant == 0 || variant == 2) {
                0u16
            } else {
                v as u16
            };
            *chunk = px.to_le_bytes();
        }

        let f = tablo[tx.index()][txtp.index()];
        f(&mut dst, w * 2, &mut coeff, eob, tx, txtp, BitDepth::Twelve);

        for (i, &c) in coeff[..sw * sh].iter().enumerate() {
            assert_eq!(
                c, 0,
                "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) coeff[{i}] != 0"
            );
        }
        for (i, &c) in coeff.iter().enumerate().skip(sw * sh) {
            assert_eq!(
                c,
                tail_value(i, rec, 12),
                "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) kuyruk coeff[{i}] bozuldu"
            );
        }
        assert_eq!(
            dst, exp_dst,
            "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) 12 bpc piksel çıktısı oracle'dan farklı"
        );

        toplam += 1;
        if txtp != TxType::WhtWht {
            wht_disi += 1;
            if itx12_avx2_lookup(token, tx, txtp)
                .or_else(|| itx12_buyuk_avx2_lookup(token, tx, txtp))
                .is_some()
            {
                avx2_sayisi += 1;
            } else {
                skaler_buyuk_sayisi += 1;
            }
        }
    }
    (toplam, wht_disi, avx2_sayisi, skaler_buyuk_sayisi)
}

#[test]
fn test_itx12_avx2_slot_install_and_8_large_dct_remain_scalar() {
    const T: &str = "itx16_12_avx2::test_itx12_avx2_slot_install_and_8_large_dct_remain_scalar";
    let Some((dsp, _)) = avx2_12_tablo(T) else {
        return;
    };
    let token = dsp.avx2_token().expect("Avx2Token olmalı");

    let mut avx2_kucuk = 0usize;
    let mut avx2_buyuk = 0usize;
    let mut skaler_buyuk_dct = 0usize;

    for &(tx, txtp) in &wht_disi_155_hucre() {
        let alan = format!("itxfm_add[{}][{}]", tx.dav1d_name(), txtp.dav1d_name());
        let impl12 = dsp.slot_implementation(BitDepth::Twelve, "itx", &alan);

        if tx.width() <= 16 && tx.height() <= 16 {
            assert_eq!(
                impl12, "avx2",
                "CpuLevel::Avx2 altında 12 bpc küçük {alan} yuvası avx2 olmalı"
            );
            assert!(itx12_avx2_lookup(token, tx, txtp).is_some());
            assert!(itx12_buyuk_avx2_lookup(token, tx, txtp).is_none());
            avx2_kucuk += 1;
        } else if txtp == TxType::DctDct && SKALER_BUYUK_8_DCT.contains(&tx) {
            assert_eq!(
                impl12, "skaler",
                "dav1d src/x86/itx.h:249-263 paritesi: 12 bpc en büyük 8 DCT_DCT ({alan}) skaler kalmalı"
            );
            assert!(itx12_avx2_lookup(token, tx, txtp).is_none());
            assert!(itx12_buyuk_avx2_lookup(token, tx, txtp).is_none());
            skaler_buyuk_dct += 1;
        } else {
            assert_eq!(
                impl12, "avx2",
                "CpuLevel::Avx2 altında 12 bpc büyük {alan} yuvası avx2 olmalı"
            );
            assert!(itx12_avx2_lookup(token, tx, txtp).is_none());
            assert!(itx12_buyuk_avx2_lookup(token, tx, txtp).is_some());
            avx2_buyuk += 1;
        }
    }

    assert_eq!(
        dsp.slot_implementation(BitDepth::Twelve, "itx", "itxfm_add[TX_4X4][WHT_WHT]"),
        "avx2",
        "12 bpc WHT_WHT yuvası CpuLevel::Avx2 altında avx2 olmalı"
    );
    assert_eq!(
        avx2_kucuk, 140,
        "140 küçük (w, h <= 16) 12 bpc yuva avx2 olmalı"
    );
    assert_eq!(
        avx2_buyuk, 7,
        "7 büyük (8x32/32x8 DCT+IDTX, 16x32/32x16/32x32 IDTX) 12 bpc yuva avx2 olmalı"
    );
    assert_eq!(
        avx2_kucuk + avx2_buyuk,
        147,
        "toplam 147 WHT-dışı 12 bpc yuva avx2 olmalı"
    );
    assert_eq!(
        skaler_buyuk_dct, 8,
        "en büyük 8 DCT_DCT 12 bpc'de skalerde kalmalı"
    );
}

#[test]
fn test_itx12_avx2_c_oracle_small_vectors() {
    const T: &str = "itx16_12_avx2::test_itx12_avx2_c_oracle_small_vectors";
    let Some((dsp, tablo)) = avx2_12_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let small_dir = manifest.join("tests/fixtures/itx");
    let (toplam, wht_disi, avx2_sayisi, skaler_buyuk_sayisi) =
        dogrula_oracle_12bpc(&small_dir, "small", &dsp, &tablo);
    assert_eq!(toplam, 936);
    assert_eq!(wht_disi, 930);
    assert_eq!(
        avx2_sayisi, 882,
        "147 * 6 = 882 AVX2 12 bpc small vektörü doğrulanmalı"
    );
    assert_eq!(
        skaler_buyuk_sayisi, 48,
        "8 * 6 = 48 skaler büyük DCT_DCT small vektörü doğrulanmalı"
    );
    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "12 bpc small oracle koşusunda sözleşme dışı skaler devir 0 olmalı"
    );
}

#[test]
fn test_itx12_avx2_c_oracle_large_vectors() {
    const T: &str = "itx16_12_avx2::test_itx12_avx2_c_oracle_large_vectors";
    let Some((dsp, tablo)) = avx2_12_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let large_dir = buyuk_oracle_dizini(&manifest)
        .expect(".cache/itx-oracle/itx-large-12-{input,expect}.bin bulunmalı");
    let (toplam, wht_disi, avx2_sayisi, skaler_buyuk_sayisi) =
        dogrula_oracle_12bpc(&large_dir, "large", &dsp, &tablo);
    assert_eq!(toplam, 156_000);
    assert_eq!(wht_disi, 155_000);
    assert_eq!(
        avx2_sayisi, 147_000,
        "147 * 1000 = 147 000 AVX2 12 bpc large vektörü doğrulanmalı"
    );
    assert_eq!(
        skaler_buyuk_sayisi, 8_000,
        "8 * 1000 = 8 000 skaler büyük DCT_DCT large vektörü doğrulanmalı"
    );
    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "12 bpc large oracle koşusunda sözleşme dışı skaler devir 0 olmalı"
    );
}

#[test]
fn test_itx12_avx2_sentinel_and_contract_fallback() {
    const T: &str = "itx16_12_avx2::test_itx12_avx2_sentinel_and_contract_fallback";
    let Some((dsp, tablo)) = avx2_12_tablo(T) else {
        return;
    };
    let token = dsp.avx2_token().expect("Avx2Token olmalı");
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();
    let skaler8 = ItxDsp::scalar::<8>().itxfm_add;
    let skaler10 = ItxDsp::scalar::<10>().itxfm_add;
    let skaler12 = ItxDsp::scalar::<12>().itxfm_add;
    let mut rng = Rng(0x5044_0012_2026_1117);

    // 1) 147 AVX2 12 bpc yuvasının tamamında SentinelBuffer + çift bayt stride + DC-only coeff[1..] koruma
    for (tx, txtp) in wht_disi_155_hucre() {
        let Some(f) =
            itx12_avx2_lookup(token, tx, txtp).or_else(|| itx12_buyuk_avx2_lookup(token, tx, txtp))
        else {
            continue;
        };
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let row_bytes = w * 2;
        let f_ref = skaler12[tx.index()][txtp.index()];

        for &stride in &[row_bytes, row_bytes + 2, row_bytes + 6, row_bytes + 14] {
            let data_len = (h - 1) * stride + row_bytes;
            for &eob in &[0usize, 1.min(n - 1), 35.min(n - 1), 135.min(n - 1), n - 1] {
                let mut c_temel = [0i32; 1024];
                if eob <= 1 && eob < n - 1 {
                    c_temel[0] = rng.aralik(-65_536, 65_536);
                } else if eob == n - 1 {
                    let lim = (65_536 / (sw.max(sh) as i32)).max(1_024);
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
                    let lim = 4_096;
                    for x in 0..sw {
                        for y in 0..sh {
                            let gecerli = if yatay_id {
                                x + y * sw <= eob
                            } else if dikey_id {
                                y + x * sh <= eob
                            } else if eob <= 35 {
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
                        let px = (rng.aralik(0, 4095) as u16).to_le_bytes();
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

                f(
                    s0.data_mut(),
                    stride,
                    &mut c0,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Twelve,
                );
                f(
                    s1.data_mut(),
                    stride,
                    &mut c1,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Twelve,
                );
                f_ref(
                    s_ref.data_mut(),
                    stride,
                    &mut c_ref,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Twelve,
                );

                assert_eq!(
                    s0.verify(),
                    Ok(()),
                    "{tx:?}/{txtp:?} stride={stride} eob={eob} 0x00 sentinel bozuldu"
                );
                assert_eq!(
                    s1.verify(),
                    Ok(()),
                    "{tx:?}/{txtp:?} stride={stride} eob={eob} 0xFF sentinel bozuldu"
                );
                assert_eq!(
                    s0.data(),
                    s_ref.data(),
                    "{tx:?}/{txtp:?} stride={stride} eob={eob} AVX2 != 12 bpc skaler"
                );
                for y in 0..h {
                    assert_eq!(
                        &s0.data()[y * stride..y * stride + row_bytes],
                        &s1.data()[y * stride..y * stride + row_bytes],
                        "{tx:?}/{txtp:?} stride={stride} y={y}: çıktı dolgu baytına bağlı (over-read)"
                    );
                    if y + 1 < h {
                        for b in row_bytes..stride {
                            assert_eq!(s0.data()[y * stride + b], 0x00);
                            assert_eq!(s1.data()[y * stride + b], 0xFF);
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "sözleşme içi 12 bpc AVX2 çağrılarında skaler devir 0 olmalı"
    );

    // 2) Sözleşme dışı çağrılarda skaler devir (`BitDepth::Ten`, `BitDepth::Eight`, tek bayt stride, kısa dilim, yanlış tx/txtp)
    for &(tx, txtp) in &[
        (TxSize::Tx4x4, TxType::DctDct),
        (TxSize::Rtx8x16, TxType::FlipadstAdst),
        (TxSize::Tx16x16, TxType::AdstAdst),
        (TxSize::Rtx8x32, TxType::DctDct),
        (TxSize::Rtx8x32, TxType::Idtx),
        (TxSize::Rtx32x8, TxType::DctDct),
        (TxSize::Tx32x32, TxType::Idtx),
    ] {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let stride = w * 2;
        let f = tablo[tx.index()][txtp.index()];

        // 2a) Yanlış `tx`
        let baska_tx = if tx == TxSize::Tx4x4 {
            TxSize::Tx8x8
        } else {
            TxSize::Tx4x4
        };
        let bw = baska_tx.width();
        let bh = baska_tx.height();
        let mut dst_simd = vec![0u8; bw * bh * 2];
        for chunk in dst_simd.as_chunks_mut::<2>().0 {
            *chunk = 1_500u16.to_le_bytes();
        }
        let mut dst_ref = dst_simd.clone();
        let mut c_simd = [0i32; 1024];
        c_simd[0] = 4_096;
        let mut c_ref = c_simd;
        let d0 = itx_skaler_devir_sayisi();
        f(
            &mut dst_simd,
            bw * 2,
            &mut c_simd,
            0,
            baska_tx,
            TxType::DctDct,
            BitDepth::Twelve,
        );
        (skaler12[baska_tx.index()][TxType::DctDct.index()])(
            &mut dst_ref,
            bw * 2,
            &mut c_ref,
            0,
            baska_tx,
            TxType::DctDct,
            BitDepth::Twelve,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d0 + 1);
        assert_eq!(dst_simd, dst_ref);

        // 2b) Yanlış `txtp`
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
                BitDepth::Twelve,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d_t + 1);
        }

        // 2c) `BitDepth::Ten` ile çağrı: sayaç +1 ve 10 bpc skaler monomorfla aynı
        let mut dst10_simd = vec![0u8; w * h * 2];
        for chunk in dst10_simd.as_chunks_mut::<2>().0 {
            *chunk = 600u16.to_le_bytes();
        }
        let mut dst10_ref = dst10_simd.clone();
        let mut c10_simd = [0i32; 1024];
        c10_simd[0] = 2_048;
        let mut c10_ref = c10_simd;
        let d10 = itx_skaler_devir_sayisi();
        f(
            &mut dst10_simd,
            stride,
            &mut c10_simd,
            0,
            tx,
            txtp,
            BitDepth::Ten,
        );
        (skaler10[tx.index()][txtp.index()])(
            &mut dst10_ref,
            stride,
            &mut c10_ref,
            0,
            tx,
            txtp,
            BitDepth::Ten,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d10 + 1);
        assert_eq!(dst10_simd, dst10_ref);

        // 2d) `BitDepth::Eight` ile çağrı: sayaç +1 ve 8 bpc skaler monomorfla aynı
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

        // 2e) Tek bayt `dst_stride`, dar `dst_stride`, geçersiz `eob`, kısa `dst`/`coeff`
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
            BitDepth::Twelve,
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
            BitDepth::Twelve,
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
                BitDepth::Twelve,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d + 1);
        }

        let mut kisa_dst = vec![0u8; (h - 1) * stride + stride - 1];
        let d_kisa_dst = itx_skaler_devir_sayisi();
        f(
            &mut kisa_dst,
            stride,
            &mut c_ok,
            0,
            tx,
            txtp,
            BitDepth::Twelve,
        );
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
            BitDepth::Twelve,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d_kisa_c + 1);
    }
}

#[test]
fn test_itx12_avx2_extreme_saturation_parity() {
    const T: &str = "itx16_12_avx2::test_itx12_avx2_extreme_saturation_parity";
    let Some((dsp, _)) = avx2_12_tablo(T) else {
        return;
    };
    let token = dsp.avx2_token().expect("Avx2Token olmalı");
    let skaler12 = ItxDsp::scalar::<12>().itxfm_add;
    let mut rng = Rng(0xDEAD_BEEF_2026_1244);

    for (tx, txtp) in wht_disi_155_hucre() {
        let Some(f) =
            itx12_avx2_lookup(token, tx, txtp).or_else(|| itx12_buyuk_avx2_lookup(token, tx, txtp))
        else {
            continue;
        };
        let f_ref = skaler12[tx.index()][txtp.index()];
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let row_bytes = w * 2;
        let stride = row_bytes + 6;
        let data_len = (h - 1) * stride + row_bytes;

        for kip in 0..8u8 {
            let mut coeff_simd = [0i32; 1024];
            for (i, v) in coeff_simd.iter_mut().enumerate().take(n) {
                *v = match kip {
                    0 => 524_287,
                    1 => -524_287,
                    2 => -524_288,
                    3 => {
                        if i & 1 == 0 {
                            524_287
                        } else {
                            -524_288
                        }
                    }
                    4 => i32::MAX,
                    5 => i32::MIN,
                    6 => {
                        if (i % 3) == 0 {
                            524_287
                        } else if (i % 3) == 1 {
                            -524_288
                        } else {
                            131_071
                        }
                    }
                    _ => rng.aralik(-524_288, 524_287),
                };
            }
            for v in coeff_simd.iter_mut().skip(n) {
                *v = 0x4242_1212;
            }
            let eob = if kip == 0 {
                coeff_simd[1..n].fill(0);
                0
            } else {
                n - 1
            };
            let mut coeff_ref = coeff_simd;

            let mut sb_simd = SentinelBuffer::with_defaults(data_len);
            let mut sb_ref = SentinelBuffer::with_defaults(data_len);
            sb_simd.data_mut().fill(0x5A);
            sb_ref.data_mut().fill(0x5A);
            for y in 0..h {
                for x in 0..w {
                    let px = match (x + y + usize::from(kip)) & 3 {
                        0 => 0u16,
                        1 => 4095u16,
                        2 => 2048u16,
                        _ => (rng.aralik(0, 4095)) as u16,
                    };
                    let off = y * stride + x * 2;
                    sb_simd.data_mut()[off..off + 2].copy_from_slice(&px.to_le_bytes());
                    sb_ref.data_mut()[off..off + 2].copy_from_slice(&px.to_le_bytes());
                }
            }

            f(
                sb_simd.data_mut(),
                stride,
                &mut coeff_simd,
                eob,
                tx,
                txtp,
                BitDepth::Twelve,
            );
            f_ref(
                sb_ref.data_mut(),
                stride,
                &mut coeff_ref,
                eob,
                tx,
                txtp,
                BitDepth::Twelve,
            );

            assert_eq!(
                sb_simd.verify(),
                Ok(()),
                "{tx:?}/{txtp:?} kip={kip}: uç 12 bpc katsayıda sentinel ihlali"
            );
            assert_eq!(
                sb_simd.data(),
                sb_ref.data(),
                "{tx:?}/{txtp:?} kip={kip} eob={eob}: uç 12 bpc katsayıda AVX2 != skaler"
            );
            assert_eq!(
                coeff_simd, coeff_ref,
                "{tx:?}/{txtp:?} kip={kip} eob={eob}: uç 12 bpc katsayıda coeff sıfırlama farkı"
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_itx12_avx2_isa_boundary_and_stack_frame_objdump() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let mut kucuk_cekirdek = 0usize;
    let mut buyuk_cekirdek = 0usize;
    let mut aktif = false;
    let mut aktif_sembol = String::new();
    let mut max_rsp_sub = 0u64;
    let mut max_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") {
            let ilgili = satir.contains("itx16_12") || satir.contains("itx16_12_buyuk");
            let uretim = ilgili && satir.contains("dsp") && satir.contains("x86");
            aktif = uretim;
            if uretim {
                aktif_sembol = satir.to_string();
                if satir.contains("itx12_avx2_cekirdek") {
                    kucuk_cekirdek += 1;
                } else if satir.contains("itx12_buyuk_avx2_cekirdek") {
                    buyuk_cekirdek += 1;
                }
            }
            continue;
        }
        if satir.is_empty() {
            aktif = false;
            continue;
        }
        if !aktif {
            continue;
        }
        let Some((_, sag)) = satir.split_once('\t') else {
            continue;
        };
        assert!(
            !sag.contains("%zmm") && !sag.contains("{%k"),
            "AVX2 12 bpc sembolünde EVEX (ZMM/maske) yönergesi: {aktif_sembol} -> {satir}"
        );
        if !cfg!(debug_assertions)
            && let Some(pos) = sag.find("sub")
        {
            let kalan = sag[pos + 3..].trim();
            if let Some(son) = kalan.strip_suffix(",%rsp")
                && let Some(hex) = son.trim().strip_prefix("$0x")
                && let Ok(deger) = u64::from_str_radix(hex, 16)
            {
                assert!(
                    deger < 0x400,
                    "yığın çerçevesi 0x400 sınırını aştı ({aktif_sembol}): sub $0x{deger:x}, %rsp"
                );
                if deger > max_rsp_sub {
                    max_rsp_sub = deger;
                    max_sembol = aktif_sembol.clone();
                }
            }
        }
    }

    assert!(
        kucuk_cekirdek >= 140,
        "en az 140 itx12_avx2_cekirdek sembolü bulunmalı: {kucuk_cekirdek}"
    );
    assert!(
        buyuk_cekirdek >= 7,
        "en az 7 itx12_buyuk_avx2_cekirdek sembolü bulunmalı: {buyuk_cekirdek}"
    );
    assert!(
        max_rsp_sub < 0x400,
        "max_rsp_sub = 0x{max_rsp_sub:x} >= 0x400 ({max_sembol})"
    );
    println!(
        "[P44 ITX12 AVX2 objdump] itx16_12={kucuk_cekirdek}, itx16_12_buyuk={buyuk_cekirdek}, max_rsp_sub=0x{max_rsp_sub:x} (< 0x400), 0 EVEX"
    );
}
