//! P43 (F4-ITX b, Adım 2) — 8 bpc WHT-dışı SSSE3 ters dönüşüm (`155` kayıt) testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **Yuva kurulumu (`155` SSSE3 8 bpc kaydı):** `CpuLevel::Ssse3` düzeyinde
//!    `itx_ssse3` (`140` kayıt, `w, h ≤ 16`) ve `itx_buyuk_ssse3` (`15` kayıt,
//!    `max(w, h) ∈ {32, 64}`) yuvalarının tamamı `"ssse3"` olarak kurulur.
//! 2. **Oracle bit-birebirliği (`itx-small-8` + `itx-large-8`):** `930` küçük
//!    WHT-dışı blok (`936` toplam) ve `.cache/itx-oracle/itx-large-8` altındaki
//!    `155 000` büyük WHT-dışı blok (`156 000` toplam) üzerinde hedef pikseller
//!    ve çağrı sonrası `coeff[0..1024]` durumu C oracle ile **%100 bit-birebir**
//!    aynıdır (`0` skaler devir).
//! 3. **Katsayı sıfırlama sözleşmesi:**
//!    - `txtp == TxType::DctDct && eob == 0` (DC-only): **yalnız `coeff[0] = 0`**
//!      sıfırlanır; `coeff[1..1024]` dokunulmadan kalır.
//!    - Genel yol (`eob ≥ 1` ya da `txtp != TxType::DctDct`): alt-kadran `eob`
//!      hızlı yolu seçilse bile `coeff[..sw·sh]` sıfırlanır ve `coeff[sw·sh..1024]`
//!      dokunulmadan kalır.
//! 4. **`SentinelBuffer` sınır ve tek-adımlı `stride` (`w`, `w+1`, `w+7`, `w+13`)
//!    denetimi:** ön/arka nöbetçiler ve satır sonu dolgu baytları (`0x00`/`0xFF`)
//!    korunur (`0` taşma okuma/yazma), çıktı skaler referansla bit-birebirdir.
//! 5. **Sözleşme dışı skaler devir:** yanlış `(tx, txtp, bit_depth)`, `eob ≥ sw·sh`,
//!    `dst_stride < w` ya da kısa `dst`/`coeff` diliminde paniklemeden
//!    [`itx_skaler_devir_sayisi`] artırılır.
//! 6. **Bozuk/keyfî `±32767` (`-32768`, `i32::MIN`, `i32::MAX` dâhil) katsayı
//!    taraması:** `--profile robust` taşma denetimleri açıkken bile `155` yuvanın
//!    hiçbirinde panik oluşmaz (`0` panik).
//! 7. **ISA saflığı (`objdump -d`):** bütün `itx_ssse3` ve `itx_buyuk_ssse3`
//!    sembollerinde `0` SSE4.1+ ve `0` VEX/EVEX (`v*`, `%ymm`, `%zmm`, `{%k`) komutu.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

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

fn ssse3_tablo(
    test: &str,
) -> Option<(DspDispatch, [[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES])> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let dsp = DspDispatch::select(detected.intersection(CpuLevel::Ssse3.allowed()));
        if dsp.ssse3_token().is_some() {
            let tablo = dsp.for_bit_depth(BitDepth::Eight).itx().itxfm_add;
            return Some((dsp, tablo));
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P43] 8 bpc ITX SSSE3 gövdeleri x86_64 SSSE3 ister",
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

fn read_i16<R: Read>(r: &mut R) -> i16 {
    i16::from_le_bytes([read_u8(r), read_u8(r)])
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
    if yerel.join("itx-large-8-input.bin").exists() && yerel.join("itx-large-8-expect.bin").exists()
    {
        return Some(yerel);
    }
    if let Ok(env_dir) = std::env::var("ITX_ORACLE_DIR") {
        let p = PathBuf::from(env_dir);
        if p.join("itx-large-8-input.bin").exists() && p.join("itx-large-8-expect.bin").exists() {
            return Some(p);
        }
    }
    let ana_depo = PathBuf::from("/home/hakanbiris/github/av1oxide/.cache/itx-oracle");
    if ana_depo.join("itx-large-8-input.bin").exists()
        && ana_depo.join("itx-large-8-expect.bin").exists()
    {
        return Some(ana_depo);
    }
    None
}

fn dogrula_oracle_8bpc(
    dir: &Path,
    profile_name: &str,
    tablo: &[[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES],
) -> (usize, usize) {
    let in_path = dir.join(format!("itx-{profile_name}-8-input.bin"));
    let exp_path = dir.join(format!("itx-{profile_name}-8-expect.bin"));
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
    assert_eq!(read_u8(&mut inp), 8);
    let profile = read_u8(&mut inp);
    assert_eq!(read_u16(&mut inp), 0);
    let count = read_u32(&mut inp);

    exp.read_exact(&mut magic).expect("exp magic");
    assert_eq!(&magic, b"ITXE");
    assert_eq!(read_u32(&mut exp), 1);
    assert_eq!(read_u8(&mut exp), 8);
    assert_eq!(read_u8(&mut exp), profile);
    assert_eq!(read_u16(&mut exp), 0);
    assert_eq!(read_u32(&mut exp), count);

    let mut toplam = 0usize;
    let mut wht_disi = 0usize;
    let mut coeff = [0i32; 1024];

    for rec in 0..u64::from(count) {
        let tx_u8 = read_u8(&mut inp);
        let txtp_u8 = read_u8(&mut inp);
        let rec_bpc = read_u8(&mut inp);
        assert_eq!(read_u8(&mut inp), 0);
        let eob = usize::from(read_u16(&mut inp));
        assert_eq!(read_u16(&mut inp), 0);
        assert_eq!(rec_bpc, 8);

        let tx = TxSize::from_index(usize::from(tx_u8)).expect("tx");
        let txtp = TxType::from_index(usize::from(txtp_u8)).expect("txtp");
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);

        for v in coeff.iter_mut().take(sw * sh) {
            *v = i32::from(read_i16(&mut inp));
        }
        for (idx, v) in coeff.iter_mut().enumerate().skip(sw * sh) {
            *v = tail_value(idx, rec, 8);
        }

        let e_tx = read_u8(&mut exp);
        let e_txtp = read_u8(&mut exp);
        let e_bpc = read_u8(&mut exp);
        assert_eq!(read_u8(&mut exp), 0);
        let e_eob = usize::from(read_u16(&mut exp));
        assert_eq!(read_u16(&mut exp), 0);
        assert_eq!((e_tx, e_txtp, e_bpc, e_eob), (tx_u8, txtp_u8, 8, eob));

        let mut exp_px = vec![0u8; w * h];
        for p in &mut exp_px {
            let v = read_u16(&mut exp);
            assert!(v <= 255);
            *p = v as u8;
        }

        let mut dst = vec![0u8; w * h];
        let mut st = pix_seed(rec, 8);
        let variant = if profile == 0 { (rec % 6) as u8 } else { 0xff };
        for p in &mut dst {
            let v = pix_next(&mut st) & 255;
            *p = if profile == 0 && (variant == 1 || variant == 5) {
                255
            } else if profile == 0 && (variant == 0 || variant == 2) {
                0
            } else {
                v as u8
            };
        }

        let f = tablo[tx.index()][txtp.index()];
        f(&mut dst, w, &mut coeff, eob, tx, txtp, BitDepth::Eight);

        for (i, &c) in coeff[..sw * sh].iter().enumerate() {
            assert_eq!(
                c, 0,
                "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) coeff[{i}] != 0"
            );
        }
        for (i, &c) in coeff.iter().enumerate().skip(sw * sh) {
            assert_eq!(
                c,
                tail_value(i, rec, 8),
                "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) kuyruk coeff[{i}] bozuldu"
            );
        }
        assert_eq!(
            dst, exp_px,
            "{profile_name} rec={rec} ({tx:?}/{txtp:?} eob={eob}) piksel çıktısı oracle'dan farklı"
        );

        toplam += 1;
        if txtp != TxType::WhtWht {
            wht_disi += 1;
        }
    }
    (toplam, wht_disi)
}

#[test]
fn p43_itx_ssse3_155_yuva_kurulumu_tam() {
    const T: &str = "itx_ssse3::p43_itx_ssse3_155_yuva_kurulumu_tam";
    let Some((dsp, _)) = ssse3_tablo(T) else {
        return;
    };
    let hucreler = wht_disi_155_hucre();
    for &(tx, txtp) in &hucreler {
        let alan = format!("itxfm_add[{}][{}]", tx.dav1d_name(), txtp.dav1d_name());
        assert_eq!(
            dsp.slot_implementation(BitDepth::Eight, "itx", &alan),
            "ssse3",
            "8 bpc {alan} yuvası ssse3 olmalı"
        );
        assert_eq!(
            dsp.slot_implementation(BitDepth::Ten, "itx", &alan),
            "skaler",
            "10 bpc {alan} bu adımda skaler kalmalı"
        );
        assert_eq!(
            dsp.slot_implementation(BitDepth::Twelve, "itx", &alan),
            "skaler",
            "12 bpc {alan} bu adımda skaler kalmalı"
        );
    }
    assert_eq!(
        dsp.slot_implementation(BitDepth::Eight, "itx", "itxfm_add[TX_4X4][WHT_WHT]"),
        "sse2"
    );
}

#[test]
fn p43_itx_ssse3_oracle_kucuk_ve_buyuk_kume_bit_exact() {
    const T: &str = "itx_ssse3::p43_itx_ssse3_oracle_kucuk_ve_buyuk_kume_bit_exact";
    let Some((_, tablo)) = ssse3_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let small_dir = manifest.join("tests/fixtures/itx");
    let (small_toplam, small_wht_disi) = dogrula_oracle_8bpc(&small_dir, "small", &tablo);
    assert_eq!(small_toplam, 936);
    assert_eq!(small_wht_disi, 930);

    let mut large_wht_disi = 0usize;
    if let Some(large_dir) = buyuk_oracle_dizini(&manifest) {
        let (large_toplam, wht_disi) = dogrula_oracle_8bpc(&large_dir, "large", &tablo);
        assert_eq!(large_toplam, 156_000);
        assert_eq!(wht_disi, 155_000);
        large_wht_disi = wht_disi;
    }

    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "oracle koşusunda skaler devir 0 olmalı"
    );
    println!(
        "[P43 ITX SSSE3 oracle] small={small_wht_disi}/930, large={large_wht_disi}/155000 WHT-dışı blok bit-exact"
    );
}

#[test]
fn p43_itx_ssse3_katsayi_sifirlama_sozlesmesi() {
    const T: &str = "itx_ssse3::p43_itx_ssse3_katsayi_sifirlama_sozlesmesi";
    let Some((_, tablo)) = ssse3_tablo(T) else {
        return;
    };
    let skaler_tablo = ItxDsp::scalar::<8>().itxfm_add;

    for (tx, txtp) in wht_disi_155_hucre() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let f = tablo[tx.index()][txtp.index()];
        let f_ref = skaler_tablo[tx.index()][txtp.index()];

        // 1) DC-only (`txtp == DctDct && eob == 0`): yalnız `coeff[0]` sıfırlanır,
        // `coeff[1..1024]` dokunulmadan kalır.
        if txtp == TxType::DctDct {
            let mut dst_simd = vec![120u8; w * h];
            let mut dst_ref = dst_simd.clone();
            let mut c_simd = [0i32; 1024];
            for (i, v) in c_simd.iter_mut().enumerate() {
                *v = 1000 + (i as i32) * 3;
            }
            c_simd[0] = -980;
            let beklenen_kuyruk = c_simd;

            let mut c_ref = [0i32; 1024];
            c_ref[0] = -980;

            f(&mut dst_simd, w, &mut c_simd, 0, tx, txtp, BitDepth::Eight);
            f_ref(&mut dst_ref, w, &mut c_ref, 0, tx, txtp, BitDepth::Eight);

            assert_eq!(dst_simd, dst_ref, "{tx:?}/DctDct DC-only piksel farkı");
            assert_eq!(c_simd[0], 0, "{tx:?}/DctDct DC-only coeff[0] != 0");
            assert_eq!(
                &c_simd[1..],
                &beklenen_kuyruk[1..],
                "{tx:?}/DctDct DC-only coeff[1..1024] dokunulmamalı"
            );
        }

        // 2) Genel yol (`eob >= 1` ve `txtp != DctDct` için `eob == 0` dâhil):
        // alt-kadran hızlı yolları seçilse bile `coeff[..sw*sh]` sıfırlanır,
        // `coeff[sw*sh..1024]` dokunulmadan kalır.
        let mut eob_adaylari = vec![1usize, n - 1];
        if txtp != TxType::DctDct {
            eob_adaylari.push(0);
        }
        for esik in [28usize, 35, 42, 43, 106, 135, 150, 278] {
            if esik < n {
                eob_adaylari.push(esik);
            }
        }
        for eob in eob_adaylari {
            let mut dst_simd = vec![90u8; w * h];
            let mut c_simd = [0i32; 1024];
            for (i, v) in c_simd.iter_mut().enumerate().take(n) {
                *v = if i == 0 { 512 } else { -17 + (i as i32 % 23) };
            }
            for (i, v) in c_simd.iter_mut().enumerate().skip(n) {
                *v = -12_345 + (i as i32) * 7;
            }
            let beklenen = c_simd;

            f(
                &mut dst_simd,
                w,
                &mut c_simd,
                eob,
                tx,
                txtp,
                BitDepth::Eight,
            );

            assert!(
                c_simd[..n].iter().all(|&v| v == 0),
                "{tx:?}/{txtp:?} eob={eob}: coeff[..{n}] tamamen sıfırlanmalı"
            );
            assert_eq!(
                &c_simd[n..],
                &beklenen[n..],
                "{tx:?}/{txtp:?} eob={eob}: coeff[{n}..1024] dokunulmamalı"
            );
        }
    }
}

#[test]
fn p43_itx_ssse3_sentinel_ve_tek_adimli_stride_denetimi() {
    const T: &str = "itx_ssse3::p43_itx_ssse3_sentinel_ve_tek_adimli_stride_denetimi";
    let Some((_, tablo)) = ssse3_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();
    let skaler_tablo = ItxDsp::scalar::<8>().itxfm_add;
    let mut rng = Rng(0x5043_0002_2026_1117);

    for (tx, txtp) in wht_disi_155_hucre() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let f = tablo[tx.index()][txtp.index()];
        let f_ref = skaler_tablo[tx.index()][txtp.index()];

        for &stride in &[w, w + 1, w + 7, w + 13] {
            let data_len = (h - 1) * stride + w;

            for &eob in &[0usize, 1.min(n - 1), 35.min(n - 1), 135.min(n - 1), n - 1] {
                let mut c_temel = [0i32; 1024];
                if eob <= 1 && eob < n - 1 {
                    c_temel[0] = rng.aralik(-4096, 4096);
                } else if eob == n - 1 {
                    let lim = (4096 / (sw.max(sh) as i32)).max(64);
                    for x in 0..sw {
                        for y in 0..sh {
                            c_temel[y + x * sh] = rng.aralik(-lim, lim);
                        }
                    }
                } else {
                    // Tarama sırası (`zigzag` / satır / sütun) içinde `scan_pos <= eob`
                    // sözleşmesini tüm `TxType`lar için kesin sağlayan alt-üçgen dolumu.
                    let (yatay_id, dikey_id) = match txtp {
                        TxType::VDct | TxType::VAdst | TxType::VFlipadst => (true, false),
                        TxType::HDct | TxType::HAdst | TxType::HFlipadst => (false, true),
                        _ => (false, false),
                    };
                    let lim = 256;
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
                        let px = rng.aralik(0, 255) as u8;
                        s0.data_mut()[y * stride + x] = px;
                        s1.data_mut()[y * stride + x] = px;
                        s_ref.data_mut()[y * stride + x] = px;
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
                    BitDepth::Eight,
                );
                f(
                    s1.data_mut(),
                    stride,
                    &mut c1,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Eight,
                );
                f_ref(
                    s_ref.data_mut(),
                    stride,
                    &mut c_ref,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Eight,
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
                    "{tx:?}/{txtp:?} stride={stride} eob={eob} SSSE3 != skaler (dolgu dâhil)"
                );

                for y in 0..h {
                    assert_eq!(
                        &s0.data()[y * stride..y * stride + w],
                        &s1.data()[y * stride..y * stride + w],
                        "{tx:?}/{txtp:?} stride={stride} y={y}: çıktı dolgu baytına bağlı (over-read)"
                    );
                    if y + 1 < h {
                        for b in w..stride {
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
        "sözleşme içi çağrılarda skaler devir 0 olmalı"
    );
}

#[test]
fn p43_itx_ssse3_sozlesme_disi_skalere_devreder() {
    const T: &str = "itx_ssse3::p43_itx_ssse3_sozlesme_disi_skalere_devreder";
    let Some((_, tablo)) = ssse3_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let skaler_tablo = ItxDsp::scalar::<8>().itxfm_add;

    for &(tx, txtp) in &[
        (TxSize::Tx4x4, TxType::DctDct),
        (TxSize::Rtx8x16, TxType::FlipadstAdst),
        (TxSize::Tx16x16, TxType::AdstAdst),
        (TxSize::Rtx8x32, TxType::Idtx),
        (TxSize::Tx32x32, TxType::DctDct),
        (TxSize::Tx64x64, TxType::DctDct),
    ] {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let f = tablo[tx.index()][txtp.index()];

        // 1) Yanlış `tx` ile çağrı: sayaç +1 ve geçerli skaler çıktı.
        let baska_tx = if tx == TxSize::Tx4x4 {
            TxSize::Tx8x8
        } else {
            TxSize::Tx4x4
        };
        let bw = baska_tx.width();
        let bh = baska_tx.height();
        let mut dst_simd = vec![100u8; bw * bh];
        let mut dst_ref = dst_simd.clone();
        let mut c_simd = [0i32; 1024];
        c_simd[0] = 320;
        let mut c_ref = c_simd;
        let d0 = itx_skaler_devir_sayisi();
        f(
            &mut dst_simd,
            bw,
            &mut c_simd,
            0,
            baska_tx,
            TxType::DctDct,
            BitDepth::Eight,
        );
        (skaler_tablo[baska_tx.index()][TxType::DctDct.index()])(
            &mut dst_ref,
            bw,
            &mut c_ref,
            0,
            baska_tx,
            TxType::DctDct,
            BitDepth::Eight,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d0 + 1);
        assert_eq!(dst_simd, dst_ref);

        // 2) Yanlış `bit_depth` (`Ten`): paniklemeden sayaç +1.
        let mut dst_ok = vec![80u8; w * h * 2];
        let mut c_ok = [0i32; 1024];
        let d1 = itx_skaler_devir_sayisi();
        f(&mut dst_ok, w * 2, &mut c_ok, 0, tx, txtp, BitDepth::Ten);
        assert_eq!(itx_skaler_devir_sayisi(), d1 + 1);

        // 3) `eob >= sw*sh` (`eob = sw*sh` ve `usize::MAX`): paniklemeden sayaç +1.
        let mut dst_eob = vec![50u8; w * h];
        for gecersiz_eob in [sw * sh, usize::MAX] {
            let d = itx_skaler_devir_sayisi();
            f(
                &mut dst_eob,
                w,
                &mut c_ok,
                gecersiz_eob,
                tx,
                txtp,
                BitDepth::Eight,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d + 1);
        }

        // 4) `dst_stride < w`: paniklemeden sayaç +1.
        let d2 = itx_skaler_devir_sayisi();
        f(&mut dst_eob, w - 1, &mut c_ok, 0, tx, txtp, BitDepth::Eight);
        assert_eq!(itx_skaler_devir_sayisi(), d2 + 1);

        // 5) Kısa `dst` dilimi: paniklemeden sayaç +1.
        let mut kisa_dst = vec![0u8; (h - 1) * w + w - 1];
        let d3 = itx_skaler_devir_sayisi();
        f(&mut kisa_dst, w, &mut c_ok, 0, tx, txtp, BitDepth::Eight);
        assert_eq!(itx_skaler_devir_sayisi(), d3 + 1);

        // 6) Kısa `coeff` dilimi (`< 1024`): paniklemeden sayaç +1.
        let mut kisa_coeff = [0i32; 512];
        let d4 = itx_skaler_devir_sayisi();
        f(
            &mut dst_eob,
            w,
            &mut kisa_coeff,
            0,
            tx,
            txtp,
            BitDepth::Eight,
        );
        assert_eq!(itx_skaler_devir_sayisi(), d4 + 1);
    }
}

#[test]
fn p43_itx_ssse3_bozuk_uc_katsayi_taramasi_paniksiz() {
    const T: &str = "itx_ssse3::p43_itx_ssse3_bozuk_uc_katsayi_taramasi_paniksiz";
    let Some((_, tablo)) = ssse3_tablo(T) else {
        return;
    };
    let mut rng = Rng(0xDEAD_BEEF_2026_0043);

    for (tx, txtp) in wht_disi_155_hucre() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let stride = w + 3;
        let data_len = (h - 1) * stride + w;
        let f = tablo[tx.index()][txtp.index()];

        for kip in 0..8u8 {
            let mut coeff = [0i32; 1024];
            for (i, v) in coeff.iter_mut().enumerate().take(n) {
                *v = match kip {
                    0 => 32767,
                    1 => -32767,
                    2 => -32768,
                    3 => {
                        if i & 1 == 0 {
                            32767
                        } else {
                            -32768
                        }
                    }
                    4 => i32::MAX,
                    5 => i32::MIN,
                    _ => rng.aralik(-32768, 32767),
                };
            }
            for v in coeff.iter_mut().skip(n) {
                *v = 0x4242;
            }
            let eob = match kip % 4 {
                0 => 0,
                1 => 1.min(n - 1),
                2 => (n / 3).min(n - 1),
                _ => n - 1,
            };

            let mut sb = SentinelBuffer::with_defaults(data_len);
            sb.data_mut().fill(0x5A);
            for y in 0..h {
                for x in 0..w {
                    sb.data_mut()[y * stride + x] = if (x + y) & 1 == 0 { 0 } else { 255 };
                }
            }

            f(
                sb.data_mut(),
                stride,
                &mut coeff,
                eob,
                tx,
                txtp,
                BitDepth::Eight,
            );

            assert_eq!(
                sb.verify(),
                Ok(()),
                "{tx:?}/{txtp:?} kip={kip}: bozuk katsayıda sentinel ihlali"
            );
            for y in 0..h.saturating_sub(1) {
                for b in w..stride {
                    assert_eq!(sb.data()[y * stride + b], 0x5A);
                }
            }
            if txtp == TxType::DctDct && eob == 0 {
                assert_eq!(coeff[0], 0);
            } else {
                assert!(coeff[..n].iter().all(|&v| v == 0));
            }
            assert!(coeff[n..].iter().all(|&v| v == 0x4242));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p43_itx_ssse3_isa_siniri_objdump_denetimi() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let yasak_sse41_ve_ustu = [
        "pmovzxbw",
        "pmovzxbd",
        "pmovzxbq",
        "pmovzxwd",
        "pmovzxwq",
        "pmovzxdq",
        "pmovsxbw",
        "pmovsxbd",
        "pmovsxbq",
        "pmovsxwd",
        "pmovsxwq",
        "pmovsxdq",
        "pblendvb",
        "pblendw",
        "blendps",
        "blendpd",
        "blendvps",
        "blendvpd",
        "packusdw",
        "pmulld",
        "pmuldq",
        "pinsrb",
        "pinsrd",
        "pinsrq",
        "pextrb",
        "pextrd",
        "pextrq",
        "insertps",
        "extractps",
        "pminsb",
        "pmaxsb",
        "pminuw",
        "pmaxuw",
        "pminud",
        "pmaxud",
        "pminsd",
        "pmaxsd",
        "ptest",
        "pcmpeqq",
        "pcmpgtq",
        "phminposuw",
        "mpsadbw",
        "roundps",
        "roundpd",
        "roundss",
        "roundsd",
        "dpps",
        "dppd",
        "movntdqa",
        "crc32",
        "popcnt",
    ];

    let mut kucuk_cekirdek = 0usize;
    let mut buyuk_cekirdek = 0usize;
    let mut aktif = false;
    let mut aktif_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") {
            let ilgili = satir.contains("itx_ssse3") || satir.contains("itx_buyuk_ssse3");
            // Test dosyasının kendi sembollerini (`itx_ssse3::p43_*` vb.) ayıkla; yalnız
            // `av1oxide::dsp::x86::itx_{buyuk_,}ssse3` üretim modülü sembollerini denetle.
            let uretim = ilgili && satir.contains("dsp") && satir.contains("x86");
            aktif = uretim;
            if uretim {
                aktif_sembol = satir.to_string();
                if satir.contains("itx8_ssse3_cekirdek") {
                    kucuk_cekirdek += 1;
                } else if satir.contains("itx8_buyuk_ssse3_cekirdek") {
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
        let Some(op) = sag.split_whitespace().next() else {
            continue;
        };
        assert!(
            !yasak_sse41_ve_ustu.contains(&op),
            "SSSE3 sembolünde SSE4.1+ yönergesi ({op}): {aktif_sembol} -> {satir}"
        );
        assert!(
            !op.starts_with('v') || op == "verw" || op == "verr",
            "SSSE3 sembolünde VEX/EVEX yönergesi ({op}): {aktif_sembol} -> {satir}"
        );
        assert!(
            !sag.contains("%ymm") && !sag.contains("%zmm") && !sag.contains("{%k"),
            "SSSE3 sembolünde YMM/ZMM/maske yazmacı: {aktif_sembol} -> {satir}"
        );
    }

    assert!(
        kucuk_cekirdek >= 140,
        "en az 140 itx8_ssse3_cekirdek sembolü bulunmalı: {kucuk_cekirdek}"
    );
    assert!(
        buyuk_cekirdek >= 15,
        "en az 15 itx8_buyuk_ssse3_cekirdek sembolü bulunmalı: {buyuk_cekirdek}"
    );
    println!(
        "[P43 ITX SSSE3 objdump] itx_ssse3={kucuk_cekirdek}, itx_buyuk_ssse3={buyuk_cekirdek} sembol temiz (0 SSE4.1+, 0 VEX/EVEX)"
    );
}
