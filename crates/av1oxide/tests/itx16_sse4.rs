//! P43 (F4-ITX b, Adım 3) — 10 bpc WHT-dışı SSE4.1 ters dönüşüm (`155` kayıt) testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **Yuva kurulumu ve 10/12 bpc paritesi (`dav1d` `src/x86/itx.h:178`):**
//!    `CpuLevel::Sse41` düzeyinde `BitDepth::Ten` için `itx16_sse4` (`140` kayıt,
//!    `w, h ≤ 16`) ve `itx16_buyuk_sse4` (`15` kayıt, `max(w, h) ∈ {32, 64}`)
//!    yuvalarının tamamı `"sse4"` olarak kurulurken, `BitDepth::Twelve` için `155`
//!    WHT-dışı yuvanın tamamı `"skaler"` kalır (`TX_4X4 × WHT_WHT` ise `"sse2"`).
//! 2. **Oracle bit-birebirliği (`itx-small-10` + `itx-large-10`):** `930` küçük
//!    WHT-dışı blok (`936` toplam) ve `.cache/itx-oracle/itx-large-10` altındaki
//!    `155 000` büyük WHT-dışı blok (`156 000` toplam) üzerinde hedef `u16` baytları
//!    ve çağrı sonrası `coeff[0..1024]` durumu C oracle ile **%100 bit-birebir**
//!    aynıdır (`0` skaler devir).
//! 3. **Katsayı sıfırlama sözleşmesi:**
//!    - `txtp == TxType::DctDct && eob == 0` (DC-only): **yalnız `coeff[0] = 0`**
//!      sıfırlanır; `coeff[1..1024]` dokunulmadan kalır.
//!    - Genel yol (`eob ≥ 1` ya da `txtp != TxType::DctDct`): alt-kadran `eob`
//!      hızlı yolu seçilse bile `coeff[..sw·sh]` sıfırlanır ve `coeff[sw·sh..1024]`
//!      dokunulmadan kalır.
//! 4. **`SentinelBuffer` sınır ve çift bayt `stride` (`w·2`, `w·2+2`, `w·2+6`,
//!    `w·2+14`) denetimi:** ön/arka nöbetçiler ve satır sonu dolgu baytları
//!    (`0x00`/`0xFF`) korunur (`0` taşma okuma/yazma), çıktı skaler 10 bpc
//!    referansla bit-birebirdir.
//! 5. **Sözleşme dışı skaler devir:** yanlış `(tx, txtp)`, `BitDepth::Twelve` ve
//!    `BitDepth::Eight`, tek sayı bayt `dst_stride` (`w·2 + 1`), `dst_stride < w·2`,
//!    `eob ≥ sw·sh` ya da kısa `dst`/`coeff` diliminde paniklemeden
//!    [`itx_skaler_devir_sayisi`] artırılır ve uygun skaler monomorfa devredilir.
//! 6. **Bozuk/keyfî `±131071` (`-131072`, `i32::MIN`, `i32::MAX` dâhil) katsayı
//!    taraması:** `--profile robust` taşma denetimleri açıkken bile `155` yuvanın
//!    hiçbirinde panik oluşmaz (`0` panik) ve tüm çıktı örnekleri `[0, 1023]`
//!    aralığında kalır.
//! 7. **ISA saflığı (`objdump -d`):** bütün `itx16_sse4` ve `itx16_buyuk_sse4`
//!    üretim sembollerinde `0` SSE4.2+ ve `0` VEX/EVEX (`v*`, `%ymm`, `%zmm`, `{%k`)
//!    yönergesi bulunur.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
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

fn sse4_tablo(test: &str) -> Option<(DspDispatch, [[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES])> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let dsp = DspDispatch::select(detected.intersection(CpuLevel::Sse41.allowed()));
        if dsp.sse41_token().is_some() {
            let tablo = dsp.for_bit_depth(BitDepth::Ten).itx().itxfm_add;
            return Some((dsp, tablo));
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P43] 10 bpc ITX SSE4.1 gövdeleri x86_64 SSE4.1 ister",
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
) -> (usize, usize) {
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
        }
    }
    (toplam, wht_disi)
}

#[test]
fn p43_itx10_sse4_155_yuva_ve_12bpc_skaler_paritesi() {
    const T: &str = "itx16_sse4::p43_itx10_sse4_155_yuva_ve_12bpc_skaler_paritesi";
    let Some((dsp, _)) = sse4_tablo(T) else {
        return;
    };
    let hucreler = wht_disi_155_hucre();
    for &(tx, txtp) in &hucreler {
        let alan = format!("itxfm_add[{}][{}]", tx.dav1d_name(), txtp.dav1d_name());
        assert_eq!(
            dsp.slot_implementation(BitDepth::Eight, "itx", &alan),
            "ssse3",
            "CpuLevel::Sse41 altında 8 bpc {alan} yuvası ssse3 olmalı"
        );
        assert_eq!(
            dsp.slot_implementation(BitDepth::Ten, "itx", &alan),
            "sse4",
            "CpuLevel::Sse41 altında 10 bpc {alan} yuvası sse4 olmalı"
        );
        assert_eq!(
            dsp.slot_implementation(BitDepth::Twelve, "itx", &alan),
            "skaler",
            "dav1d itx.h:178 paritesi: CpuLevel::Sse41 altında 12 bpc {alan} skaler kalmalı"
        );
    }
    for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
        assert_eq!(
            dsp.slot_implementation(bd, "itx", "itxfm_add[TX_4X4][WHT_WHT]"),
            "sse2",
            "{bd:?} WHT_WHT yuvası CpuLevel::Sse41 altında sse2 olmalı"
        );
    }
}

#[test]
fn p43_itx10_sse4_oracle_kucuk_ve_buyuk_kume_bit_exact() {
    const T: &str = "itx16_sse4::p43_itx10_sse4_oracle_kucuk_ve_buyuk_kume_bit_exact";
    let Some((_, tablo)) = sse4_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let small_dir = manifest.join("tests/fixtures/itx");
    let (small_toplam, small_wht_disi) = dogrula_oracle_10bpc(&small_dir, "small", &tablo);
    assert_eq!(small_toplam, 936);
    assert_eq!(small_wht_disi, 930);

    let mut large_wht_disi = 0usize;
    if let Some(large_dir) = buyuk_oracle_dizini(&manifest) {
        let (large_toplam, wht_disi) = dogrula_oracle_10bpc(&large_dir, "large", &tablo);
        assert_eq!(large_toplam, 156_000);
        assert_eq!(wht_disi, 155_000);
        large_wht_disi = wht_disi;
    }

    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "10 bpc oracle koşusunda skaler devir 0 olmalı"
    );
    println!(
        "[P43 ITX10 SSE4.1 oracle] small={small_wht_disi}/930, large={large_wht_disi}/155000 WHT-dışı blok bit-exact"
    );
}

#[test]
fn p43_itx10_sse4_katsayi_sifirlama_sozlesmesi() {
    const T: &str = "itx16_sse4::p43_itx10_sse4_katsayi_sifirlama_sozlesmesi";
    let Some((_, tablo)) = sse4_tablo(T) else {
        return;
    };
    let skaler_tablo = ItxDsp::scalar::<10>().itxfm_add;

    for (tx, txtp) in wht_disi_155_hucre() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let stride = w * 2;
        let f = tablo[tx.index()][txtp.index()];
        let f_ref = skaler_tablo[tx.index()][txtp.index()];

        // 1) DC-only (`txtp == DctDct && eob == 0`): yalnız `coeff[0]` sıfırlanır,
        // `coeff[1..1024]` dokunulmadan kalır ve hedef pikseller skalerle bit-birebirdir.
        if txtp == TxType::DctDct {
            let mut dst_simd = vec![0u8; w * h * 2];
            for chunk in dst_simd.as_chunks_mut::<2>().0 {
                *chunk = 512u16.to_le_bytes();
            }
            let mut dst_ref = dst_simd.clone();
            let mut c_simd = [0i32; 1024];
            for (i, v) in c_simd.iter_mut().enumerate() {
                *v = 40_000 + (i as i32) * 11;
            }
            c_simd[0] = -3_920;
            let beklenen_kuyruk = c_simd;

            let mut c_ref = [0i32; 1024];
            c_ref[0] = -3_920;

            f(
                &mut dst_simd,
                stride,
                &mut c_simd,
                0,
                tx,
                txtp,
                BitDepth::Ten,
            );
            f_ref(&mut dst_ref, stride, &mut c_ref, 0, tx, txtp, BitDepth::Ten);

            assert_eq!(
                dst_simd, dst_ref,
                "{tx:?}/DctDct DC-only 10 bpc piksel farkı"
            );
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
            let mut dst_simd = vec![0u8; w * h * 2];
            for chunk in dst_simd.as_chunks_mut::<2>().0 {
                *chunk = 400u16.to_le_bytes();
            }
            let mut c_simd = [0i32; 1024];
            for (i, v) in c_simd.iter_mut().enumerate().take(n) {
                *v = if i == 0 { 2_048 } else { -67 + (i as i32 % 53) };
            }
            for (i, v) in c_simd.iter_mut().enumerate().skip(n) {
                *v = -98_765 + (i as i32) * 13;
            }
            let beklenen = c_simd;

            f(
                &mut dst_simd,
                stride,
                &mut c_simd,
                eob,
                tx,
                txtp,
                BitDepth::Ten,
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
fn p43_itx10_sse4_sentinel_ve_cift_bayt_stride_denetimi() {
    const T: &str = "itx16_sse4::p43_itx10_sse4_sentinel_ve_cift_bayt_stride_denetimi";
    let Some((_, tablo)) = sse4_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();
    let skaler_tablo = ItxDsp::scalar::<10>().itxfm_add;
    let mut rng = Rng(0x5043_0003_2026_1117);

    for (tx, txtp) in wht_disi_155_hucre() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let row_bytes = w * 2;
        let f = tablo[tx.index()][txtp.index()];
        let f_ref = skaler_tablo[tx.index()][txtp.index()];

        for &stride in &[row_bytes, row_bytes + 2, row_bytes + 6, row_bytes + 14] {
            let data_len = (h - 1) * stride + row_bytes;

            for &eob in &[0usize, 1.min(n - 1), 35.min(n - 1), 135.min(n - 1), n - 1] {
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
                    "{tx:?}/{txtp:?} stride={stride} eob={eob} SSE4.1 != 10 bpc skaler (dolgu dâhil)"
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
        "sözleşme içi 10 bpc SSE4.1 çağrılarında skaler devir 0 olmalı"
    );
}

#[test]
fn p43_itx10_sse4_sozlesme_disi_skalere_devreder() {
    const T: &str = "itx16_sse4::p43_itx10_sse4_sozlesme_disi_skalere_devreder";
    let Some((_, tablo)) = sse4_tablo(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let skaler8 = ItxDsp::scalar::<8>().itxfm_add;
    let skaler10 = ItxDsp::scalar::<10>().itxfm_add;
    let skaler12 = ItxDsp::scalar::<12>().itxfm_add;

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
        let stride = w * 2;
        let f = tablo[tx.index()][txtp.index()];

        // 1) Yanlış `tx` ile çağrı: sayaç +1 ve geçerli 10 bpc skaler çıktı.
        let baska_tx = if tx == TxSize::Tx4x4 {
            TxSize::Tx8x8
        } else {
            TxSize::Tx4x4
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

        // 2) Yanlış `txtp` ile çağrı (`Tx4x4 × Idtx` vs `DctDct`): sayaç +1.
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

        // 3) `BitDepth::Twelve` ile doğrudan çağrı: sayaç +1 ve 12 bpc skaler monomorfla aynı!
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

        // 4) `BitDepth::Eight` ile doğrudan çağrı: sayaç +1 ve 8 bpc skaler monomorfla aynı!
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

        // 5) Tek sayı bayt `dst_stride` (`w * 2 + 1`): paniklemeden sayaç +1.
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

        // 6) `dst_stride < w * 2`: paniklemeden sayaç +1.
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

        // 7) `eob >= sw * sh` (`sw * sh` ve `usize::MAX`): paniklemeden sayaç +1.
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

        // 8) Kısa `dst` dilimi: paniklemeden sayaç +1.
        let mut kisa_dst = vec![0u8; (h - 1) * stride + stride - 1];
        let d_kisa_dst = itx_skaler_devir_sayisi();
        f(&mut kisa_dst, stride, &mut c_ok, 0, tx, txtp, BitDepth::Ten);
        assert_eq!(itx_skaler_devir_sayisi(), d_kisa_dst + 1);

        // 9) Kısa `coeff` dilimi (`< 1024`): paniklemeden sayaç +1.
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
fn p43_itx10_sse4_bozuk_uc_katsayi_taramasi_paniksiz() {
    const T: &str = "itx16_sse4::p43_itx10_sse4_bozuk_uc_katsayi_taramasi_paniksiz";
    let Some((_, tablo)) = sse4_tablo(T) else {
        return;
    };
    let mut rng = Rng(0xDEAD_BEEF_2026_1043);

    for (tx, txtp) in wht_disi_155_hucre() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let n = sw * sh;
        let row_bytes = w * 2;
        let stride = row_bytes + 6;
        let data_len = (h - 1) * stride + row_bytes;
        let f = tablo[tx.index()][txtp.index()];

        for kip in 0..8u8 {
            let mut coeff = [0i32; 1024];
            for (i, v) in coeff.iter_mut().enumerate().take(n) {
                *v = match kip {
                    0 => 131_071,
                    1 => -131_071,
                    2 => -131_072,
                    3 => {
                        if i & 1 == 0 {
                            131_071
                        } else {
                            -131_072
                        }
                    }
                    4 => i32::MAX,
                    5 => i32::MIN,
                    _ => rng.aralik(-131_072, 131_071),
                };
            }
            for v in coeff.iter_mut().skip(n) {
                *v = 0x4242_1010;
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
                    let px = if (x + y) & 1 == 0 { 0u16 } else { 1023u16 };
                    let off = y * stride + x * 2;
                    sb.data_mut()[off..off + 2].copy_from_slice(&px.to_le_bytes());
                }
            }

            f(
                sb.data_mut(),
                stride,
                &mut coeff,
                eob,
                tx,
                txtp,
                BitDepth::Ten,
            );

            assert_eq!(
                sb.verify(),
                Ok(()),
                "{tx:?}/{txtp:?} kip={kip}: bozuk 10 bpc katsayıda sentinel ihlali"
            );
            for y in 0..h {
                for x in 0..w {
                    let off = y * stride + x * 2;
                    let px = u16::from_le_bytes([sb.data()[off], sb.data()[off + 1]]);
                    assert!(
                        px <= 1023,
                        "{tx:?}/{txtp:?} kip={kip} ({x},{y}): çıktı {px} > 1023"
                    );
                }
                if y + 1 < h {
                    for b in row_bytes..stride {
                        assert_eq!(sb.data()[y * stride + b], 0x5A);
                    }
                }
            }
            if txtp == TxType::DctDct && eob == 0 {
                assert_eq!(coeff[0], 0);
            } else {
                assert!(coeff[..n].iter().all(|&v| v == 0));
            }
            assert!(coeff[n..].iter().all(|&v| v == 0x4242_1010));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p43_itx10_sse4_isa_siniri_objdump_denetimi() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let yasak_sse42_ve_ustu = [
        "pcmpgtq",
        "pcmpestri",
        "pcmpestrm",
        "pcmpistri",
        "pcmpistrm",
        "crc32",
        "crc32b",
        "crc32w",
        "crc32l",
        "crc32q",
        "popcnt",
        "popcntw",
        "popcntl",
        "popcntq",
        "aesenc",
        "aesenclast",
        "aesdec",
        "aesdeclast",
        "aesimc",
        "aeskeygenassist",
        "pclmulqdq",
        "lzcnt",
        "tzcnt",
        "bzhi",
        "pdep",
        "pext",
        "mulx",
        "rorx",
        "sarx",
        "shlx",
        "shrx",
    ];

    let mut kucuk_cekirdek = 0usize;
    let mut buyuk_cekirdek = 0usize;
    let mut aktif = false;
    let mut aktif_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") {
            let ilgili = satir.contains("itx16_sse4") || satir.contains("itx16_buyuk_sse4");
            let uretim = ilgili && satir.contains("dsp") && satir.contains("x86");
            aktif = uretim;
            if uretim {
                aktif_sembol = satir.to_string();
                if satir.contains("itx10_sse4_cekirdek") {
                    kucuk_cekirdek += 1;
                } else if satir.contains("itx10_buyuk_sse4_cekirdek") {
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
            !yasak_sse42_ve_ustu.contains(&op),
            "SSE4.1 sembolünde SSE4.2+ yönergesi ({op}): {aktif_sembol} -> {satir}"
        );
        assert!(
            !op.starts_with('v') || op == "verw" || op == "verr",
            "SSE4.1 sembolünde VEX/EVEX yönergesi ({op}): {aktif_sembol} -> {satir}"
        );
        assert!(
            !sag.contains("%ymm") && !sag.contains("%zmm") && !sag.contains("{%k"),
            "SSE4.1 sembolünde YMM/ZMM/maske yazmacı: {aktif_sembol} -> {satir}"
        );
    }

    assert!(
        kucuk_cekirdek >= 140,
        "en az 140 itx10_sse4_cekirdek sembolü bulunmalı: {kucuk_cekirdek}"
    );
    assert!(
        buyuk_cekirdek >= 15,
        "en az 15 itx10_buyuk_sse4_cekirdek sembolü bulunmalı: {buyuk_cekirdek}"
    );
    println!(
        "[P43 ITX10 SSE4.1 objdump] itx16_sse4={kucuk_cekirdek}, itx16_buyuk_sse4={buyuk_cekirdek} sembol temiz (0 SSE4.2+, 0 VEX/EVEX)"
    );
}
