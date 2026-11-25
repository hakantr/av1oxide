//! P43 (F4-ITX a, Adım 1) — Kayıpsız `TX_4X4 × WHT_WHT` SSE2 ve AVX2 çekirdek testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **Oracle eşitliği (`D02583`, `D02584`, `D02581`, `D02582`):** `8`, `10` ve `12` bpc
//!    `itx-small` (ve varsa `.cache/itx-oracle` `itx-large`) kayıtlarında SSE2 ve AVX2
//!    gövdeleri dav1d C oracle çıktısıyla **bit-birebir** aynıdır; `coeff[..16]` sıfırlanır,
//!    `coeff[16..1024]` kuyruğuna dokunulmaz.
//! 2. **Rastgele + uç girdiler + `SentinelBuffer` sınır nöbetçisi:** `8/10/12` bpc ×
//!    çeşitli `dst_stride` adımları × `eob` (`0, 1, 15, 16, 999, usize::MAX`: `WHT_WHT`
//!    `eob` değerine bakmaz) × uç katsayı/piksel desenleri; ön/arka `SentinelBuffer`
//!    koruyucuları ve satır arası dolgu baytları (`0xA5`) değişmez (`0` taşma).
//! 3. **Sözleşme dışı skaler devir sayacı:** sözleşme içi çağrılarda
//!    [`itx_skaler_devir_sayisi`] `0` artar; yanlış `(tx, txtp)`, yanlış `bit_depth`,
//!    16 bpc tek `dst_stride` ya da kısa `dst`/`coeff` diliminde paniklemeden sayacı
//!    artırır ve skaler monomorfa devreder.
//! 4. **ISA sınırı (`objdump -d`):** `itx_sse2_*` sembollerinde SSSE3+/SSE4.1/VEX komutu
//!    `0`, `itx_avx2_*` sembollerinde EVEX (`zmm`/`{k*}`) komutu `0`.

#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(unused_imports, dead_code, unused_variables, unused_mut)
)]
#![forbid(unsafe_code)]

mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::itx::{ItxDsp, ItxFn, TxSize, TxType, itx_skaler_devir_sayisi};
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
        lo + ((self.u64() >> 33) % (hi - lo + 1) as u64) as i32
    }
}

struct WhtYuvalar {
    sse2: [ItxFn; 3],
    avx2: [ItxFn; 3],
}

fn bd_idx(bd: BitDepth) -> usize {
    match bd {
        BitDepth::Eight => 0,
        BitDepth::Ten => 1,
        BitDepth::Twelve => 2,
    }
}

fn wht_yuvalari(test: &str) -> Option<WhtYuvalar> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let d_sse2 = DspDispatch::select(detected.intersection(CpuLevel::Sse2.allowed()));
        let d_avx2 = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()));
        if d_sse2.sse2_token().is_some() && d_avx2.avx2_token().is_some() {
            let al = |d: &DspDispatch, bd: BitDepth| {
                d.for_bit_depth(bd).itx().itxfm_add[TxSize::Tx4x4.index()][TxType::WhtWht.index()]
            };
            return Some(WhtYuvalar {
                sse2: [
                    al(&d_sse2, BitDepth::Eight),
                    al(&d_sse2, BitDepth::Ten),
                    al(&d_sse2, BitDepth::Twelve),
                ],
                avx2: [
                    al(&d_avx2, BitDepth::Eight),
                    al(&d_avx2, BitDepth::Ten),
                    al(&d_avx2, BitDepth::Twelve),
                ],
            });
        }
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P43] WHT_WHT SSE2 ve AVX2 gövdeleri x86_64 SSE2+AVX2 ister",
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

fn read_i32<R: Read>(r: &mut R) -> i32 {
    i32::from_le_bytes([read_u8(r), read_u8(r), read_u8(r), read_u8(r)])
}

fn pix_seed(record_index: u64, bpc: u8) -> u64 {
    0x9E37_79B9_7F4A_7C15u64.wrapping_mul(record_index + 1) ^ (u64::from(bpc) << 51)
}

fn pix_next(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_000 + 407);
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

fn verify_oracle_dir(dir: &Path, profile_name: &str, y: &WhtYuvalar) -> usize {
    let mut wht_records = 0usize;
    for (bpc, bd) in [
        (8u8, BitDepth::Eight),
        (10, BitDepth::Ten),
        (12, BitDepth::Twelve),
    ] {
        let in_path = dir.join(format!("itx-{profile_name}-{bpc}-input.bin"));
        let exp_path = dir.join(format!("itx-{profile_name}-{bpc}-expect.bin"));
        let (Ok(in_f), Ok(exp_f)) = (File::open(&in_path), File::open(&exp_path)) else {
            continue;
        };
        let mut inp = BufReader::new(in_f);
        let mut exp = BufReader::new(exp_f);
        let mut magic = [0u8; 4];
        inp.read_exact(&mut magic).expect("magic");
        assert_eq!(&magic, b"ITXI");
        assert_eq!(read_u32(&mut inp), 1);
        assert_eq!(read_u8(&mut inp), bpc);
        let profile = read_u8(&mut inp);
        assert_eq!(read_u16(&mut inp), 0);
        let count = read_u32(&mut inp);

        exp.read_exact(&mut magic).expect("exp magic");
        assert_eq!(&magic, b"ITXE");
        assert_eq!(read_u32(&mut exp), 1);
        assert_eq!(read_u8(&mut exp), bpc);
        let _ = read_u8(&mut exp);
        assert_eq!(read_u16(&mut exp), 0);
        assert_eq!(read_u32(&mut exp), count);

        let bpp = if bpc == 8 { 1 } else { 2 };
        let bdmax = (1u32 << bpc) - 1;

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
            let w = tx.width();
            let h = tx.height();
            let sw = w.min(32);
            let sh = h.min(32);

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

            let e_tx = read_u8(&mut exp);
            let e_txtp = read_u8(&mut exp);
            let e_bpc = read_u8(&mut exp);
            let _ = read_u8(&mut exp);
            let e_eob = usize::from(read_u16(&mut exp));
            let _ = read_u16(&mut exp);
            assert_eq!((e_tx, e_txtp, e_bpc, e_eob), (tx_u8, txtp_u8, bpc, eob));

            let mut exp_px = vec![0u16; w * h];
            for p in &mut exp_px {
                *p = read_u16(&mut exp);
            }

            if tx != TxSize::Tx4x4 || txtp != TxType::WhtWht {
                continue;
            }
            wht_records += 1;

            let stride = w * bpp;
            let mut base_dst = vec![0u8; h * stride];
            let mut st = pix_seed(rec, bpc);
            let variant = if profile == 0 { (rec % 6) as u8 } else { 0xff };
            for p in 0..w * h {
                let v = pix_next(&mut st) & bdmax;
                let px = if profile == 0 && (variant == 1 || variant == 5) {
                    bdmax
                } else if profile == 0 && (variant == 0 || variant == 2) {
                    0
                } else {
                    v
                } as u16;
                if bpp == 1 {
                    base_dst[p] = px as u8;
                } else {
                    base_dst[2 * p..2 * p + 2].copy_from_slice(&px.to_le_bytes());
                }
            }

            for (isa_ad, f) in [("sse2", y.sse2[bd_idx(bd)]), ("avx2", y.avx2[bd_idx(bd)])] {
                let mut dst = base_dst.clone();
                let mut coeff = coeff_in;
                f(&mut dst, stride, &mut coeff, eob, tx, txtp, bd);
                for (i, &c) in coeff[..16].iter().enumerate() {
                    assert_eq!(
                        c, 0,
                        "{profile_name} {bpc}bpc {isa_ad} rec={rec} coeff[{i}] != 0"
                    );
                }
                for (i, &c) in coeff.iter().enumerate().skip(16) {
                    assert_eq!(
                        c,
                        tail_value(i, rec, bpc),
                        "{profile_name} {bpc}bpc {isa_ad} rec={rec} kuyruk coeff[{i}] bozuldu"
                    );
                }
                for y_i in 0..4 {
                    for x_i in 0..4 {
                        let off = y_i * stride + x_i * bpp;
                        let got = if bpp == 1 {
                            u16::from(dst[off])
                        } else {
                            u16::from_le_bytes([dst[off], dst[off + 1]])
                        };
                        let want = exp_px[y_i * 4 + x_i];
                        assert_eq!(
                            got, want,
                            "{profile_name} {bpc}bpc {isa_ad} rec={rec} ({x_i},{y_i}) != oracle"
                        );
                    }
                }
            }
        }
    }
    wht_records
}

#[test]
fn p43_wht_oracle_fixture_ve_buyuk_kumede_bit_exact() {
    const T: &str = "itx_wht::p43_wht_oracle_fixture_ve_buyuk_kumede_bit_exact";
    let Some(y) = wht_yuvalari(T) else {
        return;
    };
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let small_dir = manifest.join("tests/fixtures/itx");
    let small_n = verify_oracle_dir(&small_dir, "small", &y);
    assert_eq!(
        small_n, 18,
        "itx-small içinde 3 bpc × 6 kayıt = 18 WHT kaydı olmalı"
    );

    let large_dir = manifest.join("../../.cache/itx-oracle");
    let large_n = verify_oracle_dir(&large_dir, "large", &y);
    println!("[P43 WHT oracle] small={small_n} kayıt, large={large_n} kayıt bit-exact");
}

#[test]
fn p43_wht_rastgele_uc_ve_sentinel_sinir_denetimi() {
    const T: &str = "itx_wht::p43_wht_rastgele_uc_ve_sentinel_sinir_denetimi";
    let Some(y) = wht_yuvalari(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();
    let mut rng = Rng(0x5043_0001_2026_1117);

    for (bpc, bd) in [
        (8u8, BitDepth::Eight),
        (10, BitDepth::Ten),
        (12, BitDepth::Twelve),
    ] {
        let bpp = if bpc == 8 { 1 } else { 2 };
        let bdmax = (1i32 << bpc) - 1;
        // WHT'de `coeff >> 2` sonrası 2× 1B WHT4 toplamı 16 katına çıkar;
        // geçerli AV1 WHT katsayı aralığı `|c| <= bdmax * 8` (8 bpc'de i16 içinde).
        let c_lim = (bdmax * 8).min(if bpc == 8 { 32767 } else { i32::MAX / 8 });
        let skaler =
            ItxDsp::scalar_for(bd).itxfm_add[TxSize::Tx4x4.index()][TxType::WhtWht.index()];

        for stride_px in [4usize, 5, 8, 13, 32] {
            let stride = stride_px * bpp;
            let data_len = 3 * stride + 4 * bpp;

            for iter in 0..600usize {
                let mut c_base = [0i32; 1024];
                for (i, v) in c_base.iter_mut().enumerate().skip(16) {
                    *v = 0x5A5A_0000u32.wrapping_add(i as u32) as i32;
                }
                match iter {
                    0 => {} // tamamı sıfır
                    1 => {
                        for v in &mut c_base[..16] {
                            *v = c_lim;
                        }
                    }
                    2 => {
                        for v in &mut c_base[..16] {
                            *v = -c_lim;
                        }
                    }
                    3..=18 => {
                        c_base[iter - 3] = if iter & 1 == 0 { c_lim } else { -c_lim };
                    }
                    _ => {
                        for v in &mut c_base[..16] {
                            *v = rng.aralik(-c_lim, c_lim);
                        }
                    }
                }

                let mut s_ref = SentinelBuffer::with_defaults(data_len);
                s_ref.data_mut().fill(0xA5);
                for r in 0..4 {
                    for c in 0..4 {
                        let px = match iter % 4 {
                            0 => 0u16,
                            1 => bdmax as u16,
                            _ => rng.aralik(0, bdmax) as u16,
                        };
                        let off = r * stride + c * bpp;
                        if bpp == 1 {
                            s_ref.data_mut()[off] = px as u8;
                        } else {
                            s_ref.data_mut()[off..off + 2].copy_from_slice(&px.to_le_bytes());
                        }
                    }
                }
                let mut s_sse2 = s_ref.clone();
                let mut s_avx2 = s_ref.clone();

                let eob = match iter % 5 {
                    0 => 0,
                    1 => 1,
                    2 => 15,
                    3 => 16,
                    _ => usize::MAX,
                };

                let mut c_ref = c_base;
                let mut c_sse2 = c_base;
                let mut c_avx2 = c_base;

                skaler(
                    s_ref.data_mut(),
                    stride,
                    &mut c_ref,
                    eob,
                    TxSize::Tx4x4,
                    TxType::WhtWht,
                    bd,
                );
                (y.sse2[bd_idx(bd)])(
                    s_sse2.data_mut(),
                    stride,
                    &mut c_sse2,
                    eob,
                    TxSize::Tx4x4,
                    TxType::WhtWht,
                    bd,
                );
                (y.avx2[bd_idx(bd)])(
                    s_avx2.data_mut(),
                    stride,
                    &mut c_avx2,
                    eob,
                    TxSize::Tx4x4,
                    TxType::WhtWht,
                    bd,
                );

                assert_eq!(s_sse2.verify(), Ok(()), "{bpc}bpc sse2 sentinel bozuldu");
                assert_eq!(s_avx2.verify(), Ok(()), "{bpc}bpc avx2 sentinel bozuldu");
                assert_eq!(
                    s_sse2.data(),
                    s_ref.data(),
                    "{bpc}bpc sse2 != skaler (stride={stride}, iter={iter})"
                );
                assert_eq!(
                    s_avx2.data(),
                    s_ref.data(),
                    "{bpc}bpc avx2 != skaler (stride={stride}, iter={iter})"
                );
                assert_eq!(&c_sse2[..16], &[0i32; 16]);
                assert_eq!(&c_avx2[..16], &[0i32; 16]);
                assert_eq!(&c_sse2[16..], &c_base[16..]);
                assert_eq!(&c_avx2[16..], &c_base[16..]);
            }
        }
    }
    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "sözleşme içi WHT çağrılarında skaler devir 0 olmalı"
    );
}

#[test]
fn p43_wht_sozlesme_disi_skalere_devreder_ve_sayilir() {
    const T: &str = "itx_wht::p43_wht_sozlesme_disi_skalere_devreder_ve_sayilir";
    let Some(y) = wht_yuvalari(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    for (bpc, bd) in [
        (8u8, BitDepth::Eight),
        (10, BitDepth::Ten),
        (12, BitDepth::Twelve),
    ] {
        let bpp = if bpc == 8 { 1 } else { 2 };
        let stride = 4 * bpp;
        let skaler =
            ItxDsp::scalar_for(bd).itxfm_add[TxSize::Tx4x4.index()][TxType::DctDct.index()];

        for f in [y.sse2[bd_idx(bd)], y.avx2[bd_idx(bd)]] {
            // 1) Yanlış `txtp` (`DctDct`): skaler monomorfa devreder ve aynı çıktıyı üretir.
            let mut dst_s = vec![80u8; 4 * stride];
            let mut dst_i = dst_s.clone();
            let mut c_s = [0i32; 1024];
            c_s[0] = 256;
            let mut c_i = c_s;
            let d0 = itx_skaler_devir_sayisi();
            skaler(
                &mut dst_s,
                stride,
                &mut c_s,
                0,
                TxSize::Tx4x4,
                TxType::DctDct,
                bd,
            );
            f(
                &mut dst_i,
                stride,
                &mut c_i,
                0,
                TxSize::Tx4x4,
                TxType::DctDct,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d0 + 1);
            assert_eq!(dst_i, dst_s);

            // 2) Yanlış `bit_depth`: skaler monomorfa devreder.
            let yanlis_bd = if bd == BitDepth::Eight {
                BitDepth::Ten
            } else {
                BitDepth::Eight
            };
            let d1 = itx_skaler_devir_sayisi();
            f(
                &mut dst_i,
                stride,
                &mut c_i,
                0,
                TxSize::Tx4x4,
                TxType::WhtWht,
                yanlis_bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d1 + 1);

            // 3) Kısa `dst` dilimi: paniklemeden sayacı artırır ve döner.
            let mut kisa_dst = vec![0u8; 3 * stride + 4 * bpp - 1];
            let d2 = itx_skaler_devir_sayisi();
            f(
                &mut kisa_dst,
                stride,
                &mut c_i,
                0,
                TxSize::Tx4x4,
                TxType::WhtWht,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d2 + 1);

            // 4) Kısa `coeff` dilimi (< 1024): paniklemeden sayacı artırır ve döner.
            let mut kisa_coeff = [0i32; 16];
            let d3 = itx_skaler_devir_sayisi();
            f(
                &mut dst_i,
                stride,
                &mut kisa_coeff,
                0,
                TxSize::Tx4x4,
                TxType::WhtWht,
                bd,
            );
            assert_eq!(itx_skaler_devir_sayisi(), d3 + 1);

            // 5) 16 bpc'de tek `dst_stride` (`9`): skaler monomorfa devreder ve aynı çıktıyı üretir.
            if bpp == 2 {
                let tek_stride = 9usize;
                let mut dst_ts = vec![40u8; 3 * tek_stride + 8];
                let mut dst_ti = dst_ts.clone();
                let mut c_ts = [0i32; 1024];
                c_ts[0] = 120;
                c_ts[5] = -60;
                let mut c_ti = c_ts;
                let sk_wht =
                    ItxDsp::scalar_for(bd).itxfm_add[TxSize::Tx4x4.index()][TxType::WhtWht.index()];
                let d4 = itx_skaler_devir_sayisi();
                sk_wht(
                    &mut dst_ts,
                    tek_stride,
                    &mut c_ts,
                    0,
                    TxSize::Tx4x4,
                    TxType::WhtWht,
                    bd,
                );
                f(
                    &mut dst_ti,
                    tek_stride,
                    &mut c_ti,
                    0,
                    TxSize::Tx4x4,
                    TxType::WhtWht,
                    bd,
                );
                assert_eq!(itx_skaler_devir_sayisi(), d4 + 1);
                assert_eq!(dst_ti, dst_ts);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p43_wht_isa_siniri_objdump_denetimi() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let yasak_sse2 = [
        "pshufb",
        "pmaddubsw",
        "pmulhrsw",
        "phaddw",
        "phaddd",
        "phaddsw",
        "phsubw",
        "phsubd",
        "phsubsw",
        "palignr",
        "pabsb",
        "pabsw",
        "pabsd",
        "psignb",
        "psignw",
        "psignd",
        "pblendvb",
        "pblendw",
        "pinsrb",
        "pinsrd",
        "pinsrq",
        "pextrb",
        "pextrd",
        "pextrq",
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
        "pmuldq",
        "pmulld",
        "pminsb",
        "pmaxsb",
        "pminuw",
        "pmaxuw",
        "pminud",
        "pmaxud",
        "pminsd",
        "pmaxsd",
        "packusdw",
    ];

    let mut sse2_bulundu = 0usize;
    let mut avx2_bulundu = 0usize;
    let mut aktif_kip: Option<&str> = None;
    let mut aktif_sembol = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") && satir.contains("itx_wht") {
            aktif_sembol = satir.to_string();
            if satir.contains("itx_sse2_wht_wht_4x4_") {
                aktif_kip = Some("sse2");
                sse2_bulundu += 1;
            } else if satir.contains("itx_avx2_wht_wht_4x4_") {
                aktif_kip = Some("avx2");
                avx2_bulundu += 1;
            } else {
                aktif_kip = None;
            }
            continue;
        }
        if satir.is_empty() {
            aktif_kip = None;
            continue;
        }
        let Some(kip) = aktif_kip else {
            continue;
        };
        let Some((_, sag)) = satir.split_once('\t') else {
            continue;
        };
        let parcalar: Vec<&str> = sag.split_whitespace().collect();
        let Some(&op) = parcalar.first() else {
            continue;
        };
        match kip {
            "sse2" => {
                assert!(
                    !yasak_sse2.contains(&op),
                    "SSE2 sembolünde SSSE3+/SSE4.1 yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !op.starts_with('v') || op == "verw",
                    "SSE2 sembolünde VEX yönergesi ({op}): {aktif_sembol} -> {satir}"
                );
                assert!(
                    !sag.contains("%ymm") && !sag.contains("%zmm"),
                    "SSE2 sembolünde YMM/ZMM yazmacı: {aktif_sembol} -> {satir}"
                );
            }
            "avx2" => {
                assert!(
                    !sag.contains("%zmm") && !sag.contains("{%k"),
                    "AVX2 sembolünde EVEX yönergesi: {aktif_sembol} -> {satir}"
                );
            }
            _ => {}
        }
    }

    assert!(
        sse2_bulundu >= 2,
        "en az 2 itx_sse2_wht_wht_4x4_* sembolü bulunmalı: {sse2_bulundu}"
    );
    assert!(
        avx2_bulundu >= 2,
        "en az 2 itx_avx2_wht_wht_4x4_* sembolü bulunmalı: {avx2_bulundu}"
    );
}
