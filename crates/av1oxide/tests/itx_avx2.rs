//! P43 (F4-ITX a, Adım 2) — WHT-dışı 8 bpc AVX2 ters dönüşüm (`155` kayıt: `itx` `140` + `itx_buyuk` `15`) testleri.
//!
//! Doğrulanan sözleşme:
//! 1. **Dispatch kurulumu (`155` yuva):** `CpuLevel::Avx2` altında 8 bpc WHT-dışı desteklenen
//!    `155` `(tx, txtp)` yuvasının tamamı `"avx2"` olarak kurulur.
//! 2. **C oracle bit-birebir eşitliği:** `tests/fixtures/itx/itx-small-8-*` (`930` WHT-dışı blok)
//!    ve `.cache/itx-oracle/itx-large-8-*` (`155 000` WHT-dışı blok) üzerinde hem `dst`
//!    pikselleri hem de çağrı sonrası `coeff[0..1024]` durumu dav1d C referansıyla **%100 bit-birebir**
//!    aynıdır.
//! 3. **Kesin `coeff` sıfırlama sözleşmesi:**
//!    - `txtp == TxType::DctDct && eob == 0` (DC-only): **yalnız `coeff[0] = 0`** sıfırlanır,
//!      `coeff[1..1024]` öğelerine dokunulmaz.
//!    - Genel yol (`eob ≥ 1` veya `txtp != DctDct`): `eob` alt-blok eşikleri dâhil her durumda
//!      `coeff[..sw * sh]` (`sw = min(w, 32), sh = min(h, 32)`) sıfırlanır, `coeff[sw * sh..1024]`
//!      kuyruğuna dokunulmaz.
//! 4. **`SentinelBuffer` sınır güvenliği ve tekil `dst_stride` (`w`, `w + 1`, `w + 7`, `w + 13`):**
//!    `155` yuvanın tamamında ön/arka nöbetçi bölgeleri ve satır arası dolgu baytları (`0xA5`)
//!    bozulmaz (`0` taşma), çıktı skaler referansla birebir aynıdır, skaler devir sayacı `0` kalır.
//! 5. **Sözleşme dışı güvenli devir:** yanlış `BitDepth`, yanlış `(tx, txtp)`, `eob ≥ sw * sh`,
//!    `dst_stride < w`, kısa `dst` veya kısa `coeff < 1024` girdilerinde paniklemeden
//!    [`itx_skaler_devir_sayisi`] artar ve uygun olduğunda skaler monomorfa devreder.
//! 6. **Bozuk akış / rastgele `±32767` katsayı dayanıklılığı (`--profile robust` dâhil):**
//!    `155` yuvanın tamamında uç ve rastgele katsayı örüntülerinde **0 panik**, **0 taşma**.
//! 7. **ISA sınırı (`objdump -d`):** `dsp::x86::itx` ve `dsp::x86::itx_buyuk` sembollerinde
//!    EVEX (`%zmm` / `{%k`) yönergesi sayısı `0`'dır.

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
use av1oxide::dsp::itx_tables::LNZC_TABLES;
use av1oxide::dsp::scalar::itx_1d::Tx1dType;
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

type Tablo = [[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES];

fn avx2_yuvalari(test: &str) -> Option<(DspDispatch, Tablo)> {
    #[cfg(target_arch = "x86_64")]
    {
        let detected = CpuFlags::detect();
        let dsp = DspDispatch::select(detected.intersection(CpuLevel::Avx2.allowed()));
        if dsp.avx2_token().is_some() {
            let tablo = dsp.for_bit_depth(BitDepth::Eight).itx().itxfm_add;
            return Some((dsp, tablo));
        }
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P43] 8 bpc WHT-dışı ITX AVX2 gövdeleri x86_64 AVX2 ister",
    );
    None
}

fn desteklenen_155_yuva() -> Vec<(TxSize, TxType)> {
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

fn tail_value(index: usize, record_index: u64) -> i32 {
    let x = (index as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add((record_index as u32).wrapping_mul(40_503))
        .wrapping_add(8 * 97);
    let v = i32::from(x as i16);
    if v == 0 { 13_579 } else { v }
}

fn verify_oracle_8bpc(dir: &Path, profile_name: &str, tablo: &Tablo) -> Option<(usize, usize)> {
    let in_path = dir.join(format!("itx-{profile_name}-8-input.bin"));
    let exp_path = dir.join(format!("itx-{profile_name}-8-expect.bin"));
    let (Ok(in_f), Ok(exp_f)) = (File::open(&in_path), File::open(&exp_path)) else {
        return None;
    };
    let mut inp = BufReader::new(in_f);
    let mut exp = BufReader::new(exp_f);

    let mut magic = [0u8; 4];
    inp.read_exact(&mut magic).expect("ITXI magic");
    assert_eq!(&magic, b"ITXI");
    assert_eq!(read_u32(&mut inp), 1);
    assert_eq!(read_u8(&mut inp), 8);
    let profile = read_u8(&mut inp);
    assert_eq!(read_u16(&mut inp), 0);
    let count = read_u32(&mut inp);

    exp.read_exact(&mut magic).expect("ITXE magic");
    assert_eq!(&magic, b"ITXE");
    assert_eq!(read_u32(&mut exp), 1);
    assert_eq!(read_u8(&mut exp), 8);
    let _ = read_u8(&mut exp);
    assert_eq!(read_u16(&mut exp), 0);
    assert_eq!(read_u32(&mut exp), count);

    let mut non_wht = 0usize;
    let mut wht = 0usize;

    for rec in 0..u64::from(count) {
        let tx_u8 = read_u8(&mut inp);
        let txtp_u8 = read_u8(&mut inp);
        let rec_bpc = read_u8(&mut inp);
        let _ = read_u8(&mut inp);
        let eob = usize::from(read_u16(&mut inp));
        let _ = read_u16(&mut inp);
        assert_eq!(rec_bpc, 8);

        let tx = TxSize::from_index(usize::from(tx_u8)).expect("tx");
        let txtp = TxType::from_index(usize::from(txtp_u8)).expect("txtp");
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);

        let mut coeff_in = [0i32; 1024];
        for v in coeff_in.iter_mut().take(sw * sh) {
            *v = i32::from(read_i16(&mut inp));
        }
        for (idx, v) in coeff_in.iter_mut().enumerate().skip(sw * sh) {
            *v = tail_value(idx, rec);
        }

        let e_tx = read_u8(&mut exp);
        let e_txtp = read_u8(&mut exp);
        let e_bpc = read_u8(&mut exp);
        let _ = read_u8(&mut exp);
        let e_eob = usize::from(read_u16(&mut exp));
        let _ = read_u16(&mut exp);
        assert_eq!((e_tx, e_txtp, e_bpc, e_eob), (tx_u8, txtp_u8, 8, eob));

        let mut exp_px = vec![0u8; w * h];
        for p in &mut exp_px {
            let v = read_u16(&mut exp);
            assert!(v <= 255);
            *p = v as u8;
        }

        if txtp == TxType::WhtWht {
            wht += 1;
            continue;
        }
        non_wht += 1;

        let is_dconly = txtp == TxType::DctDct && eob == 0;
        if is_dconly {
            // DC-only sözleşmesi: `coeff[1..1024]` hiçbir şekilde okunmamalı ve yazılmamalı.
            for (idx, v) in coeff_in.iter_mut().enumerate().skip(1) {
                *v = tail_value(idx, rec);
            }
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

        let mut coeff = coeff_in;
        let f = tablo[tx.index()][txtp.index()];
        f(&mut dst, w, &mut coeff, eob, tx, txtp, BitDepth::Eight);

        assert_eq!(
            dst, exp_px,
            "{profile_name} rec={rec} {tx:?}/{txtp:?} eob={eob} piksel uyuşmazlığı"
        );

        if is_dconly {
            assert_eq!(coeff[0], 0, "{profile_name} rec={rec} DC coeff[0] != 0");
            assert_eq!(
                &coeff[1..],
                &coeff_in[1..],
                "{profile_name} rec={rec} DC-only coeff[1..1024] koruma ihlali"
            );
        } else {
            for (i, &c) in coeff[..sw * sh].iter().enumerate() {
                assert_eq!(
                    c, 0,
                    "{profile_name} rec={rec} {tx:?}/{txtp:?} eob={eob} coeff[{i}] != 0"
                );
            }
            assert_eq!(
                &coeff[sw * sh..],
                &coeff_in[sw * sh..],
                "{profile_name} rec={rec} {tx:?}/{txtp:?} kuyruk coeff[{sw}*{sh}..1024] bozuldu"
            );
        }
    }
    Some((non_wht, wht))
}

// --- Tamsayı ileri dönüşüm yardımcısı (`SentinelBuffer` testi için geçerli katsayı üretir) ---

const COS128: [i64; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];
const SIN9: [i64; 9] = [0, 1401, 2633, 3547, 4034, 4034, 3547, 2633, 1401];
const SCALE4096: [i64; 9] = [16384, 11585, 8192, 5793, 4096, 1448, 1024, 362, 256];

fn cos128(k: usize) -> i64 {
    let m = k & 255;
    if m <= 64 {
        COS128[m]
    } else if m <= 128 {
        -COS128[128 - m]
    } else if m <= 192 {
        -COS128[m - 128]
    } else {
        COS128[256 - m]
    }
}

fn sin9(k: usize) -> i64 {
    let m = k % 18;
    if m < 9 { SIN9[m] } else { -SIN9[m - 9] }
}

fn fdct_1d(out: &mut [i64], inp: &[i64], sz: usize) {
    let step = 64 / sz;
    for (i, out_i) in out[..sz].iter_mut().enumerate() {
        let mut s = 0i64;
        for (j, &inp_j) in inp[..sz].iter().enumerate() {
            s += inp_j * cos128((2 * j + 1) * i * step);
        }
        if i == 0 {
            s = (s * 2896 + 2048) >> 12;
        }
        *out_i = (s + 2048) >> 12;
    }
}

fn fadst_1d(out: &mut [i64], inp: &[i64], sz: usize) {
    for (i, out_i) in out[..sz].iter_mut().enumerate() {
        let mut s = 0i64;
        if sz == 4 {
            for (j, &inp_j) in inp[..4].iter().enumerate() {
                s += inp_j * sin9((j + 1) * (2 * i + 1));
            }
        } else {
            let step = 32 / sz;
            for (j, &inp_j) in inp[..sz].iter().enumerate() {
                s += inp_j * cos128((2 * j + 1) * (2 * i + 1) * step + 192);
            }
        }
        *out_i = (s + 2048) >> 12;
    }
}

fn fwd_1d(kind: Tx1dType, out: &mut [i64], inp: &[i64], sz: usize) {
    match kind {
        Tx1dType::Dct => fdct_1d(out, inp, sz),
        Tx1dType::Adst | Tx1dType::FlipAdst => fadst_1d(out, inp, sz),
        Tx1dType::Identity => out[..sz].copy_from_slice(&inp[..sz]),
    }
}

/// Geçerli aralıkta ileri dönüşüm katsayıları ve alt-blok `eob` dilimlemesi üretir.
fn uret_gecerli_katsayi(tx: TxSize, txtp: TxType, kip: u8, rng: &mut Rng) -> ([i32; 1024], usize) {
    let w = tx.width();
    let h = tx.height();
    let sw = w.min(32);
    let sh = h.min(32);
    let scale = SCALE4096[(w * h).trailing_zeros() as usize - 4];
    let horiz = txtp.tx1d().0;

    let mut temp = [0i64; 64 * 64];
    let mut out = [0i64; 64 * 64];
    for i in 0..h {
        let mut inp = [0i64; 64];
        let mut t_out = [0i64; 64];
        for v in inp.iter_mut().take(w) {
            *v = i64::from(rng.aralik(-255, 255));
        }
        fwd_1d(horiz, &mut t_out, &inp, w);
        for (j, &tj) in t_out[..w].iter().enumerate() {
            temp[j * h + i] = (tj * scale + 2048) >> 12;
        }
    }
    for i in 0..w {
        fwd_1d(
            horiz,
            &mut out[i * h..i * h + h],
            &temp[i * h..i * h + h],
            h,
        );
    }

    let mut buf = [0i32; 1024];
    for y in 0..sh {
        for x in 0..sw {
            buf[y * sw + x] = out[y * w + x].clamp(-32768, 32767) as i32;
        }
    }

    let eob = match kip {
        0 => {
            buf[1..sw * sh].fill(0);
            if buf[0] == 0 {
                buf[0] = 17;
            }
            0
        }
        1 => {
            let (h1d, v1d) = txtp.tx1d();
            if v1d == Tx1dType::Identity && h1d != Tx1dType::Identity {
                // TX_CLASS_H: sütun-majör tarama (`rc = n`).
                let e = (sw * sh / 3).max(1).min(sw * sh - 1);
                buf[e + 1..sw * sh].fill(0);
                e
            } else if h1d == Tx1dType::Identity && v1d != Tx1dType::Identity {
                // TX_CLASS_V: satır-majör tarama (`rcx = n / sw, rcy = n % sw`).
                let e = (sw * sh / 3).max(1).min(sw * sh - 1);
                for n in (e + 1)..(sw * sh) {
                    let rcx = n / sw;
                    let rcy = n % sw;
                    buf[rcy * sh + rcx] = 0;
                }
                e
            } else if sw == 4 && sh == 4 {
                15
            } else {
                // TX_CLASS_2D: sol-üst 8×8 çeyrek (`itx_bench` Sub1 ile aynı).
                let d_max = if sw >= 16 && sh >= 16 { 7 } else { 8 };
                for rcx in 0..sh {
                    for rcy in 0..sw {
                        if rcx > 7 || rcy > 7 || rcx + rcy > d_max {
                            buf[rcy * sh + rcx] = 0;
                        }
                    }
                }
                if sw == 4 || sh == 4 {
                    28
                } else if sw == 8 || sh == 8 {
                    42
                } else {
                    35
                }
            }
        }
        _ => sw * sh - 1,
    };

    for (i, slot) in buf.iter_mut().enumerate().skip(sw * sh) {
        *slot = tail_value(i, kip as u64 + 1);
    }
    if txtp == TxType::DctDct && eob == 0 {
        for (i, slot) in buf.iter_mut().enumerate().skip(1) {
            *slot = tail_value(i, 99);
        }
    }
    (buf, eob)
}

#[test]
fn p43_itx8_avx2_kurulum_155_yuva_tam() {
    const T: &str = "itx_avx2::p43_itx8_avx2_kurulum_155_yuva_tam";
    let Some((dsp, _)) = avx2_yuvalari(T) else {
        return;
    };

    for (tx, txtp) in desteklenen_155_yuva() {
        let alan = format!("itxfm_add[{}][{}]", tx.dav1d_name(), txtp.dav1d_name());
        assert_eq!(
            dsp.slot_implementation(BitDepth::Eight, "itx", &alan),
            "avx2",
            "8 bpc {alan} AVX2 olarak kurulmalı"
        );
    }
    assert_eq!(
        dsp.slot_implementation(BitDepth::Eight, "itx", "itxfm_add[TX_4X4][WHT_WHT]"),
        "avx2"
    );
}

#[test]
fn p43_itx8_avx2_oracle_small_ve_large_bit_exact() {
    const T: &str = "itx_avx2::p43_itx8_avx2_oracle_small_ve_large_bit_exact";
    let Some((_, tablo)) = avx2_yuvalari(T) else {
        return;
    };

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let small_dir = manifest.join("tests/fixtures/itx");
    let (small_non_wht, small_wht) =
        verify_oracle_8bpc(&small_dir, "small", &tablo).expect("itx-small-8 fixture bulunmalı");
    assert_eq!(
        small_non_wht, 930,
        "itx-small-8 içinde 930 WHT-dışı blok olmalı"
    );
    assert_eq!(small_wht, 6, "itx-small-8 içinde 6 WHT bloğu olmalı");

    let env_dir = std::env::var_os("AV1OXIDE_ITX_ORACLE_DIR")
        .or_else(|| std::env::var_os("ITX_ORACLE_DIR"))
        .map(PathBuf::from);
    let rel_cache = manifest.join("../../.cache/itx-oracle");
    let large_dir = if let Some(d) = env_dir.filter(|d| d.join("itx-large-8-input.bin").is_file()) {
        d
    } else {
        rel_cache
    };
    if let Some((large_non_wht, large_wht)) = verify_oracle_8bpc(&large_dir, "large", &tablo) {
        assert_eq!(
            large_non_wht, 155_000,
            "itx-large-8 içinde 155 000 WHT-dışı blok olmalı"
        );
        assert_eq!(large_wht, 1_000);
        println!(
            "[P43 ITX8 AVX2 oracle] small={small_non_wht} blok, large={large_non_wht} blok %100 bit-exact"
        );
    } else {
        println!(
            "[P43 ITX8 AVX2 oracle] small={small_non_wht} blok %100 bit-exact (large önbelleği yok)"
        );
    }
}

#[test]
fn p43_itx8_avx2_katsayi_sifirlama_sozlesmesi_155_yuva() {
    const T: &str = "itx_avx2::p43_itx8_avx2_katsayi_sifirlama_sozlesmesi_155_yuva";
    let Some((_, tablo)) = avx2_yuvalari(T) else {
        return;
    };

    for (tx, txtp) in desteklenen_155_yuva() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let f = tablo[tx.index()][txtp.index()];
        let mut dst = vec![128u8; w * h];

        let mut base = [0i32; 1024];
        for (i, v) in base.iter_mut().enumerate() {
            *v = tail_value(i, (tx.index() * 31 + txtp.index()) as u64);
            assert_ne!(*v, 0);
        }

        // 1) DC-only (`txtp == DctDct && eob == 0`): yalnız `coeff[0] = 0`, `coeff[1..1024]` dokunulmamış.
        if txtp == TxType::DctDct {
            let mut coeff = base;
            f(&mut dst, w, &mut coeff, 0, tx, txtp, BitDepth::Eight);
            assert_eq!(coeff[0], 0, "{tx:?}/{txtp:?} DC-only coeff[0] != 0");
            assert_eq!(
                &coeff[1..],
                &base[1..],
                "{tx:?}/{txtp:?} DC-only coeff[1..1024] değişmemeli"
            );
        }

        // 2) Genel yol (`eob >= 1` veya `txtp != DctDct`): tüm alt-blok eşikleri dâhil `coeff[..sw*sh] == 0`,
        //    `coeff[sw*sh..1024]` dokunulmamış.
        let mut eob_listesi = vec![
            1usize,
            35.min(sw * sh - 1),
            43.min(sw * sh - 1),
            106.min(sw * sh - 1),
            107.min(sw * sh - 1),
            135.min(sw * sh - 1),
            136.min(sw * sh - 1),
            150.min(sw * sh - 1),
            151.min(sw * sh - 1),
            278.min(sw * sh - 1),
            279.min(sw * sh - 1),
            sw * sh - 1,
        ];
        if txtp != TxType::DctDct {
            eob_listesi.push(0);
        }
        eob_listesi.sort_unstable();
        eob_listesi.dedup();

        for eob in eob_listesi {
            let mut coeff = base;
            f(&mut dst, w, &mut coeff, eob, tx, txtp, BitDepth::Eight);
            assert!(
                coeff[..sw * sh].iter().all(|&c| c == 0),
                "{tx:?}/{txtp:?} eob={eob} coeff[..{}] sıfırlanmadı",
                sw * sh
            );
            assert_eq!(
                &coeff[sw * sh..],
                &base[sw * sh..],
                "{tx:?}/{txtp:?} eob={eob} kuyruk coeff[{}..1024] bozuldu",
                sw * sh
            );
        }
    }
}

#[test]
fn p43_itx8_avx2_sentinel_ve_tekil_stride_denetimi() {
    const T: &str = "itx_avx2::p43_itx8_avx2_sentinel_ve_tekil_stride_denetimi";
    let Some((_, tablo)) = avx2_yuvalari(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();
    let skaler_tablo = ItxDsp::scalar_for(BitDepth::Eight).itxfm_add;
    let mut rng = Rng(0x5043_0002_2026_1117);

    for (tx, txtp) in desteklenen_155_yuva() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let f_avx2 = tablo[tx.index()][txtp.index()];
        let f_skaler = skaler_tablo[tx.index()][txtp.index()];

        for kip in [0u8, 1, 2] {
            let (c_base, eob) = uret_gecerli_katsayi(tx, txtp, kip, &mut rng);
            let _ = LNZC_TABLES[tx.index()][eob];

            for stride in [w, w + 1, w + 7, w + 13] {
                let data_len = (h - 1) * stride + w;
                let mut s_ref = SentinelBuffer::with_defaults(data_len);
                s_ref.data_mut().fill(0xA5);
                for y in 0..h {
                    for x in 0..w {
                        s_ref.data_mut()[y * stride + x] = match (kip + x as u8 + y as u8) % 4 {
                            0 => 0,
                            1 => 255,
                            _ => rng.aralik(0, 255) as u8,
                        };
                    }
                }
                let mut s_avx2 = s_ref.clone();
                let mut c_ref = c_base;
                let mut c_avx2 = c_base;

                f_skaler(
                    s_ref.data_mut(),
                    stride,
                    &mut c_ref,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Eight,
                );
                f_avx2(
                    s_avx2.data_mut(),
                    stride,
                    &mut c_avx2,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Eight,
                );

                assert_eq!(
                    s_avx2.verify(),
                    Ok(()),
                    "{tx:?}/{txtp:?} stride={stride} kip={kip} SentinelBuffer bozuldu"
                );
                // Satır arası dolgu baytları (`0xA5`) dokunulmamış olmalı.
                for y in 0..h - 1 {
                    for x in w..stride {
                        assert_eq!(
                            s_avx2.data()[y * stride + x],
                            0xA5,
                            "{tx:?}/{txtp:?} stride={stride} dolgu baytı ({x},{y}) bozuldu"
                        );
                    }
                }
                assert_eq!(
                    s_avx2.data(),
                    s_ref.data(),
                    "{tx:?}/{txtp:?} stride={stride} kip={kip} eob={eob} != skaler"
                );
                assert_eq!(
                    &c_avx2[..],
                    &c_ref[..],
                    "{tx:?}/{txtp:?} stride={stride} kip={kip} coeff != skaler"
                );
                if !(txtp == TxType::DctDct && eob == 0) {
                    assert_eq!(&c_avx2[sw * sh..], &c_base[sw * sh..]);
                }
            }
        }
    }

    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "sözleşme içi 155 yuva çağrılarında skaler devir 0 olmalı"
    );
}

#[test]
fn p43_itx8_avx2_sozlesme_disi_skaler_devir() {
    const T: &str = "itx_avx2::p43_itx8_avx2_sozlesme_disi_skaler_devir";
    let Some((_, tablo)) = avx2_yuvalari(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let skaler_tablo = ItxDsp::scalar_for(BitDepth::Eight).itxfm_add;

    let ornekler = [
        (TxSize::Tx4x4, TxType::DctDct),
        (TxSize::Rtx8x16, TxType::AdstFlipadst),
        (TxSize::Tx16x16, TxType::FlipadstDct),
        (TxSize::Rtx8x32, TxType::DctDct),
        (TxSize::Rtx16x32, TxType::Idtx),
        (TxSize::Rtx32x16, TxType::DctDct),
        (TxSize::Tx32x32, TxType::Idtx),
        (TxSize::Tx64x64, TxType::DctDct),
    ];

    for (tx, txtp) in ornekler {
        let f = tablo[tx.index()][txtp.index()];
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);

        // 1) Yanlış `BitDepth` (`Ten`): sayacı artırır ve 8 bpc skaler monomorfa devreder.
        let mut dst = vec![90u8; w * h];
        let mut dst_ref = dst.clone();
        let mut coeff = [0i32; 1024];
        coeff[0] = 200;
        let mut coeff_ref = coeff;
        let d0 = itx_skaler_devir_sayisi();
        f(&mut dst, w, &mut coeff, 0, tx, txtp, BitDepth::Ten);
        (skaler_tablo[tx.index()][txtp.index()])(
            &mut dst_ref,
            w,
            &mut coeff_ref,
            0,
            tx,
            txtp,
            BitDepth::Eight,
        );
        assert_eq!(
            itx_skaler_devir_sayisi(),
            d0 + 1,
            "{tx:?}/{txtp:?} yanlış BitDepth"
        );
        assert_eq!(dst, dst_ref);

        // 2) Yanlış `tx` (`Tx8x8` vs `Tx4x4`): sayacı artırır ve verilen `tx` skalerine devreder.
        let baska_tx = if tx == TxSize::Tx4x4 {
            TxSize::Tx8x8
        } else {
            TxSize::Tx4x4
        };
        let mut dst2 = vec![70u8; 64];
        let mut dst2_ref = dst2.clone();
        coeff[0] = 120;
        coeff_ref = coeff;
        let d1 = itx_skaler_devir_sayisi();
        f(
            &mut dst2,
            baska_tx.width(),
            &mut coeff,
            0,
            baska_tx,
            TxType::DctDct,
            BitDepth::Eight,
        );
        (skaler_tablo[baska_tx.index()][TxType::DctDct.index()])(
            &mut dst2_ref,
            baska_tx.width(),
            &mut coeff_ref,
            0,
            baska_tx,
            TxType::DctDct,
            BitDepth::Eight,
        );
        assert_eq!(
            itx_skaler_devir_sayisi(),
            d1 + 1,
            "{tx:?}/{txtp:?} yanlış tx"
        );
        assert_eq!(dst2, dst2_ref);

        // 3) Yanlış / desteklenmeyen `txtp` (`Tx64x64` üzerinde `AdstAdst`): paniklemeden döner.
        let mut dst64 = vec![50u8; 64 * 64];
        let d2 = itx_skaler_devir_sayisi();
        f(
            &mut dst64,
            64,
            &mut coeff,
            0,
            TxSize::Tx64x64,
            TxType::AdstAdst,
            BitDepth::Eight,
        );
        assert_eq!(
            itx_skaler_devir_sayisi(),
            d2 + 1,
            "{tx:?}/{txtp:?} desteklenmeyen txtp"
        );

        // 4) `eob >= sw * sh`: paniklemeden sayacı artırır ve döner.
        let mut dst3 = vec![100u8; w * h];
        let d3 = itx_skaler_devir_sayisi();
        f(&mut dst3, w, &mut coeff, sw * sh, tx, txtp, BitDepth::Eight);
        assert_eq!(
            itx_skaler_devir_sayisi(),
            d3 + 1,
            "{tx:?}/{txtp:?} eob == sw*sh"
        );

        // 5) `dst_stride < w`: paniklemeden sayacı artırır ve döner.
        let d4 = itx_skaler_devir_sayisi();
        f(&mut dst3, w - 1, &mut coeff, 0, tx, txtp, BitDepth::Eight);
        assert_eq!(
            itx_skaler_devir_sayisi(),
            d4 + 1,
            "{tx:?}/{txtp:?} kısa stride"
        );

        // 6) Kısa `dst` dilimi: paniklemeden sayacı artırır ve döner.
        let mut kisa_dst = vec![0u8; w * h - 1];
        let d5 = itx_skaler_devir_sayisi();
        f(&mut kisa_dst, w, &mut coeff, 0, tx, txtp, BitDepth::Eight);
        assert_eq!(
            itx_skaler_devir_sayisi(),
            d5 + 1,
            "{tx:?}/{txtp:?} kısa dst"
        );

        // 7) Kısa `coeff` dilimi (`< 1024`): paniklemeden sayacı artırır ve döner.
        let mut kisa_coeff = [0i32; 1023];
        let d6 = itx_skaler_devir_sayisi();
        f(&mut dst3, w, &mut kisa_coeff, 0, tx, txtp, BitDepth::Eight);
        assert_eq!(
            itx_skaler_devir_sayisi(),
            d6 + 1,
            "{tx:?}/{txtp:?} kısa coeff"
        );
    }
}

#[test]
fn p43_itx8_avx2_bozuk_akis_katsayi_fuzz_paniksiz() {
    const T: &str = "itx_avx2::p43_itx8_avx2_bozuk_akis_katsayi_fuzz_paniksiz";
    let Some((_, tablo)) = avx2_yuvalari(T) else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = itx_skaler_devir_sayisi();
    let mut rng = Rng(0xB020_7F43_2026_1117);

    for (tx, txtp) in desteklenen_155_yuva() {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        let stride = w + 7;
        let data_len = (h - 1) * stride + w;
        let f = tablo[tx.index()][txtp.index()];

        let eob_adaylari = [0usize, 1, (sw * sh / 4).max(1), sw * sh - 1];

        for &eob in &eob_adaylari {
            for desen in 0..8u8 {
                let mut coeff = [0i32; 1024];
                for (i, v) in coeff.iter_mut().enumerate() {
                    *v = match desen {
                        0 => 32767,
                        1 => -32767,
                        2 => -32768,
                        3 => {
                            if i & 1 == 0 {
                                32767
                            } else {
                                -32767
                            }
                        }
                        4 => {
                            if ((i / sh) + (i % sh)) & 1 == 0 {
                                32767
                            } else {
                                -32768
                            }
                        }
                        5 => {
                            if i & 1 == 0 {
                                i32::MAX
                            } else {
                                i32::MIN
                            }
                        }
                        _ => rng.aralik(-32768, 32767),
                    };
                }
                let coeff_once = coeff;

                let mut sbuf = SentinelBuffer::with_defaults(data_len);
                sbuf.data_mut().fill(0xA5);
                for y in 0..h {
                    for x in 0..w {
                        sbuf.data_mut()[y * stride + x] = match (desen + x as u8) % 3 {
                            0 => 0,
                            1 => 255,
                            _ => 128,
                        };
                    }
                }

                f(
                    sbuf.data_mut(),
                    stride,
                    &mut coeff,
                    eob,
                    tx,
                    txtp,
                    BitDepth::Eight,
                );

                assert_eq!(
                    sbuf.verify(),
                    Ok(()),
                    "{tx:?}/{txtp:?} eob={eob} desen={desen} SentinelBuffer bozuldu"
                );
                for y in 0..h - 1 {
                    for x in w..stride {
                        assert_eq!(
                            sbuf.data()[y * stride + x],
                            0xA5,
                            "{tx:?}/{txtp:?} eob={eob} desen={desen} dolgu baytı bozuldu"
                        );
                    }
                }
                if txtp == TxType::DctDct && eob == 0 {
                    assert_eq!(coeff[0], 0);
                    assert_eq!(&coeff[1..], &coeff_once[1..]);
                } else {
                    assert!(coeff[..sw * sh].iter().all(|&c| c == 0));
                    assert_eq!(&coeff[sw * sh..], &coeff_once[sw * sh..]);
                }
            }
        }
    }

    assert_eq!(
        itx_skaler_devir_sayisi(),
        devir0,
        "bozuk katsayı taramasında sözleşme içi çağrılarda skaler devir 0 olmalı"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn p43_itx8_avx2_isa_siniri_evex_yok() {
    let exe = std::env::current_exe().expect("test ikilisi yolu");
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&exe)
        .output()
        .expect("objdump çalışmalı");
    assert!(out.status.success(), "objdump başarısız");
    let metin = String::from_utf8_lossy(&out.stdout);

    let mut itx_sembol = 0usize;
    let mut itx_buyuk_sembol = 0usize;
    let mut aktif: Option<&str> = None;
    let mut aktif_ad = String::new();

    for satir in metin.lines() {
        if satir.ends_with(">:") {
            aktif_ad = satir.to_string();
            if satir.contains("itx_wht") || satir.contains("avx512") || satir.contains("ssse3") {
                aktif = None;
            } else if satir.contains("itx_buyuk") || satir.contains("itx8_buyuk_avx2_") {
                aktif = Some("itx_buyuk");
                itx_buyuk_sembol += 1;
            } else if satir.contains("3dsp3x863itx") || satir.contains("itx8_avx2_") {
                aktif = Some("itx");
                itx_sembol += 1;
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
            assert!(
                !sag.contains("%zmm") && !sag.contains("{%k"),
                "{modul} AVX2 sembolünde EVEX yönergesi: {aktif_ad} -> {satir}"
            );
        }
    }

    assert!(
        itx_sembol >= 140,
        "dsp::x86::itx içinde en az 140 monomorf sembol bulunmalı: {itx_sembol}"
    );
    assert!(
        itx_buyuk_sembol >= 15,
        "dsp::x86::itx_buyuk içinde en az 15 monomorf sembol bulunmalı: {itx_buyuk_sembol}"
    );
}
