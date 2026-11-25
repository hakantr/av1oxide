//! P36 (F4-MC e) — 8 bpc AVX512ICL `put`/`prep` gövdeleri (`mc[0..9]`, `mct[0..9]`).
//!
//! P32 AVX2 testinin (`mc_avx2_8bpc.rs`) AVX512ICL karşılığı: aynı boyut, alt piksel,
//! satır adımı ve desen kümesi; eşitlik + sentinel + okuma bölgesi + kuyruksuz
//! devir; P32 stride taramalı oracle'ı (`--ignored`, release) AVX512ICL yuvalarıyla
//! dav1d C'ye karşı. İşlemci AVX512ICL açmıyorsa testler ön koşul ilkesiyle atlanır.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::{Filter2d, MC_KUYRUK, McDsp, McPrepFn, McPutFn, skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

const NOBETCI: u8 = 0xA5;

/// Devir sayacı süreç geneldir: sayaca bakan testler sırayla koşar.
static SAYAC: std::sync::Mutex<()> = std::sync::Mutex::new(());
const NOBETCI16: i16 = 0x5A5A;

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
    fn byte(&mut self) -> u8 {
        (self.u64() >> 32) as u8
    }
}

/// AVX512ICL yuvaları (`put`, `prep`) ya da ön koşul eksikliği.
fn avx512_yuvalari(test: &str) -> Option<([McPutFn; 10], [McPrepFn; 10])> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx512icl_token()
    {
        use av1oxide::dsp::x86::mc_avx512::{prep_fn, put_fn};
        return Some((
            Filter2d::ALL.map(|f| put_fn(t, f)),
            Filter2d::ALL.map(|f| prep_fn(t, f)),
        ));
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P36] 8 bpc AVX512ICL MC gövdeleri AVX512ICL ister",
    );
    None
}

/// Kaynak: dolgu hücresinden `(h+6)·ss + w+7 + MC_KUYRUK` bayt; satır
/// adımı boşlukları ve kuyruk `dolgu` deseniyle.
fn kaynak(rng: &mut Rng, desen: u8, w: usize, h: usize, ss: usize, dolgu: u8) -> Vec<u8> {
    let n = (h + 6) * ss + w + 7 + MC_KUYRUK;
    (0..n)
        .map(|i| {
            let (r, c) = (i / ss, i % ss);
            if c >= w + 7 || r >= h + 7 {
                return dolgu;
            }
            match desen {
                0 => rng.byte(),
                1 => 0,
                2 => 255,
                _ => {
                    if (r + c) % 2 == 0 {
                        0
                    } else {
                        255
                    }
                }
            }
        })
        .collect()
}

/// Boyut × alt piksel kümesi: dar bloklarda bütün `mx`, `my`; geniş
/// bloklarda kenar ve ara konumlar.
fn konumlar(w: usize, h: usize) -> Vec<(u8, u8)> {
    if w * h <= 256 {
        (0..16u8)
            .flat_map(|mx| (0..16u8).map(move |my| (mx, my)))
            .collect()
    } else {
        let k = [0u8, 1, 4, 8, 11, 15];
        k.iter()
            .flat_map(|&mx| k.iter().map(move |&my| (mx, my)))
            .collect()
    }
}

fn boyutlar() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for w in [2usize, 4, 8, 16, 32, 64, 128] {
        for h in [2usize, 4, 8, 16, 32, 64, 128] {
            let uygun = match w {
                2 => h <= 8,
                4 => h <= 16,
                8 => h <= 32,
                16 => (4..=64).contains(&h),
                32 => (8..=64).contains(&h),
                64 => (16..=128).contains(&h),
                _ => h >= 64,
            };
            if uygun {
                v.push((w, h));
            }
        }
    }
    v
}

#[test]
fn p36_avx512_skalerle_bit_exact_ve_sentinel() {
    let Some((put, prep)) =
        avx512_yuvalari("mc_avx512_8bpc::p36_avx512_skalerle_bit_exact_ve_sentinel")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let skaler = McDsp::scalar::<8>();
    let devir0 = skaler_devir_sayisi();
    let mut rng = Rng(0x0F4_3C32);
    let mut n = 0usize;
    for (w, h) in boyutlar() {
        for (mi, (mx, my)) in konumlar(w, h).into_iter().enumerate() {
            // Desen ve adım konuma göre döner (kapsama, süre dengesi).
            let desen = (mi % 7).min(3) as u8;
            let ss = w + 7 + [0usize, 5, 32][mi % 3];
            let ds = w + [0usize, 3][mi % 2];
            let dolgu = [0u8, 255][mi % 2];
            let src = kaynak(&mut rng, desen, w, h, ss, dolgu);
            for f in Filter2d::ALL {
                let i = f.index();
                let ctx = format!("{f:?} {w}x{h} mx={mx} my={my} ss={ss} ds={ds} desen={desen}");
                // put: hedef satır aralarında ve sonunda nöbetçi.
                let dlen = (h - 1) * ds + w + 16;
                let mut beklenen = vec![NOBETCI; dlen];
                (skaler.mc[i])(
                    &mut beklenen,
                    ds,
                    &src,
                    ss,
                    w,
                    h,
                    mx,
                    my,
                    BitDepth::Eight,
                    255,
                );
                let mut d = vec![NOBETCI; dlen];
                (put[i])(&mut d, ds, &src, ss, w, h, mx, my, BitDepth::Eight, 255);
                assert_eq!(d, beklenen, "put {ctx}: skaler farkı ya da w ötesine yazma");
                // prep: `w·h` sonrasında nöbetçi.
                let mut tb = vec![NOBETCI16; w * h + 16];
                (skaler.mct[i])(&mut tb, &src, ss, w, h, mx, my, BitDepth::Eight, 255);
                let mut t = vec![NOBETCI16; w * h + 16];
                (prep[i])(&mut t, &src, ss, w, h, mx, my, BitDepth::Eight, 255);
                assert_eq!(t, tb, "prep {ctx}: skaler farkı ya da w·h ötesine yazma");
                assert!(
                    t[w * h..].iter().all(|&v| v == NOBETCI16),
                    "prep {ctx}: nöbetçi"
                );
                n += 2;
            }
        }
    }
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "sözleşmedeki çağrılar skalere devretmemeli"
    );
    println!("[P36] 8 bpc AVX512ICL put/prep: {n} çağrı skalerle bit-exact (20 yuva)");
}

/// Kaynak dilimi tam okuma bölgesi + `MC_KUYRUK` ile biter; kuyruk ve satır
/// boşlukları iki farklı desenle doldurulunca çıktı değişmez (bölge dışı
/// bayt çıktıya karışmaz) ve dilim sonu aşılmaz (aşım panik olurdu).
#[test]
fn p36_avx512_okuma_bolgesi_disi_ciktiya_karismaz() {
    let Some((put, prep)) =
        avx512_yuvalari("mc_avx512_8bpc::p36_avx512_okuma_bolgesi_disi_ciktiya_karismaz")
    else {
        return;
    };
    let mut n = 0usize;
    for (w, h) in boyutlar() {
        for (mx, my) in [(0u8, 0u8), (5, 0), (0, 11), (7, 9), (15, 15)] {
            let ss = w + 7;
            for f in Filter2d::ALL {
                let i = f.index();
                let mut cikti: Option<(Vec<u8>, Vec<i16>)> = None;
                for dolgu in [0u8, 255] {
                    let src = kaynak(&mut Rng(0x5EED + (w * h) as u64), 0, w, h, ss, dolgu);
                    assert_eq!(src.len(), (h + 6) * ss + w + 7 + MC_KUYRUK);
                    let mut d = vec![0u8; (h - 1) * w + w];
                    (put[i])(&mut d, w, &src, ss, w, h, mx, my, BitDepth::Eight, 255);
                    let mut t = vec![0i16; w * h];
                    (prep[i])(&mut t, &src, ss, w, h, mx, my, BitDepth::Eight, 255);
                    match &cikti {
                        None => cikti = Some((d, t)),
                        Some((d0, t0)) => {
                            assert_eq!(
                                &d, d0,
                                "put {f:?} {w}x{h} mx={mx} my={my}: dolgu çıktıya karıştı"
                            );
                            assert_eq!(
                                &t, t0,
                                "prep {f:?} {w}x{h} mx={mx} my={my}: dolgu çıktıya karıştı"
                            );
                        }
                    }
                    n += 2;
                }
            }
        }
    }
    println!("[P36] AVX512ICL okuma bölgesi + K sınırı: {n} çağrı");
}

/// Kuyruğu (`MC_KUYRUK`) olmayan dilim sözleşme dışıdır: giriş skaler gövdeye
/// devreder, sonucu skalerle aynıdır ve devir sayılır.
#[test]
fn p36_avx512_kuyruksuz_dilim_skalere_devreder() {
    let Some((put, prep)) =
        avx512_yuvalari("mc_avx512_8bpc::p36_avx512_kuyruksuz_dilim_skalere_devreder")
    else {
        return;
    };
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let skaler = McDsp::scalar::<8>();
    let (w, h, ss) = (8usize, 8usize, 15usize);
    let mut src = kaynak(&mut Rng(7), 0, w, h, ss, 0);
    src.truncate((h + 6) * ss + w + 7);
    let once = skaler_devir_sayisi();
    let mut d = vec![0u8; w * h];
    (put[0])(&mut d, w, &src, ss, w, h, 3, 5, BitDepth::Eight, 255);
    let mut b = vec![0u8; w * h];
    (skaler.mc[0])(&mut b, w, &src, ss, w, h, 3, 5, BitDepth::Eight, 255);
    assert_eq!(d, b);
    let mut t = vec![0i16; w * h];
    (prep[2])(&mut t, &src, ss, w, h, 3, 5, BitDepth::Eight, 255);
    let mut tb = vec![0i16; w * h];
    (skaler.mct[2])(&mut tb, &src, ss, w, h, 3, 5, BitDepth::Eight, 255);
    assert_eq!(t, tb);
    assert_eq!(
        skaler_devir_sayisi(),
        once + 2,
        "kuyruksuz dilim devri sayılmalı"
    );
}

/// P32 stride taramalı oracle (`scripts/oracle/mc_p32_oracle.c`, dav1d C):
/// genişlik 2..128 × en küçük/en büyük yükseklik × üç satır adımı × dört
/// piksel deseni (en büyük piksel dahil) × 10 filtre × `put`/`prep` × `mx`,
/// `my` 0..15. AVX512ICL ve skaler yuvaların hücre özeti dav1d C ile aynı olmalı.
/// Veri `.cache/mc-p32-oracle/` (üretim: `bash scripts/oracle/run_mc_oracle.sh`).
#[test]
#[ignore = "P32 stride taraması: .cache/mc-p32-oracle (--ignored, release)"]
fn p36_avx512_stride_taramasi_dav1d_c_ile_bit_exact() {
    let kok = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dizin = std::env::var_os("AV1OXIDE_MC_P32_ORACLE").map_or_else(
        || kok.join(".cache/mc-p32-oracle"),
        std::path::PathBuf::from,
    );
    let (Ok(girdi), Ok(beklenen)) = (
        std::fs::read(dizin.join("p32-input.bin")),
        std::fs::read(dizin.join("p32-expect.bin")),
    ) else {
        on_kosul::eksik(
            "mc_avx512_8bpc::p36_avx512_stride_taramasi_dav1d_c_ile_bit_exact",
            Neden::OracleVerisiYok,
            ".cache/mc-p32-oracle yok; üretmek için: bash scripts/oracle/run_mc_oracle.sh",
        );
        return;
    };
    let Some((put, prep)) =
        avx512_yuvalari("mc_avx512_8bpc::p36_avx512_stride_taramasi_dav1d_c_ile_bit_exact")
    else {
        return;
    };
    let skaler = McDsp::scalar::<8>();
    let u32le =
        |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
    assert_eq!(&girdi[..8], b"AV1OXMP1");
    assert_eq!(&beklenen[..8], b"AV1OXMP2");
    let ndz = u32le(&girdi, 12);
    let mut duzlemler = Vec::with_capacity(ndz);
    let mut o = 16usize;
    for _ in 0..ndz {
        let (w, h, ss, n) = (
            u32le(&girdi, o),
            u32le(&girdi, o + 4),
            u32le(&girdi, o + 8),
            u32le(&girdi, o + 16),
        );
        o += 20;
        let mut src = girdi[o..o + n].to_vec();
        src.resize(n + MC_KUYRUK, 0);
        o += n;
        duzlemler.push((w, h, ss, src));
    }
    assert_eq!(o, girdi.len(), "girdi uzunluğu");
    let fnv = |h: &mut u64, b: &[u8]| {
        for &x in b {
            *h ^= u64::from(x);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    let nhucre = u32le(&beklenen, 8);
    let _kilit = SAYAC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let devir0 = skaler_devir_sayisi();
    let mut cagri = 0usize;
    for c in 0..nhucre {
        let b = 12 + c * 20;
        let (dz, islev, f) = (
            u32le(&beklenen, b),
            u32le(&beklenen, b + 4),
            u32le(&beklenen, b + 8),
        );
        let want = u64::from_le_bytes(beklenen[b + 12..b + 20].try_into().expect("8 bayt"));
        let (w, h, ss, src) = &duzlemler[dz];
        let (w, h, ss) = (*w, *h, *ss);
        for (ad, avx) in [("avx512icl", true), ("skaler", false)] {
            let mut oz = 0xcbf2_9ce4_8422_2325u64;
            for mx in 0..16u8 {
                for my in 0..16u8 {
                    if islev == 0 {
                        let mut d = vec![0u8; w * h];
                        let k = if avx { put[f] } else { skaler.mc[f] };
                        k(&mut d, w, src, ss, w, h, mx, my, BitDepth::Eight, 255);
                        fnv(&mut oz, &d);
                    } else {
                        let mut t = vec![0i16; w * h];
                        let k = if avx { prep[f] } else { skaler.mct[f] };
                        k(&mut t, src, ss, w, h, mx, my, BitDepth::Eight, 255);
                        for v in t {
                            fnv(&mut oz, &v.to_le_bytes());
                        }
                    }
                    cagri += 1;
                }
            }
            assert_eq!(
                oz,
                want,
                "{ad} {} filtre {f} {w}x{h} ss={ss}: dav1d C özeti farklı",
                if islev == 0 { "put" } else { "prep" }
            );
        }
    }
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "stride taramasında skaler devir olmamalı"
    );
    println!(
        "[P36 stride] {ndz} düzlem · {nhucre} hücre · {cagri} çağrı (AVX512ICL + skaler) dav1d C ile bit-exact"
    );
}
