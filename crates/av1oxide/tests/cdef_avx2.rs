//! S15.4 — 7(b) AVX2 ↔ skaler **bit-exact** ve 7(c) sınır dışı okuma/yazma yok.
//!
//! ## 7(b) kapsamı
//!
//! Her (işlev, bit derinliği, blok boyutu) hücresi için sabit tohumlu rastgele
//! bloklar üretilir ve **aynı girdi** hem skaler hem AVX2 yolundan geçirilip
//! bit-exact karşılaştırılır. AVX2 yolu 8-bit içindir; 10/12-bit hücreler
//! güvenli sarmalayıcının **skaler devir** yolunu sınar (S15.4 kapsamı) ve bu
//! sonuç raporda açıkça belirtilir.
//!
//! Sentetik üretim **tahminî değildir**: üretilen parametreler gerçek AV1
//! sözleşmesine uyar (`damping >= ulog2(sec_strength)`, `!(pri == 0 && sec == 0)`,
//! güçler bit derinliği ölçeğiyle).
//!
//! ## 7(c) yöntemi
//!
//! - [`SentinelBuffer`]: `dst` bölgesinin önüne/arkasına koruyucu koyar →
//!   **yazma taşması** yakalanır.
//! - [`OverReadProbe`]: aynı veri iki farklı padding deseniyle koşturulur →
//!   veri aralığı dışından **okuma** çıktıyı değiştirir ve yakalanır.
//!
//! Her iki ölçüm **her iki yol** (skaler + AVX2) için yapılır ve AVX2 yoksa
//! test açıkça atlanır (gerekçe yazılır).

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::cdef::{CdefBitDepth, CdefBlockSize, CdefEdgeFlags};
use av1oxide::dsp::scalar::cdef as scalar;
use common::cdef_fixture::{
    Case, Rng, bit_depth, block_size, edges, params, random_case, run_dir, run_filter, scalar_dir,
    scalar_filter,
};
use common::on_kosul::{self, Neden};
use common::sentinel::{OverReadProbe, SentinelBuffer, VARSAYILAN_KORUMA};

/// Hücre başına rastgele blok sayısı (S15.4/7: en az 10 000).
const HUCRE_BLOK: usize = 10_000;

/// AVX2 yolu bu platformda kullanılabilir mi?
fn avx2_var() -> bool {
    // F3b: AVX2 yolu yalnız kurulumda üretilen belirteçle açılır.
    av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect())
        .avx2_token()
        .is_some()
}

/// `bd` bit derinliğinin AVX2 CDEF gövdeleri (`dir`, `fb`): P30'dan beri
/// yuva daha üst bir düzeyin (AVX512ICL) gövdesini tutabileceğinden, gövde
/// yuvadan değil AVX2 adayından alınır. AVX2 adayı yoksa (AVX2'siz işlemci,
/// x86_64 dışı hedef) `None`.
fn avx2_govde(bd: u8) -> Option<common::cdef_fixture::IsaGovde> {
    common::cdef_fixture::isa_govdeleri(bd)
        .0
        .into_iter()
        .find(|g| g.ad == "avx2")
}

/// AVX2 `dir` çekirdeği (P29: 10/12 bit 16 bpc AVX2 gövdesi). Yalnız
/// `avx2_var()` korumasından **sonra** çağrılır (yoksa panik).
fn avx2_dir(bd: u8) -> av1oxide::dsp::cdef::CdefDirFn {
    avx2_govde(bd).and_then(|g| g.dir).expect("AVX2 dir")
}

/// AVX2 `filter` çekirdeği; AVX2 girişi üç blok boyutunu taşır. Yalnız
/// `avx2_var()` korumasından **sonra** çağrılır (yoksa panik).
fn avx2_filter(bd: u8) -> av1oxide::dsp::cdef::CdefFilterFn {
    avx2_govde(bd).and_then(|g| g.fb).expect("AVX2 fb")
}

/// Yol adı ve gövdeyi alan işlev (gövde yoksa `None`).
type Yol<K> = (&'static str, fn() -> Option<K>);

/// 7(c) yolları: yol listesi gövdeyi değil gövdeyi **alan** işlevi taşır; AVX2
/// yolu ön koşul korumasından (`avx2`) **sonra** alınır. aarch64'te AVX2 adayı
/// yoktur; gövde korumadan önce alınınca test `on_kosul::eksik`'e varmadan
/// panikliyordu (P30 düzeltme 1, bulgu a). Eksik yol `eksik` ile bildirilip
/// atlanır.
fn yollari_sec<K>(
    yollar: [Yol<K>; 2],
    avx2: bool,
    mut eksik: impl FnMut(),
) -> Vec<(&'static str, K)> {
    let mut secilen = Vec::with_capacity(yollar.len());
    for (ad, al) in yollar {
        if ad == "avx2" && !avx2 {
            eksik();
            continue;
        }
        secilen.push((ad, al().unwrap_or_else(|| panic!("{ad} gövdesi yok"))));
    }
    secilen
}

/// 7(b) — `dir`: her bit derinliği için ≥ 10 000 blok, AVX2 ↔ skaler.
#[test]
fn yedi_b_dir_avx2_skaler_ile_bit_ayni() {
    if !avx2_var() {
        on_kosul::eksik(
            "cdef_avx2::yedi_b_dir_avx2_skaler_ile_bit_ayni",
            Neden::avx2(),
            "[7(b) dir] AVX2 yolu; skaler yol koşar",
        );
        return;
    }
    let mut rng = Rng::new(0x5A15_04C0_7B00_0001);
    let patterns = [0u8, 1, 2, 3];
    let mut toplam = 0usize;
    for bd in [8u8, 10, 12] {
        let mut fark = 0usize;
        for i in 0..HUCRE_BLOK {
            let case = random_case(&mut rng, 0, bd, 0, patterns[i % patterns.len()]);
            let a = run_dir(&case, scalar_dir(&case));
            let b = run_dir(&case, avx2_dir(bd));
            if a != b {
                fark += 1;
                if fark == 1 {
                    println!(
                        "  ilk fark ({bd}-bit, blok {i}): skaler={a:02x?} avx2={b:02x?} \
                         (girdi {:02x?})",
                        &case.input[..16]
                    );
                }
            }
            toplam += 1;
        }
        assert_eq!(fark, 0, "{bd}-bit dir: {fark}/{HUCRE_BLOK} blok farklı");
        println!("  dir {bd:2}-bit : {HUCRE_BLOK} blok AVX2 == skaler (OK)");
    }
    println!("[7(b) dir] toplam {toplam} blok karşılaştırıldı");
}

/// 7(b) — `filter`: her (bit derinliği, blok boyutu) hücresi için ≥ 10 000 blok.
#[test]
fn yedi_b_filter_avx2_skaler_ile_bit_ayni() {
    if !avx2_var() {
        on_kosul::eksik(
            "cdef_avx2::yedi_b_filter_avx2_skaler_ile_bit_ayni",
            Neden::avx2(),
            "[7(b) filter] AVX2 yolu; skaler yol koşar",
        );
        return;
    }
    let mut rng = Rng::new(0x5A15_04C0_7B00_0002);
    let patterns = [0u8, 1, 2, 3];
    let mut toplam = 0usize;
    for bd in [8u8, 10, 12] {
        for size in 0u8..3 {
            let mut fark = 0usize;
            for i in 0..HUCRE_BLOK {
                let case = random_case(&mut rng, 1, bd, size, patterns[i % patterns.len()]);
                let a = run_filter(&case, scalar_filter(&case));
                let b = run_filter(&case, avx2_filter(case.bit_depth));
                if a != b {
                    fark += 1;
                    if fark == 1 {
                        println!(
                            "  ilk fark (bd={bd}, blok={size}, örnek {i}, kenar={:#06b}, \
                             pri={}, sec={}, dir={}, damping={})",
                            case.edges, case.pri, case.sec, case.dir, case.damping
                        );
                    }
                }
                toplam += 1;
            }
            let blok = block_size(size);
            assert_eq!(
                fark,
                0,
                "{bd}-bit {}x{} filter: {fark}/{HUCRE_BLOK} blok farklı",
                blok.w(),
                blok.h()
            );
            println!(
                "  filter {bd:2}-bit {}x{} : {HUCRE_BLOK} blok AVX2 == skaler (OK)",
                blok.w(),
                blok.h()
            );
        }
    }
    println!("[7(b) filter] toplam {toplam} blok karşılaştırıldı");
}

/// 7(b) — bağımsız olarak: AVX2 çekirdekleri **oracle fixture'ıyla** da aynı
/// sonucu vermeli (skaler zaten oracle'a eşit; bu test zincirin kapanışıdır).
#[test]
fn yedi_b_avx2_oracle_fixtureinda_da_bit_ayni() {
    if !avx2_var() {
        on_kosul::eksik(
            "cdef_avx2::yedi_b_avx2_oracle_fixtureinda_da_bit_ayni",
            Neden::avx2(),
            "[7(b) oracle] AVX2 yolu; skaler yol koşar",
        );
        return;
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("cdef");
    let input = std::fs::read(dir.join("cdef-small-input.bin")).expect("küçük girdi");
    let expect = std::fs::read(dir.join("cdef-small-expect.bin")).expect("küçük beklenen");
    let cases = common::cdef_fixture::read_cases(&input, &expect);
    let mut fark = 0usize;
    for case in &cases {
        let got = if case.kind == 0 {
            run_dir(case, avx2_dir(case.bit_depth))
        } else {
            run_filter(case, avx2_filter(case.bit_depth))
        };
        if got != case.expect {
            fark += 1;
        }
    }
    assert_eq!(
        fark,
        0,
        "AVX2 ↔ oracle: {fark}/{} kayıt farklı",
        cases.len()
    );
    println!(
        "[7(b) oracle] {} kayıt AVX2 == dav1d C (transitif: AVX2 == skaler == C)",
        cases.len()
    );
}

/// 7(b) — zorlanmış skaler maske: `DspDispatch` AVX2 biti silinince skaler
/// çekirdeği gösterir ve **aynı girdide aynı çıktıyı** verir (DSP-01/5).
#[test]
fn yedi_b_zorlanmis_skaler_maske_ayni_cikti() {
    use av1oxide::cpu::CpuFlags;
    use av1oxide::dsp::DspDispatch;

    let tam = DspDispatch::select(CpuFlags::detect());
    // P30: SSSE3/SSE4.1 gövdeleri de olduğundan zorlanmış skaler maske SSSE3
    // ve üstünü kapatır (dav1d `sse2` düzeyi).
    let maskeli = DspDispatch::select(
        CpuFlags::detect().intersection(av1oxide::cpu::CpuLevel::Sse2.allowed()),
    );
    // F3c: maske 8 bit tablosunun `dir`/`fb` yuvalarını skaler bırakır. P30:
    // yuva kimliği seçim kaydıyla denetlenir (işlev adresi optimize derlemede
    // kimlik kanıtı değildir); davranış eşitliği aşağıda.
    let skaler = av1oxide::dsp::cdef::CdefDsp::scalar::<8>();
    for field in ["dir", "fb[0]", "fb[1]", "fb[2]"] {
        assert_eq!(
            maskeli.slot_implementation(av1oxide::frame::BitDepth::Eight, "cdef", field),
            "skaler",
            "maske ISA yolunu kapatmalı ({field})"
        );
    }
    // P43/P46/P47: `CpuLevel::Sse2` düzeyinde CDEF yuvaları skaler kalır; `Sse2Token`
    // varsa 3 `itx_2d[TX_4X4][WHT_WHT]` yuvası (8/10/12 bpc), `refmvs/splat_mv`,
    // 2 `looprestoration/wiener[0..2]@8` ve 7 `msac/*` (`13` yuva) atanır.
    let beklenen_atanan = if maskeli.sse2_token().is_some() {
        13
    } else {
        0
    };
    assert_eq!(maskeli.assigned_count(), beklenen_atanan);
    assert_eq!(DspDispatch::select(CpuFlags::NONE).assigned_count(), 0);

    let mut rng = Rng::new(0x5A15_04C0_7B00_0003);
    for i in 0..256usize {
        let case = random_case(&mut rng, 1, 8, (i % 3) as u8, (i % 4) as u8);
        let a = run_filter(
            &case,
            tam.for_bit_depth(av1oxide::frame::BitDepth::Eight)
                .cdef()
                .fb[0],
        );
        let b = run_filter(
            &case,
            maskeli
                .for_bit_depth(av1oxide::frame::BitDepth::Eight)
                .cdef()
                .fb[0],
        );
        assert_eq!(
            a, b,
            "zorlanmış skaler maske farklı çıktı verdi (örnek {i})"
        );
        assert_eq!(
            b,
            run_filter(&case, skaler.fb[0]),
            "skaler gövdeyle aynı (örnek {i})"
        );
    }
    let mut rng = Rng::new(0x5A15_04C0_7B00_0004);
    for i in 0..256usize {
        let case = random_case(&mut rng, 0, 8, 0, (i % 4) as u8);
        let a = run_dir(
            &case,
            tam.for_bit_depth(av1oxide::frame::BitDepth::Eight)
                .cdef()
                .dir,
        );
        let b = run_dir(
            &case,
            maskeli
                .for_bit_depth(av1oxide::frame::BitDepth::Eight)
                .cdef()
                .dir,
        );
        assert_eq!(a, b, "zorlanmış skaler maske farklı yön verdi (örnek {i})");
        assert_eq!(
            b,
            run_dir(&case, skaler.dir),
            "skaler gövdeyle aynı (örnek {i})"
        );
    }
    println!("[7(b) maske] zorlanmış skaler maske ile aynı çıktı (512 örnek)");
}

/// 7(c) — sentinel: `dst` bölgesine yazma, koruyucu bölgeye **taşmamalı**.
#[test]
fn yedi_c_sentinel_yazma_tasmasi_yok() {
    let mut rng = Rng::new(0x5A15_04C0_7C00_0001);
    let yollar = yollari_sec::<av1oxide::dsp::cdef::CdefFilterFn>(
        [
            ("skaler", || Some(scalar::filter::<8>)),
            ("avx2", || avx2_govde(8).and_then(|g| g.fb)),
        ],
        avx2_var(),
        || {
            on_kosul::eksik(
                "cdef_avx2::yedi_c_sentinel_yazma_tasmasi_yok[avx2]",
                Neden::avx2(),
                "[7(c) sentinel:avx2] AVX2 yolu; skaler yol koşar",
            );
        },
    );
    let mut toplam = 0usize;
    for (ad, kernel) in yollar {
        for i in 0..512usize {
            let case = random_case(&mut rng, 1, 8, (i % 3) as u8, (i % 4) as u8);
            let (w, h) = (
                block_size(case.block_size).w(),
                block_size(case.block_size).h(),
            );
            let (dst_in, left, top, bottom) = common::cdef_fixture::slices(&case);
            let mut buffer = SentinelBuffer::new(h * case.stride, VARSAYILAN_KORUMA, 0xA5);
            assert!(buffer.write_data(dst_in));
            {
                let dst = buffer.data_mut();
                kernel(dst, case.stride, left, top, bottom, params(&case));
            }
            if let Err(violation) = buffer.verify() {
                panic!(
                    "[7(c) sentinel:{ad}] koruyucu ihlali: {violation:?} \
                     (blok {w}x{h}, örnek {i}, kenar={:#06b})",
                    case.edges
                );
            }
            // Yazılan bölge gerçekten güncellenmiş olmalı (boş test olmasın).
            let yazilan = &buffer.data()[..w];
            assert!(
                yazilan.iter().any(|v| *v != 0xA5),
                "[7(c) sentinel:{ad}] çekirdek hiç yazmadı (örnek {i})"
            );
            toplam += 1;
        }
        println!("[7(c) sentinel:{ad}] 512 blok, koruyucu ihlali yok (OK)");
    }
    assert!(toplam > 0, "hiç koşum yapılmadı");
}

/// 7(c) — over-read: iki farklı padding deseni **aynı** çıktıyı vermeli.
#[test]
fn yedi_c_overread_veri_disindan_okuma_yok() {
    let yollar = yollari_sec::<av1oxide::dsp::cdef::CdefFilterFn>(
        [
            ("skaler", || Some(scalar::filter::<8>)),
            ("avx2", || avx2_govde(8).and_then(|g| g.fb)),
        ],
        avx2_var(),
        || {
            on_kosul::eksik(
                "cdef_avx2::yedi_c_overread_veri_disindan_okuma_yok[avx2]",
                Neden::avx2(),
                "[7(c) over-read:avx2] AVX2 yolu; skaler yol koşar",
            );
        },
    );
    let mut rng = Rng::new(0x5A15_04C0_7C00_0002);
    for (ad, kernel) in yollar {
        for i in 0..256usize {
            let case = random_case(&mut rng, 1, 8, (i % 3) as u8, (i % 4) as u8);
            let bpp = 1usize;
            let (w, h) = (
                block_size(case.block_size).w(),
                block_size(case.block_size).h(),
            );
            let dst_len = h * case.stride;
            let left_len = h * 2 * bpp;
            let edge_len = 2 * case.stride;
            let in_len = dst_len + left_len + 2 * edge_len;
            let out_len = h * w * bpp;
            // Çekirdeğin **sözleşmeli** girdi yerleşimi tek bir yük olarak
            // padding'in ortasına konur; padding'den bir bayt okunursa çıktı değişir.
            let probe = OverReadProbe::new(in_len, 64, out_len);
            let offset = probe.data_offset();
            let payload = probe.payload();
            let params_ref = params(&case);
            let stride = case.stride;
            let sonuc = probe.run(|input, out| {
                let block = &input[offset..offset + in_len];
                let mut dst = block[..dst_len].to_vec();
                let left = &block[dst_len..dst_len + left_len];
                let top = &block[dst_len + left_len..dst_len + left_len + edge_len];
                let bottom = &block[dst_len + left_len + edge_len..];
                kernel(&mut dst, stride, left, top, bottom, params_ref);
                for y in 0..h {
                    out[y * w * bpp..(y + 1) * w * bpp]
                        .copy_from_slice(&dst[y * stride..y * stride + w * bpp]);
                }
            });
            if let Err(over) = sonuc {
                panic!(
                    "[7(c) over-read:{ad}] veri dışından okuma: {over:?} \
                     (blok {w}x{h}, örnek {i}, kenar={:#06b}, pri={}, sec={})",
                    case.edges, case.pri, case.sec
                );
            }
            let _ = payload;
        }
        println!("[7(c) over-read:{ad}] 256 blok, padding okunmadı (OK)");
    }
}

/// 7(c) — `dir` çekirdeği 8x8 bloğun **tam** dışına çıkmamalı.
#[test]
fn yedi_c_dir_overread_yok() {
    let yollar = yollari_sec::<av1oxide::dsp::cdef::CdefDirFn>(
        [
            ("skaler", || Some(scalar::find_dir::<8>)),
            ("avx2", || avx2_govde(8).and_then(|g| g.dir)),
        ],
        avx2_var(),
        || {
            on_kosul::eksik(
                "cdef_avx2::yedi_c_dir_overread_yok[avx2]",
                Neden::avx2(),
                "[7(c) dir over-read:avx2] AVX2 yolu; skaler yol koşar",
            );
        },
    );
    let mut rng = Rng::new(0x5A15_04C0_7C00_0003);
    for (ad, kernel) in yollar {
        for i in 0..256usize {
            let case = random_case(&mut rng, 0, 8, 0, (i % 4) as u8);
            let probe = OverReadProbe::new(case.input.len(), 64, 8);
            let offset = probe.data_offset();
            let len = case.input.len();
            let stride = case.stride;
            let sonuc = probe.run(|input, out| {
                let r = kernel(&input[offset..offset + len], stride, CdefBitDepth::B8);
                out[0] = r.dir;
                out[1..5].copy_from_slice(&r.var.to_le_bytes());
            });
            if let Err(over) = sonuc {
                panic!("[7(c) dir over-read:{ad}] veri dışından okuma: {over:?} (örnek {i})");
            }
        }
        println!("[7(c) dir over-read:{ad}] 256 blok, padding okunmadı (OK)");
    }
}

/// P30 düzeltme 1 (bulgu a): AVX2'siz ortamda (aarch64, AVX2'siz x86) 7(c)
/// yol seçimi AVX2 gövdesini **istemez**, yolu `eksik` ile bildirir ve skaler
/// yolu koşturur. x86'da taklit: koruma "yok" der, AVX2 alıcısı çağrılırsa
/// panikler (gövde korumadan önce alınırsa bu test düşer; mutasyon M1).
#[test]
fn p30d1_avx2siz_ortamda_govde_korumadan_once_istenmez() {
    fn istenmemeli() -> Option<av1oxide::dsp::cdef::CdefFilterFn> {
        panic!("AVX2 gövdesi ön koşul korumasından önce istendi")
    }
    let mut bildirim = 0usize;
    let yollar = yollari_sec::<av1oxide::dsp::cdef::CdefFilterFn>(
        [
            ("skaler", || Some(scalar::filter::<8>)),
            ("avx2", istenmemeli),
        ],
        false,
        || bildirim += 1,
    );
    assert_eq!(bildirim, 1, "eksik AVX2 yolu bir kez bildirilmeli");
    assert_eq!(
        yollar.iter().map(|(ad, _)| *ad).collect::<Vec<_>>(),
        ["skaler"],
        "yalnız skaler yol koşmalı"
    );
    // AVX2 varken iki yol da seçilir (taklit gövde: skaler).
    let iki = yollari_sec::<av1oxide::dsp::cdef::CdefFilterFn>(
        [
            ("skaler", || Some(scalar::filter::<8>)),
            ("avx2", || Some(scalar::filter::<8>)),
        ],
        true,
        || panic!("AVX2 varken eksik bildirilmemeli"),
    );
    assert_eq!(iki.len(), 2);
}

/// 7(c) — negatif kontrol: sentinel gerçekten **çalışıyor** mu?
///
/// Kasıtlı bir taşma üretilir ve yakalandığı doğrulanır; aksi hâlde yukarıdaki
/// "ihlal yok" sonuçları hiçbir şey kanıtlamaz.
#[test]
fn yedi_c_sentinel_negatif_kontrol_calisiyor() {
    let mut buffer = SentinelBuffer::new(16, 8, 0xA5);
    assert_eq!(buffer.verify(), Ok(()));
    buffer.raw_mut()[0] = 0x00; // ön koruyucuya kasıtlı yazma
    assert!(buffer.verify().is_err(), "kasıtlı taşma yakalanmalı");
    let mut buffer = SentinelBuffer::new(16, 8, 0xA5);
    let son = buffer.raw_mut().len() - 1;
    buffer.raw_mut()[son] = 0x00; // arka koruyucuya kasıtlı yazma
    assert!(buffer.verify().is_err(), "arka taşma yakalanmalı");
}

/// 7(c) — sentetik `Case` üreteci sözleşmeye uygun mu? (test altyapısının testi)
#[test]
fn yedi_c_uretec_sozlesmeye_uygun() {
    let mut rng = Rng::new(0x5A15_04C0_7C00_0004);
    for i in 0..2048usize {
        let bd = [8u8, 10, 12][i % 3];
        let size = (i % 3) as u8;
        let case = random_case(&mut rng, 1, bd, size, (i % 4) as u8);
        let p = params(&case);
        assert!(
            !(p.pri_strength == 0 && p.sec_strength == 0),
            "sözleşme dışı güç"
        );
        if p.sec_strength != 0 {
            let ulog = 31 - u32::from(p.sec_strength).leading_zeros();
            assert!(
                u32::from(p.damping) >= ulog,
                "sec_shift negatif olurdu (C'de UB): damping={} sec={}",
                p.damping,
                p.sec_strength
            );
        }
        assert!(p.dir < 8);
        assert!(u32::from(p.pri_strength) <= 15 << bit_depth(bd).min_8());
        assert!(u32::from(p.sec_strength) <= 4 << bit_depth(bd).min_8());
        assert!(CdefEdgeFlags::from_bits(case.edges).is_some());
        assert_eq!(case.stride, {
            let b = block_size(size);
            (b.w() + 4) * bit_depth(bd).bytes()
        });
        let _: Case = case.clone();
        let _ = edges(&case);
        let _ = CdefBlockSize::ALL.contains(&block_size(size));
    }
    println!("[7(c) üreteç] 2048 örnek AV1 sözleşmesine uygun");
}

/// F4-CDEF a — `dir` uç girdiler: sabit 0/255, satranç tahtaları, satır/sütun
/// şeritleri ve her yönü baskın kılan çizgi desenleri; AVX2 (64 piksel tek
/// geçiş, 256 bit eşlenmiş kısmi toplamlar) skalerle aynı `(dir, var)`.
#[test]
fn f4_dir_uc_girdiler_skaler_ile_ayni() {
    if !avx2_var() {
        on_kosul::eksik(
            "cdef_avx2::f4_dir_uc_girdiler_skaler_ile_ayni",
            Neden::avx2(),
            "[F4 dir uç] AVX2 yolu; skaler yol koşar",
        );
        return;
    }
    let mut desenler: Vec<[u8; 64]> = Vec::new();
    for v in [0u8, 1, 127, 128, 254, 255] {
        desenler.push([v; 64]);
    }
    for (a, b) in [(0u8, 255u8), (255, 0), (16, 240)] {
        desenler.push(core::array::from_fn(|i| {
            if (i / 8 + i % 8) % 2 == 0 { a } else { b }
        }));
        desenler.push(core::array::from_fn(
            |i| if (i / 8) % 2 == 0 { a } else { b },
        ));
        desenler.push(core::array::from_fn(
            |i| if (i % 8) % 2 == 0 { a } else { b },
        ));
    }
    // Yön çizgileri: (y·p + x·q) mod 8 sabit olan pikseller parlak.
    for (p, q) in [
        (1i32, 0i32),
        (0, 1),
        (1, 1),
        (1, -1),
        (2, 1),
        (1, 2),
        (2, -1),
        (1, -2),
    ] {
        for faz in 0..8 {
            desenler.push(core::array::from_fn(|i| {
                let (y, x) = ((i / 8) as i32, (i % 8) as i32);
                if (y * p + x * q).rem_euclid(8) == faz {
                    255
                } else {
                    0
                }
            }));
        }
    }
    let mut n = 0;
    for d in &desenler {
        for stride in [8usize, 16] {
            let mut img = vec![0x5Au8; 7 * stride + 8];
            for y in 0..8 {
                img[y * stride..y * stride + 8].copy_from_slice(&d[y * 8..y * 8 + 8]);
            }
            let a = scalar::find_dir::<8>(&img, stride, CdefBitDepth::B8);
            let b = avx2_dir(8)(&img, stride, CdefBitDepth::B8);
            assert_eq!(
                (a.dir, a.var),
                (b.dir, b.var),
                "dir uç girdi {d:?} stride={stride}"
            );
            n += 1;
        }
    }
    println!("[F4 dir uç] {n} uç blok AVX2 == skaler");
}

/// F4-CDEF a — `filter` kenar yolu: 3 boyut × 16 kenar birleşimi, her biri
/// rastgele 400 blok, eksik kenar dilimi hem tam uzunluklu hem **boş** (çözme
/// hattı biçimi); AVX2 (maskeli kenar yolu + dört kenarlı hızlı yol) skalerle
/// bit-aynı.
#[test]
fn f4_filter_kenar_yolu_skaler_ile_ayni() {
    if !avx2_var() {
        on_kosul::eksik(
            "cdef_avx2::f4_filter_kenar_yolu_skaler_ile_ayni",
            Neden::avx2(),
            "[F4 filter kenar] AVX2 yolu; skaler yol koşar",
        );
        return;
    }
    let mut rng = Rng::new(0xF4CD_EF0A_0000_0001);
    let mut n = 0usize;
    for size_code in 0u8..3 {
        for e in 0u8..16 {
            for i in 0..400usize {
                let mut case = random_case(&mut rng, 1, 8, size_code, (i % 4) as u8);
                case.edges = e;
                let a = run_filter(&case, scalar_filter(&case));
                let b = run_filter(&case, avx2_filter(case.bit_depth));
                assert_eq!(a, b, "filter {size_code} kenar {e:04b} blok {i}");
                // Boş dilimler (eksik kenar): skaler ve AVX2 aynı sonucu verir.
                let (w, h) = (block_size(size_code).w(), block_size(size_code).h());
                let ef = CdefEdgeFlags::from_bits(e).expect("kenar");
                let (dst0, left, top, bottom) = common::cdef_fixture::slices(&case);
                let bos: &[u8] = &[];
                let l = if ef.contains(CdefEdgeFlags::HAVE_LEFT) {
                    left
                } else {
                    bos
                };
                let t = if ef.contains(CdefEdgeFlags::HAVE_TOP) {
                    top
                } else {
                    bos
                };
                let bt = if ef.contains(CdefEdgeFlags::HAVE_BOTTOM) {
                    bottom
                } else {
                    bos
                };
                let mut d1 = dst0.to_vec();
                let mut d2 = dst0.to_vec();
                scalar_filter(&case)(&mut d1, case.stride, l, t, bt, params(&case));
                avx2_filter(case.bit_depth)(&mut d2, case.stride, l, t, bt, params(&case));
                for y in 0..h {
                    let r = y * case.stride..y * case.stride + w;
                    assert_eq!(
                        d1[r.clone()],
                        d2[r.clone()],
                        "boş dilim {size_code} {e:04b} {i}"
                    );
                    assert_eq!(d2[r], a[y * w..y * w + w], "boş dilim = tam dilim {e:04b}");
                }
                n += 1;
            }
        }
    }
    println!("[F4 filter kenar] {n} blok × 2 dilim biçimi AVX2 == skaler");
}

/// P29 — 16 bpc `dir` uç girdileri (10 ve 12 bit): sabit 0 / 1 / orta /
/// en büyük − 1 / en büyük, dama ve şerit desenleri, sekiz yönde en büyük
/// genlikli çizgiler; iki satır adımı. Skaler ile aynı (`dir`, `var`).
#[test]
fn f4b_dir16_uc_girdiler_skaler_ile_ayni() {
    if !avx2_var() {
        on_kosul::eksik(
            "cdef_avx2::f4b_dir16_uc_girdiler_skaler_ile_ayni",
            Neden::avx2(),
            "[F4b dir16 uç] AVX2 yolu; skaler yol koşar",
        );
        return;
    }
    let mut n = 0;
    for (bd, cbd) in [(10u8, CdefBitDepth::B10), (12, CdefBitDepth::B12)] {
        let mx = (1u16 << bd) - 1;
        let mut desenler: Vec<[u16; 64]> = Vec::new();
        for v in [0u16, 1, mx / 2, mx / 2 + 1, mx - 1, mx] {
            desenler.push([v; 64]);
        }
        for (a, b) in [(0u16, mx), (mx, 0), (mx / 16, mx - mx / 16)] {
            desenler.push(core::array::from_fn(|i| {
                if (i / 8 + i % 8) % 2 == 0 { a } else { b }
            }));
            desenler.push(core::array::from_fn(
                |i| if (i / 8) % 2 == 0 { a } else { b },
            ));
            desenler.push(core::array::from_fn(
                |i| if (i % 8) % 2 == 0 { a } else { b },
            ));
        }
        for (p, q) in [
            (1i32, 0i32),
            (0, 1),
            (1, 1),
            (1, -1),
            (2, 1),
            (1, 2),
            (2, -1),
            (1, -2),
        ] {
            for faz in 0..8 {
                desenler.push(core::array::from_fn(|i| {
                    let (y, x) = ((i / 8) as i32, (i % 8) as i32);
                    if (y * p + x * q).rem_euclid(8) == faz {
                        mx
                    } else {
                        0
                    }
                }));
            }
        }
        for d in &desenler {
            for stride in [16usize, 32] {
                let mut img = vec![0x5Au8; 7 * stride + 16];
                for y in 0..8 {
                    for x in 0..8 {
                        let o = y * stride + 2 * x;
                        img[o..o + 2].copy_from_slice(&d[y * 8 + x].to_ne_bytes());
                    }
                }
                let a = match cbd {
                    CdefBitDepth::B10 => scalar::find_dir::<10>(&img, stride, cbd),
                    _ => scalar::find_dir::<12>(&img, stride, cbd),
                };
                let b = avx2_dir(bd)(&img, stride, cbd);
                assert_eq!(
                    (a.dir, a.var),
                    (b.dir, b.var),
                    "dir16 uç girdi {bd} bit {d:?} stride={stride}"
                );
                n += 1;
            }
        }
    }
    println!("[F4b dir16 uç] {n} uç blok (10/12 bit) AVX2 == skaler");
}

/// P29 — 16 bpc `filter` uç girdileri (10 ve 12 bit): 16 kenar birleşimi ×
/// güç uçları (en büyük birincil + ikincil, yalnız birincil, yalnız ikincil,
/// en küçük) × sönüm uçları (`3, 6` + `bitdepth_min_8`) × en büyük / en küçük
/// piksel desenleri (rastgele, {0, en büyük}, yürüyüş, en büyük).
#[test]
fn f4b_filter16_uc_girdiler_skaler_ile_ayni() {
    if !avx2_var() {
        on_kosul::eksik(
            "cdef_avx2::f4b_filter16_uc_girdiler_skaler_ile_ayni",
            Neden::avx2(),
            "[F4b filter16 uç] AVX2 yolu; skaler yol koşar",
        );
        return;
    }
    let mut rng = Rng::new(0xF4CD_EF16_0000_0001);
    let mut n = 0usize;
    for (bd, s) in [(10u8, 2u8), (12, 4)] {
        for size_code in 0u8..3 {
            for e in 0u8..16 {
                for (pri, sec) in [(15u8, 4u8), (15, 0), (0, 4), (1, 1), (14, 2)] {
                    for damping in [3u8, 6] {
                        for desen in 0u8..4 {
                            let mut case = random_case(&mut rng, 1, bd, size_code, desen);
                            case.edges = e;
                            case.pri = pri << s;
                            case.sec = sec << s;
                            case.damping = damping + s;
                            case.dir = (rng.u64() % 8) as u8;
                            let a = run_filter(&case, scalar_filter(&case));
                            let b = run_filter(&case, avx2_filter(bd));
                            assert_eq!(
                                a, b,
                                "filter16 {bd} bit boyut {size_code} kenar {e:04b} pri {pri} sec {sec} damp {damping} desen {desen}"
                            );
                            n += 1;
                        }
                    }
                }
            }
        }
    }
    println!("[F4b filter16 uç] {n} blok (10/12 bit) AVX2 == skaler");
}
