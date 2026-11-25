//! S15.5 — 8(b) AVX2 ↔ skaler **bit-exact** ve 8(c) sınır dışı okuma/yazma yok.
//!
//! ## 8(b) kapsamı
//!
//! Her (bit derinliği, genişlik) hücresi için **≥ 2 000** sabit tohumlu
//! rastgele blok üretilir; yükseklik o genişlik için dav1d'nin desteklediği
//! yüksekliklerden (en küçük/en büyük dahil) seçilir. Aynı girdi hem skaler hem
//! AVX2 yolundan geçirilip **bit-exact** karşılaştırılır. AVX2 yolu 8-bit
//! içindir; 10/12-bit hücreler güvenli sarmalayıcının **skaler devir** yolunu
//! sınar (S15.5 kapsamı) ve bu raporda açıkça belirtilir.
//!
//! ## 8(c) yöntemi
//!
//! - [`SentinelBuffer`]: `dst` bölgesinin önüne/arkasına koruyucu koyar →
//!   **yazma taşması** yakalanır (negatif kontrolüyle birlikte).
//! - Sözleşme dışı okuma: kaynak dilimi tam sözleşme uzunluğunda **artı 64
//!   bayt dolgu** olarak verilir; iki farklı dolgu deseniyle koşulur. Çekirdek
//!   sözleşme bölgesinin dışından okursa çıktı değişir. Negatif kontrol, bu
//!   farkın gerçekten yakalandığını gösterir.
//!
//! Her iki ölçüm **her iki yol** (skaler + AVX2) için yapılır ve AVX2 yoksa
//! test açıkça atlanır (gerekçe yazılır).

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::{McBitDepth, McFn};
use av1oxide::dsp::scalar::mc as scalar;
use common::mc_fixture::{Case, HEIGHTS, Rng, WIDTHS, bit_depth, dst_len, random_case, run_case};
use common::on_kosul::{self, Neden};
use common::sentinel::{SentinelBuffer, VARSAYILAN_KORUMA};

/// Hücre başına rastgele blok sayısı (S15.5/8: en az 2 000).
const HUCRE_BLOK: usize = 2_000;

/// AVX2 yolu bu platformda kullanılabilir mi?
fn avx2_var() -> bool {
    // F3b: AVX2 yolu yalnız kurulumda üretilen belirteçle açılır.
    av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect())
        .avx2_token()
        .is_some()
}

/// AVX2 MC çekirdeği: belirteçli kurulumun yuvası (AVX2 yoksa skaler —
/// çağrılmaz).
fn avx2_kernel() -> McFn {
    // F3c: pilotun `McFn` girişi belirteçle alınır (yuva uyarlayıcısı
    // `put_8tap_regular_slot_fn` aynı girişe gider).
    #[cfg(target_arch = "x86_64")]
    if let Some(token) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        return av1oxide::dsp::x86::mc::put_8tap_regular_fn(token);
    }
    scalar::put_8tap_regular
}

/// 8(b) — her (bit derinliği, genişlik) hücresi için ≥ 2 000 blok.
#[test]
fn sekiz_b_avx2_skaler_ile_bit_ayni() {
    if !avx2_var() {
        on_kosul::eksik(
            "mc_avx2::sekiz_b_avx2_skaler_ile_bit_ayni",
            Neden::avx2(),
            "[8(b)] AVX2 yolu; skaler yol koşar",
        );
        return;
    }
    let mut rng = Rng::new(0x5A15_05C0_7B00_0001);
    let mut toplam = 0usize;
    for bd in McBitDepth::ALL {
        for (width_index, &w) in WIDTHS.iter().enumerate() {
            let heights = HEIGHTS[width_index];
            let mut fark = 0usize;
            for i in 0..HUCRE_BLOK {
                let h = heights[rng.below(heights.len())];
                let mx = rng.below(16) as u8;
                let my = rng.below(16) as u8;
                let case = random_case(&mut rng, bd, w, h, mx, my);
                let skaler = run_case(scalar::put_8tap_regular, &case);
                let isa = run_case(avx2_kernel(), &case);
                if skaler != isa {
                    fark += 1;
                    if fark == 1 {
                        let at = skaler
                            .iter()
                            .zip(isa.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(0);
                        println!(
                            "  ilk fark ({:?}, w={w}, h={h}, blok {i}): bayt {at} \
                             skaler={:02x?} isa={:02x?} (mx={mx}, my={my})",
                            bd,
                            &skaler[at.saturating_sub(2)..(at + 6).min(skaler.len())],
                            &isa[at.saturating_sub(2)..(at + 6).min(isa.len())],
                        );
                    }
                }
                toplam += 1;
            }
            assert_eq!(fark, 0, "{bd:?} w={w}: {fark}/{HUCRE_BLOK} blok farklı");
            println!("  {bd:?} w={w:3} : {HUCRE_BLOK} blok AVX2 == skaler (OK)");
        }
    }
    assert_eq!(toplam, 3 * WIDTHS.len() * HUCRE_BLOK);
    println!("[8(b)] toplam {toplam} blok karşılaştırıldı");
}

/// `src` sözleşme uzunluğu: `(h+6)·src_stride + (w+7)·bpp`.
fn contract_src_len(case: &Case) -> usize {
    let bpp = bit_depth(case.bit_depth).bytes_per_pixel();
    (case.h + 6) * case.src_stride + (case.w + 7) * bpp
}

/// Kaynağı `pad` ile dolgulanmış **büyük** dilim olarak verir (over-read probu).
fn run_with_trailing_padding(kernel: McFn, case: &Case, pad: u8, extra: usize) -> Vec<u8> {
    let need = contract_src_len(case);
    assert!(case.src.len() >= need);
    let mut input = vec![pad; need + extra];
    input[..need].copy_from_slice(&case.src[..need]);
    let mut dst = vec![0u8; dst_len(case)];
    kernel(
        &mut dst,
        case.dst_stride,
        &input,
        case.src_stride,
        case.w,
        case.h,
        case.mx,
        case.my,
        bit_depth(case.bit_depth),
    );
    let bpp = bit_depth(case.bit_depth).bytes_per_pixel();
    let mut out = Vec::with_capacity(case.w * case.h * bpp);
    for y in 0..case.h {
        let start = y * case.dst_stride;
        out.extend_from_slice(&dst[start..start + case.w * bpp]);
    }
    out
}

/// 8(c) — sentinel: hedefe taşma yok (her iki yol).
#[test]
fn sekiz_c_hedefe_tasma_yok() {
    let mut rng = Rng::new(0x5A15_05C0_7B00_0002);
    let mut checked = 0usize;
    for bd in McBitDepth::ALL {
        for w in [2usize, 4, 8, 16, 64, 128] {
            let heights = HEIGHTS[WIDTHS.iter().position(|&x| x == w).unwrap()];
            for i in 0..8 {
                let h = heights[i % heights.len()];
                let case = random_case(&mut rng, bd, w, h, (i as u8) % 16, ((i + 5) as u8) % 16);
                for kernel in [scalar::put_8tap_regular, avx2_kernel()] {
                    let mut buffer = SentinelBuffer::with_defaults(dst_len(&case));
                    kernel(
                        buffer.data_mut(),
                        case.dst_stride,
                        &case.src,
                        case.src_stride,
                        case.w,
                        case.h,
                        case.mx,
                        case.my,
                        bd,
                    );
                    assert_eq!(
                        buffer.verify(),
                        Ok(()),
                        "hedefe taşma: bd={bd:?} w={w} h={h} mx={} my={}",
                        case.mx,
                        case.my
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0);
    println!("[8(c) sentinel] {checked} koşu: hedef koruyucuları temiz");

    // Negatif kontrol: koruyucuya yazma **yakalanmalı** (tespit çalışıyor mu?).
    let mut buffer = SentinelBuffer::with_defaults(32);
    let offset = buffer.data_offset() + buffer.data_len();
    buffer.raw_mut()[offset] = 0x00;
    assert!(
        buffer.verify().is_err(),
        "negatif kontrol: koruyucuya yazma yakalanmadı"
    );
    println!("[8(c) sentinel] negatif kontrol: arka koruyucuya yazma yakalandı");
}

/// 8(c) — sözleşme dışı okuma yok (her iki yol, iki dolgu deseni).
#[test]
fn sekiz_c_sozlesme_disi_okuma_yok() {
    let mut rng = Rng::new(0x5A15_05C0_7B00_0003);
    let mut checked = 0usize;
    for bd in McBitDepth::ALL {
        for w in [2usize, 4, 8, 16, 32, 64, 128] {
            let heights = HEIGHTS[WIDTHS.iter().position(|&x| x == w).unwrap()];
            for i in 0..6 {
                let h = heights[i % heights.len()];
                let case = random_case(&mut rng, bd, w, h, (i as u8 * 3) % 16, (i as u8 * 5) % 16);
                for kernel in [scalar::put_8tap_regular, avx2_kernel()] {
                    let a = run_with_trailing_padding(kernel, &case, 0x00, 64);
                    let b = run_with_trailing_padding(kernel, &case, 0xFF, 64);
                    assert_eq!(
                        a, b,
                        "sözleşme dışı okuma: bd={bd:?} w={w} h={h} mx={} my={}",
                        case.mx, case.my
                    );
                    checked += 1;
                }
            }
        }
    }
    println!("[8(c) over-read] {checked} koşu: dolgu deseni çıktıyı değiştirmiyor");

    // Negatif kontrol: sözleşme dışı baytı okuyan bir çekirdek **yakalanmalı**.
    #[allow(clippy::too_many_arguments)]
    fn sozlesme_disi_okuyan(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: u8,
        my: u8,
        bd: McBitDepth,
    ) {
        scalar::put_8tap_regular(dst, dst_stride, src, src_stride, w, h, mx, my, bd);
        let need = (h + 6) * src_stride + (w + 7) * bd.bytes_per_pixel();
        if src.len() > need {
            dst[0] = dst[0].wrapping_add(src[need]);
        }
    }
    let case = {
        let mut rng = Rng::new(0x5A15_05C0_7B00_0004);
        random_case(&mut rng, McBitDepth::B8, 8, 8, 3, 5)
    };
    let a = run_with_trailing_padding(sozlesme_disi_okuyan, &case, 0x00, 64);
    let b = run_with_trailing_padding(sozlesme_disi_okuyan, &case, 0xFF, 64);
    assert_ne!(
        a, b,
        "negatif kontrol: sözleşme dışı okuma yakalanmadı (probe etkisiz)"
    );
    println!("[8(c) over-read] negatif kontrol: sözleşme dışı okuma yakalandı");
}

/// S15.5/7 — zorlanmış skaler maske: `DspDispatch::mc` yuvası skaler kalır.
/// P30: yuva kimliği seçim kaydıyla denetlenir (işlev adresi optimize
/// derlemede kimlik kanıtı değildir).
#[test]
fn zorlanmis_skaler_maskesi_mc_yuvasini_skaler_birakir() {
    use av1oxide::cpu::CpuFlags;
    use av1oxide::dsp::DspDispatch;
    use av1oxide::frame::BitDepth;

    const YUVA: &str = "mc[FILTER_2D_8TAP_REGULAR]";
    let masked = CpuFlags::AVX2
        .union(CpuFlags::SSE2)
        .union(CpuFlags::AVX512ICL)
        .difference(CpuFlags::AVX2);
    // F3c: pilot 8 bit tablosunun `mc[FILTER_2D_8TAP_REGULAR]` yuvasıdır.
    let dsp = DspDispatch::select(masked);
    // P36: her belirteç kendi bayrağına bakar (P30); AVX2 kapalıyken yuva
    // AVX2 gövdesi tutmaz, AVX512ICL belirteci varsa AVX512ICL gövdesidir.
    let beklenen = if dsp.avx512icl_token().is_some() {
        "avx512icl"
    } else {
        "skaler"
    };
    assert_eq!(
        dsp.slot_implementation(BitDepth::Eight, "mc", YUVA),
        beklenen,
        "AVX2 biti silinince MC yuvası AVX2 gövdesi tutmamalı"
    );

    #[cfg(target_arch = "x86_64")]
    {
        let dsp = DspDispatch::select(CpuFlags::AVX2);
        let beklenen = if dsp.avx2_token().is_some()
            && av1oxide::dsp::ISA_CANDIDATES
                .iter()
                .any(|c| c.family == "mc" && c.implementation == "avx2" && c.default)
        {
            "avx2"
        } else {
            // F3b: AVX2 tespit edilmeyen makinede bayrak verilse de yuva açılmaz.
            "skaler"
        };
        assert_eq!(
            dsp.slot_implementation(BitDepth::Eight, "mc", YUVA),
            beklenen
        );
    }
    println!("[S15.5/7] zorlanmış skaler maske: MC yuvası skaler (OK)");
}

/// Sentinel koruma uzunluğunun yeterliliği (VARSAYILAN_KORUMA kullanımı).
const _: () = assert!(VARSAYILAN_KORUMA >= 16);
