//! P52 (`F5b` — `F5-MC`) `aarch64` NEON (`96` yuva) + 8-bpc DOTPROD (`18` aday) motion
//! compensation bit-birebir doğruluk testleri.
//!
//! - **Bit-exact:** her yuva, skaler Rust referansıyla (`DspDispatch::select(NONE)`;
//!   skaler gövde dav1d C oracle'ıyla ayrıca bit-exact doğrulanmıştır:
//!   `mc_scalar_oracle`, `mc_family_oracle`) aynı rastgele girdide karşılaştırılır;
//!   DOTPROD yuvaları ayrıca NEON yuvalarıyla karşılaştırılır.
//! - **Aşırı okuma:** kaynak dilimleri sözleşmenin **en küçük** uzunluğuyla verilir
//!   (`(h+6)·ss + (w+7)·bpp + MC_KUYRUK`); bunun ötesine her okuma dilim sınırında
//!   paniğe yol açar (`OverReadProbe` eşdeğeri, kopyasız).
//! - **Sentinel:** hedef tamponların önünde/arkasında koruyucu baytlar
//!   (`SentinelBuffer`) ve satır adımı dolgusu yazım taşmasını yakalar.
//! - **Skaler devir:** sözleşme içi çağrılarda `mc::skaler_devir_sayisi()` artmaz.

#![forbid(unsafe_code)]

#[cfg(target_arch = "aarch64")]
#[path = "common/helpers.rs"]
mod common;

use av1oxide::cpu::CpuFlags;
use av1oxide::dsp::ARM_ISA_CANDIDATES;

#[test]
fn p52_arm64_mc_aday_sayilari() {
    let neon = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "mc" && c.implementation == "neon")
        .count();
    let resize = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "resize" && c.implementation == "neon")
        .count();
    let dot: Vec<_> = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "mc" && c.implementation == "neon_dotprod")
        .collect();
    assert_eq!(
        neon, 156,
        "mc neon adayı: 3 bit derinliği × 52 alan (32 temel + 20 ölçekli)"
    );
    assert_eq!(resize, 3, "resize neon adayı: 8 / 10 / 12 bpc");
    assert_eq!(
        ARM_ISA_CANDIDATES.len(),
        785,
        "toplam ARM64 aday sayısı 785 (767 NEON + 18 DOTPROD) olmalı"
    );
    assert_eq!(
        dot.len(),
        18,
        "mc neon_dotprod adayı: 8 bpc mc[0..8] + mct[0..8]"
    );
    for c in &dot {
        assert_eq!(c.bit_depth, 8);
        assert_eq!(c.requires, CpuFlags::NEON | CpuFlags::DOTPROD);
        assert!(!c.field.contains("BILINEAR"));
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::common::sentinel::SentinelBuffer;
    use av1oxide::cpu::{CpuFlags, CpuLevel};
    use av1oxide::dsp::DspDispatch;
    use av1oxide::dsp::mc::{MC_KUYRUK, McDsp, skaler_devir_sayisi};
    use av1oxide::frame::BitDepth;

    const SENTINEL: u8 = 0xA5;
    const KORUMA: usize = 64;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn aralik(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn bpp(bd: BitDepth) -> usize {
        if bd == BitDepth::Eight { 1 } else { 2 }
    }

    fn maxv(bd: BitDepth) -> u16 {
        ((1u32 << bd.bits()) - 1) as u16
    }

    /// Rastgele piksel düzlemi (`n` piksel, `bpp` bayt).
    fn pikseller(r: &mut Rng, bd: BitDepth, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * bpp(bd));
        for _ in 0..n {
            // Uç değer yoğunluğu: %25 0/max, kalanı düzgün.
            let p = match r.aralik(8) {
                0 => 0,
                1 => maxv(bd),
                _ => r.aralik(u64::from(maxv(bd)) + 1) as u16,
            };
            if bd == BitDepth::Eight {
                v.push(p as u8);
            } else {
                v.extend_from_slice(&p.to_ne_bytes());
            }
        }
        v
    }

    const BOYUTLAR: [(usize, &[usize]); 7] = [
        (2, &[2, 4, 8]),
        (4, &[2, 4, 8, 16]),
        (8, &[2, 4, 8, 16, 32]),
        (16, &[4, 8, 16, 32, 64]),
        (32, &[8, 16, 32, 64]),
        (64, &[16, 32, 64, 128]),
        (128, &[64, 128]),
    ];

    const BDLER: [BitDepth; 3] = [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve];

    fn tablolar() -> (DspDispatch, DspDispatch, DspDispatch) {
        let tespit = CpuFlags::detect();
        let neon = DspDispatch::select(tespit.intersection(CpuLevel::Neon.allowed()));
        let tam = DspDispatch::select(tespit);
        let skaler = DspDispatch::select(CpuFlags::NONE);
        assert!(
            neon.neon_token().is_some(),
            "Apple M4 / aarch64: NEON tespit edilmeli"
        );
        (neon, tam, skaler)
    }

    #[test]
    fn p52_mc_yuva_kurulumu_neon_ve_dotprod() {
        let (neon, tam, skaler) = tablolar();
        for bd in BDLER {
            for alan in [
                "mc[FILTER_2D_8TAP_REGULAR]",
                "mct[FILTER_2D_BILINEAR]",
                "avg",
                "w_avg",
                "mask",
                "w_mask[0]",
                "w_mask[1]",
                "w_mask[2]",
                "blend",
                "blend_v",
                "blend_h",
                "warp8x8",
                "warp8x8t",
                "emu_edge",
            ] {
                assert_eq!(
                    neon.slot_implementation(bd, "mc", alan),
                    "neon",
                    "{alan}@{bd:?}"
                );
                assert_eq!(skaler.slot_implementation(bd, "mc", alan), "skaler");
            }
            assert_eq!(
                neon.slot_implementation(bd, "mc", "mc_scaled[FILTER_2D_8TAP_REGULAR]"),
                "neon"
            );
            assert_eq!(
                neon.slot_implementation(bd, "mc", "mct_scaled[FILTER_2D_BILINEAR]"),
                "neon"
            );
            assert_eq!(neon.slot_implementation(bd, "resize", "resize"), "neon");
        }
        assert_eq!(
            neon.assigned_count(),
            767,
            "aarch64 NEON altında 767 / 767 DspDispatch yuvası atanmış olmalı"
        );
        if tam.dotprod_token().is_some() {
            assert_eq!(
                tam.slot_implementation(BitDepth::Eight, "mc", "mc[FILTER_2D_8TAP_SHARP]"),
                "neon_dotprod"
            );
            assert_eq!(
                tam.slot_implementation(BitDepth::Eight, "mc", "mct[FILTER_2D_8TAP_SMOOTH]"),
                "neon_dotprod"
            );
            assert_eq!(
                tam.slot_implementation(BitDepth::Eight, "mc", "mc[FILTER_2D_BILINEAR]"),
                "neon"
            );
            assert_eq!(
                tam.slot_implementation(BitDepth::Ten, "mc", "mc[FILTER_2D_8TAP_SHARP]"),
                "neon"
            );
        }
        let mc_neon = neon
            .slot_states()
            .iter()
            .filter(|s| s.key.family == "mc" && s.is_assigned())
            .count();
        assert_eq!(mc_neon, 156, "mc: 3 × 52 NEON yuvası atanmalı");
        let mc_dot = tam
            .slot_states()
            .iter()
            .filter(|s| s.key.family == "mc" && s.implementation == "neon_dotprod")
            .count();
        if tam.dotprod_token().is_some() {
            assert_eq!(mc_dot, 18, "mc: 18 DOTPROD yuvası NEON'un üzerine yazmalı");
        }
    }

    /// `put` (`mc[0..9]`) ve `prep` (`mct[0..9]`): NEON ↔ skaler ↔ DOTPROD.
    #[test]
    fn p52_put_prep_8_10_12bpc_bit_exact_sentinel_overread() {
        let (neon, tam, skaler) = tablolar();
        let devir0 = skaler_devir_sayisi();
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let mut sayac = 0usize;
        let mxler: [u8; 5] = [0, 1, 7, 8, 15];
        for bd in BDLER {
            let b = bpp(bd);
            let (tn, tt, ts) = (
                *neon.for_bit_depth(bd).mc(),
                *tam.for_bit_depth(bd).mc(),
                *skaler.for_bit_depth(bd).mc(),
            );
            for &(w, hs) in &BOYUTLAR {
                for &h in hs {
                    for fi in 0..10 {
                        let mx = mxler[r.aralik(5) as usize];
                        let my = mxler[r.aralik(5) as usize];
                        let ss = (w + 7) * b + b * r.aralik(5) as usize;
                        let src_len = (h + 6) * ss + (w + 7) * b + MC_KUYRUK;
                        let src = pikseller(&mut r, bd, src_len.div_ceil(b));
                        let src = &src[..src_len];
                        // put
                        let ds = w * b + b * r.aralik(3) as usize;
                        let dlen = (h - 1) * ds + w * b;
                        let mut d_ref = vec![0u8; dlen];
                        (ts.mc[fi])(&mut d_ref, ds, src, ss, w, h, mx, my, bd, maxv(bd));
                        for tablo in [&tn, &tt] {
                            let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                            (tablo.mc[fi])(sb.data_mut(), ds, src, ss, w, h, mx, my, bd, maxv(bd));
                            sb.verify().expect("put sentinel");
                            for y in 0..h {
                                let o = y * ds;
                                assert_eq!(
                                    &sb.data()[o..o + w * b],
                                    &d_ref[o..o + w * b],
                                    "put bd={bd:?} w={w} h={h} fi={fi} mx={mx} my={my} satır {y}"
                                );
                            }
                        }
                        // prep
                        let mut t_ref = vec![0i16; w * h];
                        (ts.mct[fi])(&mut t_ref, src, ss, w, h, mx, my, bd, maxv(bd));
                        for tablo in [&tn, &tt] {
                            let mut t = vec![0x5A5Ai16; w * h + 16];
                            (tablo.mct[fi])(&mut t[..w * h], src, ss, w, h, mx, my, bd, maxv(bd));
                            assert_eq!(
                                &t[..w * h],
                                &t_ref[..],
                                "prep bd={bd:?} w={w} h={h} fi={fi} mx={mx} my={my}"
                            );
                            assert!(t[w * h..].iter().all(|&v| v == 0x5A5A), "prep taşma");
                        }
                        sayac += 1;
                    }
                }
            }
        }
        assert_eq!(
            skaler_devir_sayisi(),
            devir0,
            "sözleşme içi çağrıda skaler devir olmamalı"
        );
        assert!(sayac >= 3 * 27 * 10);
    }

    /// Bütün `mx × my` birleşimleri (dar ve geniş bloklarda tüm 1/16 konumlar).
    #[test]
    fn p52_put_prep_tum_alt_piksel_konumlari() {
        let (neon, tam, skaler) = tablolar();
        let mut r = Rng(0x1234_5678_9ABC_DEF1);
        for bd in BDLER {
            let b = bpp(bd);
            let (tn, tt, ts) = (
                *neon.for_bit_depth(bd).mc(),
                *tam.for_bit_depth(bd).mc(),
                *skaler.for_bit_depth(bd).mc(),
            );
            for (w, h) in [(2usize, 2usize), (4, 4), (8, 8), (16, 4), (32, 8)] {
                let ss = (w + 7) * b;
                let src_len = (h + 6) * ss + (w + 7) * b + MC_KUYRUK;
                let src = pikseller(&mut r, bd, src_len.div_ceil(b));
                let src = &src[..src_len];
                for fi in 0..10 {
                    for mx in 0..16u8 {
                        for my in 0..16u8 {
                            let ds = w * b;
                            let mut d_ref = vec![0u8; h * ds];
                            (ts.mc[fi])(&mut d_ref, ds, src, ss, w, h, mx, my, bd, maxv(bd));
                            let mut t_ref = vec![0i16; w * h];
                            (ts.mct[fi])(&mut t_ref, src, ss, w, h, mx, my, bd, maxv(bd));
                            for tablo in [&tn, &tt] {
                                let mut d = vec![0u8; h * ds];
                                (tablo.mc[fi])(&mut d, ds, src, ss, w, h, mx, my, bd, maxv(bd));
                                assert_eq!(
                                    d, d_ref,
                                    "put bd={bd:?} {w}x{h} fi={fi} mx={mx} my={my}"
                                );
                                let mut t = vec![0i16; w * h];
                                (tablo.mct[fi])(&mut t, src, ss, w, h, mx, my, bd, maxv(bd));
                                assert_eq!(
                                    t, t_ref,
                                    "prep bd={bd:?} {w}x{h} fi={fi} mx={mx} my={my}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn ara_degerler(r: &mut Rng, n: usize, bd: BitDepth) -> Vec<i16> {
        // `prep` çıktısı aralığı (8 bpc: ~[-2^13, 2^14); 16 bpc: PREP_BIAS'lı).
        let (lo, hi) = if bd == BitDepth::Eight {
            (-8192i64, 16384i64)
        } else {
            (-16384i64, 16384i64)
        };
        (0..n)
            .map(|_| (lo + r.aralik((hi - lo) as u64) as i64) as i16)
            .collect()
    }

    /// `avg`, `w_avg`, `mask`, `w_mask[0..2]`.
    #[test]
    fn p52_bilesik_8_10_12bpc_bit_exact_sentinel() {
        let (neon, _, skaler) = tablolar();
        let devir0 = skaler_devir_sayisi();
        let mut r = Rng(0xDEAD_BEEF_CAFE_F00D);
        for bd in BDLER {
            let b = bpp(bd);
            let (tn, ts): (McDsp, McDsp) =
                (*neon.for_bit_depth(bd).mc(), *skaler.for_bit_depth(bd).mc());
            for w in [4usize, 8, 16, 32, 64, 128] {
                for h in [4usize, 8, 16, 32, 64, 128] {
                    let n = w * h;
                    let t1 = ara_degerler(&mut r, n, bd);
                    let t2 = ara_degerler(&mut r, n, bd);
                    let mask: Vec<u8> = (0..n).map(|_| r.aralik(65) as u8).collect();
                    let ds = w * b + b * r.aralik(4) as usize;
                    let dlen = (h - 1) * ds + w * b;
                    let kontrol = |sb: &SentinelBuffer, ref_: &[u8], ad: &str| {
                        sb.verify()
                            .unwrap_or_else(|e| panic!("{ad} sentinel {e:?}"));
                        for y in 0..h {
                            let o = y * ds;
                            assert_eq!(
                                &sb.data()[o..o + w * b],
                                &ref_[o..o + w * b],
                                "{ad} bd={bd:?} {w}x{h} satır {y}"
                            );
                        }
                    };
                    let mut d_ref = vec![0u8; dlen];
                    (ts.avg)(&mut d_ref, ds, &t1, &t2, w, h, bd, maxv(bd));
                    let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                    (tn.avg)(sb.data_mut(), ds, &t1, &t2, w, h, bd, maxv(bd));
                    kontrol(&sb, &d_ref, "avg");

                    let weight = r.aralik(17) as i32;
                    (ts.w_avg)(&mut d_ref, ds, &t1, &t2, w, h, weight, bd, maxv(bd));
                    let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                    (tn.w_avg)(sb.data_mut(), ds, &t1, &t2, w, h, weight, bd, maxv(bd));
                    kontrol(&sb, &d_ref, "w_avg");

                    (ts.mask)(&mut d_ref, ds, &t1, &t2, w, h, &mask, bd, maxv(bd));
                    let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                    (tn.mask)(sb.data_mut(), ds, &t1, &t2, w, h, &mask, bd, maxv(bd));
                    kontrol(&sb, &d_ref, "mask");

                    if w >= 8 && h >= 8 {
                        for sub in 0..3usize {
                            let (sh, sv) = (sub != 0, sub == 2);
                            let mlen = (w >> usize::from(sh)) * (h >> usize::from(sv));
                            for sign in 0..2 {
                                let mut m_ref = vec![0u8; mlen];
                                (ts.w_mask[sub])(
                                    &mut d_ref,
                                    ds,
                                    &t1,
                                    &t2,
                                    w,
                                    h,
                                    &mut m_ref,
                                    sign,
                                    sh,
                                    sv,
                                    bd,
                                    maxv(bd),
                                );
                                let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                                let mut m = vec![0u8; mlen + 8];
                                (tn.w_mask[sub])(
                                    sb.data_mut(),
                                    ds,
                                    &t1,
                                    &t2,
                                    w,
                                    h,
                                    &mut m[..mlen],
                                    sign,
                                    sh,
                                    sv,
                                    bd,
                                    maxv(bd),
                                );
                                kontrol(&sb, &d_ref, "w_mask");
                                assert_eq!(
                                    &m[..mlen],
                                    &m_ref[..],
                                    "w_mask[{sub}] maske bd={bd:?} {w}x{h} sign={sign}"
                                );
                                assert!(m[mlen..].iter().all(|&v| v == 0));
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(skaler_devir_sayisi(), devir0);
    }

    /// `blend`, `blend_v`, `blend_h`.
    #[test]
    fn p52_karisim_8_10_12bpc_bit_exact_sentinel() {
        let (neon, _, skaler) = tablolar();
        let devir0 = skaler_devir_sayisi();
        let mut r = Rng(0x0BAD_C0DE_1234_4321);
        for bd in BDLER {
            let b = bpp(bd);
            let (tn, ts): (McDsp, McDsp) =
                (*neon.for_bit_depth(bd).mc(), *skaler.for_bit_depth(bd).mc());
            for w in [2usize, 4, 8, 16, 32, 64, 128] {
                for h in [2usize, 4, 8, 16, 32, 64, 128] {
                    let n = w * h;
                    let tmp = pikseller(&mut r, bd, n);
                    let mask: Vec<u8> = (0..n).map(|_| r.aralik(65) as u8).collect();
                    let ds = w * b + b * r.aralik(4) as usize;
                    let dlen = (h - 1) * ds + w * b;
                    let taban = pikseller(&mut r, bd, dlen.div_ceil(b));
                    let mut isler: Vec<(&str, usize)> = Vec::new();
                    if (4..=32).contains(&w) && (4..=32).contains(&h) {
                        isler.push(("blend", h));
                    }
                    if w <= 32 {
                        isler.push(("blend_v", h));
                    }
                    if h <= 32 {
                        isler.push(("blend_h", h));
                    }
                    for (ad, _) in isler {
                        let mut d_ref = taban[..dlen].to_vec();
                        let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                        sb.data_mut().copy_from_slice(&taban[..dlen]);
                        match ad {
                            "blend" => {
                                (ts.blend)(&mut d_ref, ds, &tmp, w, h, &mask, bd, maxv(bd));
                                (tn.blend)(sb.data_mut(), ds, &tmp, w, h, &mask, bd, maxv(bd));
                            }
                            "blend_v" => {
                                (ts.blend_v)(&mut d_ref, ds, &tmp, w, h, bd, maxv(bd));
                                (tn.blend_v)(sb.data_mut(), ds, &tmp, w, h, bd, maxv(bd));
                            }
                            _ => {
                                (ts.blend_h)(&mut d_ref, ds, &tmp, w, h, bd, maxv(bd));
                                (tn.blend_h)(sb.data_mut(), ds, &tmp, w, h, bd, maxv(bd));
                            }
                        }
                        sb.verify()
                            .unwrap_or_else(|e| panic!("{ad} sentinel {e:?}"));
                        assert_eq!(sb.data(), &d_ref[..], "{ad} bd={bd:?} {w}x{h}");
                    }
                }
            }
        }
        assert_eq!(skaler_devir_sayisi(), devir0);
    }

    /// `warp8x8`, `warp8x8t` (rastgele afin parametreler, sözleşme içi).
    #[test]
    fn p52_warp_8_10_12bpc_bit_exact_sentinel() {
        let (neon, _, skaler) = tablolar();
        let devir0 = skaler_devir_sayisi();
        let mut r = Rng(0x7777_1111_3333_5555);
        let mut gecerli = 0usize;
        for bd in BDLER {
            let b = bpp(bd);
            let (tn, ts): (McDsp, McDsp) =
                (*neon.for_bit_depth(bd).mc(), *skaler.for_bit_depth(bd).mc());
            for _ in 0..600 {
                let abcd: [i16; 4] =
                    core::array::from_fn(|_| (r.aralik(2049) as i32 - 1024) as i16);
                let mx = r.aralik(8192) as i32 - 4096;
                let my = r.aralik(8192) as i32 - 4096;
                if !av1oxide::dsp::mc::warp_indeksleri_icinde(&abcd, mx, my) {
                    continue;
                }
                gecerli += 1;
                let ss = 15 * b + b * r.aralik(4) as usize;
                let src_len = 14 * ss + 15 * b + MC_KUYRUK;
                let src = pikseller(&mut r, bd, src_len.div_ceil(b));
                let src = &src[..src_len];
                let ds = 8 * b + b * r.aralik(3) as usize;
                let dlen = 7 * ds + 8 * b;
                let mut d_ref = vec![0u8; dlen];
                (ts.warp8x8)(&mut d_ref, ds, src, ss, &abcd, mx, my, bd, maxv(bd));
                let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                (tn.warp8x8)(sb.data_mut(), ds, src, ss, &abcd, mx, my, bd, maxv(bd));
                sb.verify().expect("warp8x8 sentinel");
                for y in 0..8 {
                    assert_eq!(
                        &sb.data()[y * ds..y * ds + 8 * b],
                        &d_ref[y * ds..y * ds + 8 * b],
                        "warp8x8 bd={bd:?} abcd={abcd:?} mx={mx} my={my}"
                    );
                }
                let ts_ = 8 + r.aralik(3) as usize;
                let tlen = 7 * ts_ + 8;
                let mut t_ref = vec![0i16; tlen];
                (ts.warp8x8t)(&mut t_ref, ts_, src, ss, &abcd, mx, my, bd, maxv(bd));
                let mut t = vec![0x1111i16; tlen + 8];
                (tn.warp8x8t)(&mut t[..tlen], ts_, src, ss, &abcd, mx, my, bd, maxv(bd));
                for y in 0..8 {
                    assert_eq!(
                        &t[y * ts_..y * ts_ + 8],
                        &t_ref[y * ts_..y * ts_ + 8],
                        "warp8x8t bd={bd:?} abcd={abcd:?}"
                    );
                }
                assert!(t[tlen..].iter().all(|&v| v == 0x1111));
            }
        }
        assert!(gecerli > 300, "geçerli warp örneği: {gecerli}");
        assert_eq!(skaler_devir_sayisi(), devir0);
    }

    /// `emu_edge` (rastgele konum/boyut, negatif ve taşan koordinatlar).
    #[test]
    fn p52_emu_edge_8_10_12bpc_bit_exact_sentinel() {
        let (neon, _, skaler) = tablolar();
        let devir0 = skaler_devir_sayisi();
        let mut r = Rng(0x2468_ACE0_1357_9BDF);
        for bd in BDLER {
            let b = bpp(bd);
            let (tn, ts): (McDsp, McDsp) =
                (*neon.for_bit_depth(bd).mc(), *skaler.for_bit_depth(bd).mc());
            for _ in 0..400 {
                let bw = 1 + r.aralik(135) as usize;
                let bh = 1 + r.aralik(135) as usize;
                let iw = 1 + r.aralik(200) as usize;
                let ih = 1 + r.aralik(200) as usize;
                let x = r.aralik(iw as u64 + 2 * bw as u64) as i32 - bw as i32;
                let y = r.aralik(ih as u64 + 2 * bh as u64) as i32 - bh as i32;
                let ss = iw * b + b * r.aralik(4) as usize;
                let src = pikseller(&mut r, bd, (ih * ss).div_ceil(b));
                let src = &src[..(ih - 1) * ss + iw * b];
                let ds = bw * b + b * r.aralik(4) as usize;
                let dlen = (bh - 1) * ds + bw * b;
                let gecerli = if b == 1 {
                    av1oxide::dsp::mc::kenar_sozlesmesi(
                        bw,
                        bh,
                        iw,
                        ih,
                        x,
                        y,
                        dlen,
                        ds,
                        src.len(),
                        ss,
                    )
                } else {
                    av1oxide::dsp::mc::kenar16_sozlesmesi(
                        bw,
                        bh,
                        iw,
                        ih,
                        x,
                        y,
                        dlen,
                        ds,
                        src.len(),
                        ss,
                        bd,
                        maxv(bd),
                    )
                };
                if !gecerli {
                    continue;
                }
                let mut d_ref = vec![0u8; dlen];
                (ts.emu_edge)(bw, bh, iw, ih, x, y, &mut d_ref, ds, src, ss, bd, maxv(bd));
                let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                (tn.emu_edge)(
                    bw,
                    bh,
                    iw,
                    ih,
                    x,
                    y,
                    sb.data_mut(),
                    ds,
                    src,
                    ss,
                    bd,
                    maxv(bd),
                );
                sb.verify().expect("emu_edge sentinel");
                for row in 0..bh {
                    assert_eq!(
                        &sb.data()[row * ds..row * ds + bw * b],
                        &d_ref[row * ds..row * ds + bw * b],
                        "emu_edge bd={bd:?} {bw}x{bh} iw={iw} ih={ih} x={x} y={y}"
                    );
                }
            }
        }
        assert_eq!(skaler_devir_sayisi(), devir0);
    }

    /// `mc_scaled[0..=9]` ve `mct_scaled[0..=9]` (`60` yuva: 8/10/12 bpc, 8-tap + bilinear).
    #[test]
    fn arm64_mc_scaled_ve_mct_scaled_8_10_12bpc_bit_exact_sentinel() {
        let (neon, _, skaler) = tablolar();
        let devir0 = skaler_devir_sayisi();
        let mut r = Rng(0x5CA1_ED4A_0003_2026);
        for bd in BDLER {
            let b = bpp(bd);
            let bm = maxv(bd);
            let (tn, ts): (McDsp, McDsp) =
                (*neon.for_bit_depth(bd).mc(), *skaler.for_bit_depth(bd).mc());
            for fi in 0..10usize {
                for &(w, hs) in &BOYUTLAR {
                    let h = hs[(r.next() as usize) % hs.len()];
                    let mx = r.aralik(1024) as i32;
                    let my = r.aralik(1024) as i32;
                    let dx =
                        [1024i32, 2048, 512, 1 + r.aralik(2048) as i32][(r.next() as usize) & 3];
                    let dy =
                        [1024i32, 2048, 512, 1 + r.aralik(2048) as i32][(r.next() as usize) & 3];
                    let max_sx = ((mx + (w as i32 - 1) * dx) >> 10) as usize + 8;
                    let max_sy = ((my + (h as i32 - 1) * dy) >> 10) as usize + 8;
                    let ss = max_sx * b + b * (r.aralik(4) as usize);
                    let src_len = (max_sy - 1) * ss + max_sx * b;
                    let src = pikseller(&mut r, bd, src_len.div_ceil(b));
                    let src = &src[..src_len];

                    let ds = w * b + b * (r.aralik(4) as usize);
                    let dlen = (h - 1) * ds + w * b;
                    let mut d_ref = vec![0u8; dlen];
                    (ts.mc_scaled[fi])(&mut d_ref, ds, src, ss, w, h, mx, my, dx, dy, bd, bm);
                    let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                    (tn.mc_scaled[fi])(sb.data_mut(), ds, src, ss, w, h, mx, my, dx, dy, bd, bm);
                    sb.verify().expect("mc_scaled sentinel");
                    for row in 0..h {
                        assert_eq!(
                            &sb.data()[row * ds..row * ds + w * b],
                            &d_ref[row * ds..row * ds + w * b],
                            "mc_scaled[{fi}] bd={bd:?} {w}x{h} mx={mx} my={my} dx={dx} dy={dy}"
                        );
                    }

                    let tlen = w * h;
                    let mut t_ref = vec![0i16; tlen];
                    (ts.mct_scaled[fi])(&mut t_ref, src, ss, w, h, mx, my, dx, dy, bd, bm);
                    let mut t_got = vec![0x3333i16; tlen + 16];
                    (tn.mct_scaled[fi])(&mut t_got[..tlen], src, ss, w, h, mx, my, dx, dy, bd, bm);
                    assert_eq!(
                        &t_got[..tlen],
                        &t_ref[..],
                        "mct_scaled[{fi}] bd={bd:?} {w}x{h} mx={mx} my={my} dx={dx} dy={dy}"
                    );
                    assert!(t_got[tlen..].iter().all(|&v| v == 0x3333));
                }
            }
        }
        assert_eq!(skaler_devir_sayisi(), devir0);
    }

    /// `resize/resize` (`3` yuva: 8/10/12 bpc, SuperRes yatay büyütme).
    #[test]
    fn arm64_resize_8_10_12bpc_bit_exact_sentinel() {
        use av1oxide::dsp::resize::ResizeBitDepth;
        let (neon, _, skaler) = tablolar();
        let devir0 = skaler_devir_sayisi();
        let mut r = Rng(0x52E5_12E0_4A03_2026);
        for (bd, rbd) in [
            (BitDepth::Eight, ResizeBitDepth::B8),
            (BitDepth::Ten, ResizeBitDepth::B10),
            (BitDepth::Twelve, ResizeBitDepth::B12),
        ] {
            let b = bpp(bd);
            let f_neon = neon.for_bit_depth(bd).resize();
            let f_sk = skaler.for_bit_depth(bd).resize();
            for &(src_w, dst_w, h) in &[
                (1usize, 1usize, 1usize),
                (3, 5, 2),
                (7, 13, 4),
                (16, 32, 8),
                (33, 65, 5),
                (64, 128, 16),
                (127, 255, 7),
            ] {
                let dx = 4096 + r.aralik(16_384) as i32;
                let mx0 = r.aralik(0x4000) as i32;
                let ss = src_w * b + b * (r.aralik(4) as usize);
                let slen = (h - 1) * ss + src_w * b;
                let src = pikseller(&mut r, bd, slen.div_ceil(b));
                let src = &src[..slen];
                let ds = dst_w * b + b * (r.aralik(4) as usize);
                let dlen = (h - 1) * ds + dst_w * b;
                let mut d_ref = vec![0u8; dlen];
                f_sk(&mut d_ref, ds, src, ss, dst_w, h, src_w, dx, mx0, rbd);
                let mut sb = SentinelBuffer::new(dlen, KORUMA, SENTINEL);
                f_neon(sb.data_mut(), ds, src, ss, dst_w, h, src_w, dx, mx0, rbd);
                sb.verify().expect("resize sentinel");
                for row in 0..h {
                    assert_eq!(
                        &sb.data()[row * ds..row * ds + dst_w * b],
                        &d_ref[row * ds..row * ds + dst_w * b],
                        "resize bd={bd:?} {src_w}->{dst_w}x{h} dx={dx} mx0={mx0}"
                    );
                }
            }
        }
        assert_eq!(skaler_devir_sayisi(), devir0);
    }
}
