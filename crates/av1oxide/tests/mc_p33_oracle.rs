//! P33 (F4-MC b) — 8 bpc kalan MC işlevlerinin taramalı dav1d C oracle'ı.
//!
//! `scripts/oracle/mc_p33_oracle.c` hücre başına beklenen FNV-1a 64 özetini
//! yazar (`run_mc_oracle.sh` tam kipi, adım 6g → `.cache/mc-p33-oracle/`).
//! Girdi baytları hücre tohumundan belirlenimli üretilir; bu test aynı üreteçle
//! (xorshift64*, aynı çekiliş sırası) girdiyi yeniden kurar ve **AVX2, SSSE3 ve
//! SSE4.1 (P35; gövdesi olmayan yuvalar skaler) ve skaler** yuvaların özetini
//! dav1d C ile karşılaştırır: bütün boyutlar × parametre
//! uçları × üç hedef satır adımı.
//!
//! İşlevler: 0 `avg`, 1 `w_avg` (ağırlık 0…16), 2 `mask` (4 maske deseni),
//! 3…5 `w_mask` 4:4:4/4:2:2/4:2:0 (`sign` 0/1), 6 `blend` (4 maske deseni),
//! 7 `blend_v`, 8 `blend_h`, 9 `warp8x8`, 10 `warp8x8t` (27 parametre seti:
//! `abcd` ∈ [−1024, 1024], `mx`/`my` filtre indeksini köşelerde [0, 192]'de
//! tutan aralığın uçları ve iç noktası), 11 `emu_edge` (11 `bw` × 25 konum:
//! içeride / kısmi / tamamen taşma, x ve y ayrı; görüntü 160 ya da 48),
//! 12 `mc_scaled`, 13 `mct_scaled` (10 filtre × `w` 2…128 × 6 parametre seti:
//! `dy` 1024 / 2048 / genel, `dx` 1024 / 2048 / genel / 1, `mx`/`my` uçları;
//! kayıtta `ds` alanı `filtre·8 + set`; sürüm 6: ek olarak `h` 6/12/24 — OBMC
//! üst komşusunun `3·2^k` tahmin yüksekliği — × `w` 2…64).

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::mc::{
    AvgFn, BlendDirFn, BlendFn, EmuEdgeFn, MaskFn, McDsp, McPrepScaledFn, McPutScaledFn, WAvgFn,
    WMaskFn, Warp8x8Fn, Warp8x8tFn, skaler_devir_sayisi,
};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

/// 8 bpc `prep` çıktı aralığı (dav1d `mc_tmpl.c:40`).
const PREP_ARALIK: (i32, i32) = (-5132, 9212);
/// Oracle sürümü (`mc_p33_oracle.c`).
const SURUM: u32 = 6;
/// Warp hücrelerinin başladığı hücre indeksi (`(3·36 + 3·25 + 16 + 35 + 35)·3`).
const WARP_BAS: u32 = 807;
/// `emu_edge` hücrelerinin başladığı hücre indeksi (`WARP_BAS + 2·27·3`).
const EMU_BAS: u32 = 969;

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
    /// `[lo, hi]` — C: `lo + (u64 >> 33) % (hi − lo + 1)`.
    fn aralik(&mut self, lo: i32, hi: i32) -> i32 {
        lo + ((self.u64() >> 33) % (hi - lo + 1) as u64) as i32
    }
}

struct Yuvalar {
    avg: AvgFn,
    w_avg: WAvgFn,
    mask: MaskFn,
    w_mask: [WMaskFn; 3],
    blend: BlendFn,
    blend_v: BlendDirFn,
    blend_h: BlendDirFn,
    warp8x8: Warp8x8Fn,
    warp8x8t: Warp8x8tFn,
    emu_edge: EmuEdgeFn,
    mc_scaled: [McPutScaledFn; 10],
    mct_scaled: [McPrepScaledFn; 10],
}

impl Yuvalar {
    fn skaler() -> Self {
        let s = McDsp::scalar_for(BitDepth::Eight);
        Self {
            avg: s.avg,
            w_avg: s.w_avg,
            mask: s.mask,
            w_mask: s.w_mask,
            blend: s.blend,
            blend_v: s.blend_v,
            blend_h: s.blend_h,
            warp8x8: s.warp8x8,
            warp8x8t: s.warp8x8t,
            emu_edge: s.emu_edge,
            mc_scaled: s.mc_scaled,
            mct_scaled: s.mct_scaled,
        }
    }
}

fn avx2_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        use av1oxide::dsp::x86::{
            mc_bilesik as b, mc_karisim as k, mc_kenar as kn, mc_olcekli as ol, mc_warp as wp,
        };
        return Some(Yuvalar {
            avg: b::avg_fn(t),
            w_avg: b::w_avg_fn(t),
            mask: b::mask_fn(t),
            w_mask: [b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)],
            blend: k::blend_fn(t),
            blend_v: k::blend_v_fn(t),
            blend_h: k::blend_h_fn(t),
            warp8x8: wp::warp8x8_fn(t),
            warp8x8t: wp::warp8x8t_fn(t),
            emu_edge: kn::emu_edge_fn(t),
            mc_scaled: av1oxide::dsp::mc::Filter2d::ALL.map(|f| ol::put_scaled_fn(t, f)),
            mct_scaled: av1oxide::dsp::mc::Filter2d::ALL.map(|f| ol::prep_scaled_fn(t, f)),
        });
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P33] 8 bpc AVX2 MC gövdeleri AVX2 ister",
    );
    None
}

/// P35: SSSE3 yuvaları (bütün yuvaların SSSE3 gövdesi var).
fn ssse3_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).ssse3_token()
    {
        use av1oxide::dsp::x86::{
            mc_bilesik_ssse3 as b, mc_karisim_ssse3 as k, mc_kenar_ssse3 as kn,
            mc_olcekli_ssse3 as ol, mc_warp_ssse3 as wp,
        };
        return Some(Yuvalar {
            avg: b::avg_fn(t),
            w_avg: b::w_avg_fn(t),
            mask: b::mask_fn(t),
            w_mask: [b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)],
            blend: k::blend_fn(t),
            blend_v: k::blend_v_fn(t),
            blend_h: k::blend_h_fn(t),
            warp8x8: wp::warp8x8_fn(t),
            warp8x8t: wp::warp8x8t_fn(t),
            emu_edge: kn::emu_edge_fn(t),
            mc_scaled: av1oxide::dsp::mc::Filter2d::ALL.map(|f| ol::put_scaled_fn(t, f)),
            mct_scaled: av1oxide::dsp::mc::Filter2d::ALL.map(|f| ol::prep_scaled_fn(t, f)),
        });
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P35] 8 bpc SSSE3 MC gövdeleri SSSE3 ister",
    );
    None
}

/// P36: AVX512ICL düzeyi — bileşik, karışım ve warp AVX512ICL gövdeleri; `emu_edge`
/// ve ölçekli işlevler AVX2 (dav1d'nin AVX512ICL düzeyi gibi; AVX512ICL gövdesi yok).
fn avx512_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx512icl_token()
    {
        use av1oxide::dsp::x86::{
            mc_bilesik_avx512 as b, mc_karisim_avx512 as k, mc_warp_avx512 as wp,
        };
        return Some(Yuvalar {
            avg: b::avg_fn(t),
            w_avg: b::w_avg_fn(t),
            mask: b::mask_fn(t),
            w_mask: [b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)],
            blend: k::blend_fn(t),
            blend_v: k::blend_v_fn(t),
            blend_h: k::blend_h_fn(t),
            warp8x8: wp::warp8x8_fn(t),
            warp8x8t: wp::warp8x8t_fn(t),
            ..avx2_yuvalari(test)?
        });
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P36] 8 bpc AVX512ICL MC gövdeleri AVX512ICL ister",
    );
    None
}

/// P35: SSE4.1 yuvaları (warp) skaler tablonun üzerine.
fn sse41_yuvalari(test: &str) -> Option<Yuvalar> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).sse41_token()
    {
        use av1oxide::dsp::x86::mc_warp_sse41 as wp;
        return Some(Yuvalar {
            warp8x8: wp::warp8x8_fn(t),
            warp8x8t: wp::warp8x8t_fn(t),
            ..Yuvalar::skaler()
        });
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P35] 8 bpc SSE4.1 MC gövdeleri SSE4.1 ister",
    );
    None
}

/// C `tmp_uret` ile aynı.
fn tmp_uret(rng: &mut Rng, t: &mut [i16], desen: u32, ikinci: usize) {
    let (lo, hi) = PREP_ARALIK;
    for (i, v) in t.iter_mut().enumerate() {
        *v = match desen {
            0 => rng.aralik(lo, hi),
            1 => {
                if (i + ikinci) & 1 != 0 {
                    hi
                } else {
                    lo
                }
            }
            2 => {
                if ikinci != 0 {
                    hi
                } else {
                    lo
                }
            }
            _ => rng.aralik(-64, 64),
        } as i16;
    }
}

/// C `maske_uret` ile aynı.
fn maske_uret(rng: &mut Rng, m: &mut [u8], desen: u32) {
    for (i, v) in m.iter_mut().enumerate() {
        *v = match desen {
            0 => rng.aralik(0, 64) as u8,
            1 => 0,
            2 => 64,
            _ => {
                if i & 1 != 0 {
                    64
                } else {
                    0
                }
            }
        };
    }
}

/// C `piksel_uret` ile aynı.
fn piksel_uret(rng: &mut Rng, p: &mut [u8], desen: u32, ikinci: usize) {
    for (i, v) in p.iter_mut().enumerate() {
        *v = match desen {
            1 => {
                if ikinci != 0 {
                    255
                } else {
                    0
                }
            }
            2 => {
                if (i + ikinci) & 1 != 0 {
                    255
                } else {
                    0
                }
            }
            _ => rng.aralik(0, 255) as u8,
        };
    }
}

fn fnv(h: &mut u64, b: &[u8]) {
    for &x in b {
        *h ^= u64::from(x);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

fn satirlar(h: &mut u64, d: &[u8], ds: usize, w: usize, yuk: usize) {
    for y in 0..yuk {
        fnv(h, &d[y * ds..y * ds + w]);
    }
}

/// Bir oracle hücresinin özeti.
fn hucre_ozeti(
    y: &Yuvalar,
    islev: u32,
    w: usize,
    h: usize,
    ds: usize,
    desen: u32,
    hucre: u32,
) -> u64 {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(u64::from(hucre) + 1));
    let n = w * h;
    let bd = BitDepth::Eight;
    let mut oz = 0xcbf2_9ce4_8422_2325u64;
    if islev >= 12 {
        // `ds` alanı = filtre·8 + parametre seti.
        let (f, ps) = (ds / 8, (ds % 8) as i32);
        let dy = match ps % 3 {
            0 => 1024,
            1 => 2048,
            _ => rng.aralik(512, 2048),
        };
        let dx = match ps {
            0 => 1024,
            1 => 2048,
            5 => 1,
            _ => rng.aralik(512, 2048),
        };
        let mx = match ps % 4 {
            0 => 0,
            1 => 1023,
            _ => rng.aralik(0, 1023),
        };
        let my = match (ps / 2) % 3 {
            0 => 0,
            1 => 1023,
            _ => rng.aralik(0, 1023),
        };
        let son_y = ((my + dy * (h as i32 - 1)) >> 10) as usize;
        let son_x = ((mx + dx * (w as i32 - 1)) >> 10) as usize;
        let ek = if f == 9 { 4 } else { 7 };
        let ss = son_x + 8 + 5 * (ps as usize % 3);
        let mut src = vec![0u8; (son_y + ek) * ss + son_x + ek + 1];
        piksel_uret(&mut rng, &mut src, desen, 1);
        if islev == 12 {
            let dsp = w + 3;
            let mut d = vec![0u8; dsp * h];
            (y.mc_scaled[f])(&mut d, dsp, &src, ss, w, h, mx, my, dx, dy, bd, 255);
            satirlar(&mut oz, &d, dsp, w, h);
        } else {
            let mut t = vec![0i16; w * h];
            (y.mct_scaled[f])(&mut t, &src, ss, w, h, mx, my, dx, dy, bd, 255);
            for v in t {
                fnv(&mut oz, &v.to_le_bytes());
            }
        }
        return oz;
    }
    let mut dst = vec![0u8; ds * h];
    if islev == 11 {
        const BW: [usize; 11] = [4, 8, 11, 16, 23, 32, 39, 64, 71, 128, 135];
        let e = (hucre - EMU_BAS) as usize;
        let (pk, xk, yk) = (e % 25, e % 25 % 5, e % 25 / 5);
        let _ = pk;
        let (iw, ih) = (
            if e % 4 < 2 { 160 } else { 48 },
            if e % 4 == 1 || e % 4 == 2 { 48 } else { 160 },
        );
        let ss = iw + 7 * (e % 2);
        let konum = |kk: usize, b: usize, i: usize| -> i32 {
            let (b, i) = (b as i32, i as i32);
            match kk {
                0 => {
                    if i >= b {
                        (i - b) / 2
                    } else {
                        0
                    }
                }
                1 => -(b / 2),
                2 => -b - 3,
                3 => i - b / 2,
                _ => i + 3,
            }
        };
        assert_eq!(
            (w, h),
            (BW[e / 25], BW[(e / 25 + 3) % 11]),
            "emu hücre sırası"
        );
        let mut src = vec![0u8; ss * ih];
        piksel_uret(&mut rng, &mut src, desen, 1);
        let (x, yy) = (konum(xk, w, iw), konum(yk, h, ih));
        (y.emu_edge)(w, h, iw, ih, x, yy, &mut dst, ds, &src, ss, bd, 255);
        satirlar(&mut oz, &dst, ds, w, h);
        return oz;
    }
    if islev >= 9 {
        let ps = ((hucre - WARP_BAS) / 3) % 27;
        let ss = ds + 7;
        let mut src = vec![0u8; 14 * ss + 31];
        piksel_uret(&mut rng, &mut src, desen, 1);
        let abcd: [i16; 4] = core::array::from_fn(|_| rng.aralik(-1024, 1024) as i16);
        let mut mxy = [0i32; 2];
        for (e, v) in mxy.iter_mut().enumerate() {
            let (sr, ks, kx) = if e == 1 {
                (7, i32::from(abcd[3]), i32::from(abcd[2]))
            } else {
                (14, i32::from(abcd[1]), i32::from(abcd[0]))
            };
            let omin = sr * ks.min(0) + 7 * kx.min(0);
            let omax = sr * ks.max(0) + 7 * kx.max(0);
            let (lo, hi) = (-66048 - omin, 131_583 - omax);
            let kip = if e == 1 { (ps / 3) % 3 } else { ps % 3 };
            *v = match kip {
                0 => lo,
                1 => hi,
                _ => rng.aralik(lo, hi),
            };
        }
        if islev == 9 {
            (y.warp8x8)(&mut dst, ds, &src, ss, &abcd, mxy[0], mxy[1], bd, 255);
            satirlar(&mut oz, &dst, ds, 8, 8);
        } else {
            let mut t = vec![0i16; ds * 8];
            (y.warp8x8t)(&mut t, ds, &src, ss, &abcd, mxy[0], mxy[1], bd, 255);
            for r in 0..8 {
                for &v in &t[r * ds..r * ds + 8] {
                    fnv(&mut oz, &v.to_le_bytes());
                }
            }
        }
        return oz;
    }
    if islev >= 6 {
        let mut d0 = vec![0u8; ds * h];
        let mut tp = vec![0u8; n];
        piksel_uret(&mut rng, &mut d0, desen, 0);
        piksel_uret(&mut rng, &mut tp, desen, 1);
        let ncagri = if islev == 6 { 4 } else { 1 };
        let mut m = vec![0u8; n];
        for md in 0..ncagri {
            dst.copy_from_slice(&d0);
            match islev {
                6 => {
                    maske_uret(&mut rng, &mut m, md);
                    (y.blend)(&mut dst, ds, &tp, w, h, &m, bd, 255);
                }
                7 => (y.blend_v)(&mut dst, ds, &tp, w, h, bd, 255),
                _ => (y.blend_h)(&mut dst, ds, &tp, w, h, bd, 255),
            }
            satirlar(&mut oz, &dst, ds, w, h);
        }
        return oz;
    }
    let (mut t1, mut t2) = (vec![0i16; n], vec![0i16; n]);
    tmp_uret(&mut rng, &mut t1, desen, 0);
    tmp_uret(&mut rng, &mut t2, desen, 1);
    match islev {
        0 => {
            (y.avg)(&mut dst, ds, &t1, &t2, w, h, bd, 255);
            satirlar(&mut oz, &dst, ds, w, h);
        }
        1 => {
            for wt in 0..=16 {
                dst.fill(0);
                (y.w_avg)(&mut dst, ds, &t1, &t2, w, h, wt, bd, 255);
                satirlar(&mut oz, &dst, ds, w, h);
            }
        }
        2 => {
            let mut m = vec![0u8; n];
            for md in 0..4 {
                maske_uret(&mut rng, &mut m, md);
                dst.fill(0);
                (y.mask)(&mut dst, ds, &t1, &t2, w, h, &m, bd, 255);
                satirlar(&mut oz, &dst, ds, w, h);
            }
        }
        _ => {
            let alt = (islev - 3) as usize;
            let (sh, sv) = (alt != 0, alt == 2);
            let mn = (w >> usize::from(sh)) * (h >> usize::from(sv));
            for sign in 0..2 {
                dst.fill(0);
                let mut m = vec![0u8; mn];
                (y.w_mask[alt])(&mut dst, ds, &t1, &t2, w, h, &mut m, sign, sh, sv, bd, 255);
                satirlar(&mut oz, &dst, ds, w, h);
                fnv(&mut oz, &m);
            }
        }
    }
    oz
}

#[test]
#[ignore = "dav1d C oracle çıktısı ister: bash scripts/oracle/run_mc_oracle.sh (tam kip)"]
fn p33_taramali_oracle_dav1d_c_ile_bit_exact() {
    let Some(avx) = avx2_yuvalari("mc_p33_oracle::p33_taramali_oracle_dav1d_c_ile_bit_exact")
    else {
        return;
    };
    let Some(sse) = ssse3_yuvalari("mc_p33_oracle::p33_taramali_oracle_dav1d_c_ile_bit_exact")
    else {
        return;
    };
    let Some(s41) = sse41_yuvalari("mc_p33_oracle::p33_taramali_oracle_dav1d_c_ile_bit_exact")
    else {
        return;
    };
    // P39 ara karar 3: AVX512ICL kısmı isteğe bağlı; yoksa yalnız o kısım ön koşul
    // kaydıyla atlanır, AVX2/SSSE3/SSE4.1/skaler taraması her x86_64'te koşar.
    let z = avx512_yuvalari("mc_p33_oracle::p33_taramali_oracle_dav1d_c_ile_bit_exact[avx512icl]");
    let dizin = std::env::var("AV1OXIDE_MC_P33_ORACLE").map_or_else(
        |_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/mc-p33-oracle"),
        std::path::PathBuf::from,
    );
    let veri = std::fs::read(dizin.join("p33-expect.bin")).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — önce bash scripts/oracle/run_mc_oracle.sh",
            dizin.display()
        )
    });
    assert_eq!(&veri[..8], b"AV1OXMP3");
    let u32le = |o: usize| u32::from_le_bytes(veri[o..o + 4].try_into().expect("4 bayt"));
    assert_eq!(
        u32le(8),
        SURUM,
        "oracle sürümü (yeniden üret: run_mc_oracle.sh)"
    );
    let nhucre = u32le(12) as usize;
    let skaler = Yuvalar::skaler();
    let mut yollar = vec![("avx2", &avx), ("ssse3", &sse), ("sse41", &s41)];
    if let Some(z) = &z {
        yollar.push(("avx512icl", z));
    }
    yollar.push(("skaler", &skaler));
    let devir0 = skaler_devir_sayisi();
    let mut islev_say = [0usize; 14];
    for c in 0..nhucre {
        let b = 16 + c * 28;
        let (islev, w, h, ds, desen) = (
            u32le(b),
            u32le(b + 4) as usize,
            u32le(b + 8) as usize,
            u32le(b + 12) as usize,
            u32le(b + 16),
        );
        let want = u64::from_le_bytes(veri[b + 20..b + 28].try_into().expect("8 bayt"));
        for &(ad, y) in &yollar {
            let got = hucre_ozeti(y, islev, w, h, ds, desen, c as u32);
            assert_eq!(
                got, want,
                "{ad} işlev {islev} {w}x{h} ds={ds} desen={desen}: dav1d C özeti farklı"
            );
        }
        islev_say[islev as usize] += 1;
    }
    // Bu test kendi başına koşar (sayaç kilidi yok); devir sayacı süreç geneldir.
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "oracle taramasında skaler devir olmamalı"
    );
    let kume = if z.is_some() {
        "AVX2 + SSSE3 + SSE4.1 + AVX512ICL + skaler"
    } else {
        "AVX2 + SSSE3 + SSE4.1 + skaler; AVX512ICL atlandı: ön koşul"
    };
    println!(
        "[P33 oracle] {nhucre} hücre ({kume}) dav1d C ile bit-exact · işlev başına {islev_say:?}"
    );
}
