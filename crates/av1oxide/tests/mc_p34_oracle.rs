//! P34 (F4-MC c) — 16 bpc (10/12 bit) taramalı oracle: AVX2 ve skaler yuvalar
//! dav1d C ile bit-exact (`scripts/oracle/mc_p34_oracle.c`, `--ignored`).
//!
//! Girdi düzlemleri ve hücre özetleri `.cache/mc-p34-oracle/` altındadır
//! (`bash scripts/oracle/run_mc_oracle.sh` üretir; dizin
//! `AV1OXIDE_MC_P34_ORACLE` ile değiştirilebilir). `put`/`prep` düzlemleri
//! girdi dosyasından, öbür işlevlerin girdileri hücre tohumundan (C ile aynı
//! üreteç) gelir. Kaynak dilimine kopyasız sözleşmenin kuyruğu (`MC_KUYRUK`)
//! test tarafında eklenir; sözleşmedeki çağrılarda skaler devir 0 olmalıdır.

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::mc::{MC_KUYRUK, McDsp, skaler_devir_sayisi};
use av1oxide::frame::BitDepth;
use common::on_kosul::{self, Neden};

/// Oracle biçim sürümü (`mc_p34_oracle.c`).
const SURUM: usize = 6;

/// AVX2 16 bpc yuvaları skaler tablonun üzerine (10 bit, 12 bit).
fn avx2_tablolari(test: &str) -> Option<[McDsp; 2]> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx2_token()
    {
        use av1oxide::dsp::mc::Filter2d;
        use av1oxide::dsp::x86::{
            mc16, mc16_bilesik as b, mc16_karisim as k, mc16_kenar as kn, mc16_olcekli as o,
            mc16_warp as wp,
        };
        let kur = |mut d: McDsp| {
            d.mc = Filter2d::ALL.map(|f| mc16::put16_fn(t, f));
            d.mct = Filter2d::ALL.map(|f| mc16::prep16_fn(t, f));
            d.avg = b::avg_fn(t);
            d.w_avg = b::w_avg_fn(t);
            d.mask = b::mask_fn(t);
            d.w_mask = [b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)];
            d.blend = k::blend_fn(t);
            d.blend_v = k::blend_v_fn(t);
            d.blend_h = k::blend_h_fn(t);
            d.warp8x8 = wp::warp8x8_fn(t);
            d.warp8x8t = wp::warp8x8t_fn(t);
            d.emu_edge = kn::emu_edge_fn(t);
            d.mc_scaled = Filter2d::ALL.map(|f| o::put_scaled_fn(t, f));
            d.mct_scaled = Filter2d::ALL.map(|f| o::prep_scaled_fn(t, f));
            d
        };
        return Some([kur(McDsp::scalar::<10>()), kur(McDsp::scalar::<12>())]);
    }
    on_kosul::eksik(
        test,
        Neden::avx2(),
        "[P34] 16 bpc AVX2 MC gövdeleri AVX2 ister",
    );
    None
}

/// P35: SSSE3 16 bpc yuvaları skaler tablonun üzerine (10 bit, 12 bit); SSSE3
/// gövdesi olmayan yuvalar skaler kalır.
fn ssse3_tablolari(test: &str) -> Option<[McDsp; 2]> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).ssse3_token()
    {
        use av1oxide::dsp::mc::Filter2d;
        use av1oxide::dsp::x86::{
            mc16_bilesik_ssse3 as b, mc16_karisim_ssse3 as k, mc16_kenar_ssse3 as kn,
            mc16_olcekli_ssse3 as o, mc16_ssse3, mc16_warp_ssse3 as wp,
        };
        let kur = |mut d: McDsp| {
            d.mc = Filter2d::ALL.map(|f| mc16_ssse3::put16_fn(t, f));
            d.mct = Filter2d::ALL.map(|f| mc16_ssse3::prep16_fn(t, f));
            d.avg = b::avg_fn(t);
            d.w_avg = b::w_avg_fn(t);
            d.mask = b::mask_fn(t);
            d.w_mask = [b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)];
            d.blend = k::blend_fn(t);
            d.blend_v = k::blend_v_fn(t);
            d.blend_h = k::blend_h_fn(t);
            d.warp8x8 = wp::warp8x8_fn(t);
            d.warp8x8t = wp::warp8x8t_fn(t);
            d.emu_edge = kn::emu_edge_fn(t);
            d.mc_scaled = Filter2d::ALL.map(|f| o::put_scaled_fn(t, f));
            d.mct_scaled = Filter2d::ALL.map(|f| o::prep_scaled_fn(t, f));
            d
        };
        return Some([kur(McDsp::scalar::<10>()), kur(McDsp::scalar::<12>())]);
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P35] 16 bpc SSSE3 MC gövdeleri SSSE3 ister",
    );
    None
}

/// P36: AVX512ICL 16 bpc düzeyi (10 bit, 12 bit) — put/prep, bileşik, karışım ve warp
/// AVX512ICL gövdeleri; `emu_edge` ve ölçekli işlevler AVX2 (dav1d'nin AVX512ICL düzeyi gibi).
fn avx512_tablolari(test: &str) -> Option<[McDsp; 2]> {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) =
        av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect()).avx512icl_token()
    {
        use av1oxide::dsp::mc::Filter2d;
        use av1oxide::dsp::x86::{
            mc16_avx512, mc16_bilesik_avx512 as b, mc16_karisim_avx512 as k, mc16_warp_avx512 as wp,
        };
        let kur = |mut d: McDsp| {
            d.mc = Filter2d::ALL.map(|f| mc16_avx512::put16_fn(t, f));
            d.mct = Filter2d::ALL.map(|f| mc16_avx512::prep16_fn(t, f));
            d.avg = b::avg_fn(t);
            d.w_avg = b::w_avg_fn(t);
            d.mask = b::mask_fn(t);
            d.w_mask = [b::w_mask_fn(t, 0), b::w_mask_fn(t, 1), b::w_mask_fn(t, 2)];
            d.blend = k::blend_fn(t);
            d.blend_v = k::blend_v_fn(t);
            d.blend_h = k::blend_h_fn(t);
            d.warp8x8 = wp::warp8x8_fn(t);
            d.warp8x8t = wp::warp8x8t_fn(t);
            d
        };
        let [a10, a12] = avx2_tablolari(test)?;
        return Some([kur(a10), kur(a12)]);
    }
    on_kosul::eksik(
        test,
        Neden::isa(),
        "[P36] 16 bpc AVX512ICL MC gövdeleri AVX512ICL ister",
    );
    None
}

fn u32le(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

fn fnv(h: &mut u64, b: &[u8]) {
    for &x in b {
        *h ^= u64::from(x);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

/// `h` satır × `2w` bayt özete.
fn satirlar(oz: &mut u64, d: &[u8], ds: usize, w: usize, h: usize) {
    for y in 0..h {
        fnv(oz, &d[y * ds..y * ds + 2 * w]);
    }
}

/// xorshift64* (C tarafıyla aynı).
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
    /// `[lo, hi]` (C: `lo + (u64 >> 33) % (hi − lo + 1)`).
    fn aralik(&mut self, lo: i32, hi: i32) -> i32 {
        lo + ((self.u64() >> 33) % (hi - lo + 1) as u64) as i32
    }
}

/// 16 bpc `prep` aralığı (dav1d `mc_tmpl.c:41-48`): 10 bit, 12 bit.
const PREP16_ARALIK: [(i32, i32); 2] = [(-28780, 28764), (-28794, 28791)];

fn tmp_uret(rng: &mut Rng, n: usize, bi: usize, desen: usize, ikinci: usize) -> Vec<i16> {
    let (lo, hi) = PREP16_ARALIK[bi];
    (0..n)
        .map(|i| {
            (match desen {
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
                _ => rng.aralik(-32768, 32767),
            }) as i16
        })
        .collect()
}

fn maske_uret(rng: &mut Rng, n: usize, desen: usize) -> Vec<u8> {
    (0..n)
        .map(|i| match desen {
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
        })
        .collect()
}

struct Duzlem {
    bpc: usize,
    w: usize,
    h: usize,
    ss: usize,
    src: Vec<u8>,
}

/// `put`/`prep` hücresi (işlev 0, 1): 256 çağrının özeti.
fn put_prep(dsp: &McDsp, islev: usize, dz: &Duzlem, f: usize, bd: BitDepth, max: u16) -> u64 {
    let (w, h, ss) = (dz.w, dz.h, dz.ss);
    let mut oz = 0xcbf2_9ce4_8422_2325u64;
    for mx in 0..16u8 {
        for my in 0..16u8 {
            if islev == 0 {
                let mut d = vec![0u8; 2 * w * h];
                (dsp.mc[f])(&mut d, 2 * w, &dz.src, ss, w, h, mx, my, bd, max);
                fnv(&mut oz, &d);
            } else {
                let mut t = vec![0i16; w * h];
                (dsp.mct[f])(&mut t, &dz.src, ss, w, h, mx, my, bd, max);
                for v in t {
                    fnv(&mut oz, &v.to_le_bytes());
                }
            }
        }
    }
    oz
}

/// Bileşik hücre (işlev 2..7): girdi hücre tohumundan; (özet, çağrı sayısı).
#[allow(clippy::too_many_arguments)]
fn bilesik(
    dsp: &McDsp,
    islev: usize,
    hucre: usize,
    bi: usize,
    w: usize,
    h: usize,
    ds: usize,
    desen: usize,
    bd: BitDepth,
    max: u16,
) -> (u64, usize) {
    let n = w * h;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(hucre as u64 + 1));
    let t1 = tmp_uret(&mut rng, n, bi, desen, 0);
    let t2 = tmp_uret(&mut rng, n, bi, desen, 1);
    let mut oz = 0xcbf2_9ce4_8422_2325u64;
    let mut cagri = 0;
    let mut d = vec![0u8; ds * h];
    match islev {
        2 => {
            (dsp.avg)(&mut d, ds, &t1, &t2, w, h, bd, max);
            satirlar(&mut oz, &d, ds, w, h);
            cagri += 1;
        }
        3 => {
            for wt in 0..=16 {
                d.fill(0);
                (dsp.w_avg)(&mut d, ds, &t1, &t2, w, h, wt, bd, max);
                satirlar(&mut oz, &d, ds, w, h);
                cagri += 1;
            }
        }
        4 => {
            for md in 0..4 {
                let m = maske_uret(&mut rng, n, md);
                d.fill(0);
                (dsp.mask)(&mut d, ds, &t1, &t2, w, h, &m, bd, max);
                satirlar(&mut oz, &d, ds, w, h);
                cagri += 1;
            }
        }
        _ => {
            let alt = islev - 5;
            let (sh, sv) = (alt != 0, alt == 2);
            let mn = (w >> usize::from(sh)) * (h >> usize::from(sv));
            for sign in 0..2 {
                d.fill(0);
                let mut m = vec![0u8; mn];
                (dsp.w_mask[alt])(&mut d, ds, &t1, &t2, w, h, &mut m, sign, sh, sv, bd, max);
                satirlar(&mut oz, &d, ds, w, h);
                fnv(&mut oz, &m);
                cagri += 1;
            }
        }
    }
    (oz, cagri)
}

/// Karışım pikselleri (`u16` yerel bayt; desen: 0 ve 3 rastgele, 1 hepsi 0 /
/// `max`, 2 dama).
fn piksel_uret(rng: &mut Rng, n: usize, max: u16, desen: usize, ikinci: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let v = match desen {
                1 => {
                    if ikinci != 0 {
                        max
                    } else {
                        0
                    }
                }
                2 => {
                    if (i + ikinci) & 1 != 0 {
                        max
                    } else {
                        0
                    }
                }
                _ => rng.aralik(0, i32::from(max)) as u16,
            };
            v.to_ne_bytes()
        })
        .collect()
}

/// Karışım hücresi (işlev 8..10): girdi hücre tohumundan; (özet, çağrı sayısı).
#[allow(clippy::too_many_arguments)]
fn karisim(
    dsp: &McDsp,
    islev: usize,
    hucre: usize,
    w: usize,
    h: usize,
    ds: usize,
    desen: usize,
    bd: BitDepth,
    max: u16,
) -> (u64, usize) {
    let n = w * h;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(hucre as u64 + 1));
    let d0 = piksel_uret(&mut rng, (ds / 2) * h, max, desen, 0);
    let tp = piksel_uret(&mut rng, n, max, desen, 1);
    let mut oz = 0xcbf2_9ce4_8422_2325u64;
    let ncagri = if islev == 8 { 4 } else { 1 };
    for md in 0..ncagri {
        let mut d = d0.clone();
        match islev {
            8 => {
                let m = maske_uret(&mut rng, n, md);
                (dsp.blend)(&mut d, ds, &tp, w, h, &m, bd, max);
            }
            9 => (dsp.blend_v)(&mut d, ds, &tp, w, h, bd, max),
            _ => (dsp.blend_h)(&mut d, ds, &tp, w, h, bd, max),
        }
        satirlar(&mut oz, &d, ds, w, h);
    }
    (oz, ncagri)
}

/// Warp hücresi (işlev 11 `warp8x8`, 12 `warp8x8t`): girdi hücre tohumundan;
/// `ps` parametre seti (0…26; bölümde hücre sırası `bpc × işlev × ps × ss`).
#[allow(clippy::too_many_arguments)]
fn warp(
    dsp: &McDsp,
    islev: usize,
    hucre: usize,
    ps: usize,
    ss: usize,
    desen: usize,
    bd: BitDepth,
    max: u16,
) -> u64 {
    let ek = ss - 30;
    let (sp, ds, ts) = (ss / 2, 16 + ek, 8 + ek / 2);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(hucre as u64 + 1));
    let mut src = piksel_uret(&mut rng, 14 * sp + 15, max, desen, 1);
    src.resize(src.len() + MC_KUYRUK, 0);
    let abcd: [i16; 4] = core::array::from_fn(|_| rng.aralik(-1024, 1024) as i16);
    let mut mxy = [0i32; 2];
    for (e, v) in mxy.iter_mut().enumerate() {
        // e = 0: yatay (satır 0..14, katsayılar b, a); e = 1: dikey (0..7, d, c).
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
    let mut oz = 0xcbf2_9ce4_8422_2325u64;
    if islev == 11 {
        let mut d = vec![0u8; ds * 8];
        (dsp.warp8x8)(&mut d, ds, &src, ss, &abcd, mxy[0], mxy[1], bd, max);
        satirlar(&mut oz, &d, ds, 8, 8);
    } else {
        let mut t = vec![0i16; ts * 8];
        (dsp.warp8x8t)(&mut t, ts, &src, ss, &abcd, mxy[0], mxy[1], bd, max);
        for y in 0..8 {
            for v in &t[y * ts..y * ts + 8] {
                fnv(&mut oz, &v.to_le_bytes());
            }
        }
    }
    oz
}

/// `emu_edge` hücresi (işlev 13): `e` bit derinliği içindeki emu hücre sırası
/// (görüntü boyu, satır adımları ve konum ondan türetilir).
#[allow(clippy::too_many_arguments)]
fn emu(
    dsp: &McDsp,
    hucre: usize,
    e: usize,
    bw: usize,
    bh: usize,
    ds: usize,
    desen: usize,
    bd: BitDepth,
    max: u16,
) -> u64 {
    let (xk, yk) = (e % 25 % 5, e % 25 / 5);
    let iw = if e % 4 < 2 { 160usize } else { 48 };
    let ih = if e % 4 == 1 || e % 4 == 2 {
        48usize
    } else {
        160
    };
    let ss = 2 * (iw + 7 * (e % 2));
    let konum = |b: usize, i: usize, kk: usize| -> i32 {
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
    let (x, y) = (konum(bw, iw, xk), konum(bh, ih, yk));
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(hucre as u64 + 1));
    let goruntu = piksel_uret(&mut rng, ss / 2 * ih, max, desen, 1);
    let mut d = vec![0u8; ds * bh];
    (dsp.emu_edge)(bw, bh, iw, ih, x, y, &mut d, ds, &goruntu, ss, bd, max);
    let mut oz = 0xcbf2_9ce4_8422_2325u64;
    satirlar(&mut oz, &d, ds, bw, bh);
    oz
}

/// Ölçekli hücre (işlev 14 `mc_scaled`, 15 `mct_scaled`): `a = f·8 + ps`,
/// parametreler ve pikseller hücre tohumundan.
#[allow(clippy::too_many_arguments)]
fn olcekli(
    dsp: &McDsp,
    islev: usize,
    hucre: usize,
    w: usize,
    h: usize,
    a: usize,
    desen: usize,
    bd: BitDepth,
    max: u16,
) -> u64 {
    let (f, ps) = (a / 8, (a % 8) as i32);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(hucre as u64 + 1));
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
    let sp = son_x + 8 + 5 * (ps as usize % 3);
    let n = (son_y + ek) * sp + son_x + ek + 1;
    let src = piksel_uret(&mut rng, n, max, desen, 1);
    let mut oz = 0xcbf2_9ce4_8422_2325u64;
    if islev == 14 {
        let ds = 2 * w + 6;
        let mut d = vec![0u8; ds * h];
        (dsp.mc_scaled[f])(&mut d, ds, &src, 2 * sp, w, h, mx, my, dx, dy, bd, max);
        satirlar(&mut oz, &d, ds, w, h);
    } else {
        let mut t = vec![0i16; w * h];
        (dsp.mct_scaled[f])(&mut t, &src, 2 * sp, w, h, mx, my, dx, dy, bd, max);
        for v in t {
            fnv(&mut oz, &v.to_le_bytes());
        }
    }
    oz
}

const ISLEV_ADI: [&str; 16] = [
    "put",
    "prep",
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
    "mc_scaled",
    "mct_scaled",
];

#[test]
#[ignore = "taramalı oracle: .cache/mc-p34-oracle (--ignored, release)"]
fn p34_taramali_oracle_dav1d_c_ile_bit_exact() {
    let kok = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dizin = std::env::var_os("AV1OXIDE_MC_P34_ORACLE").map_or_else(
        || kok.join(".cache/mc-p34-oracle"),
        std::path::PathBuf::from,
    );
    let (Ok(girdi), Ok(beklenen)) = (
        std::fs::read(dizin.join("p34-input.bin")),
        std::fs::read(dizin.join("p34-expect.bin")),
    ) else {
        on_kosul::eksik(
            "mc_p34_oracle::p34_taramali_oracle_dav1d_c_ile_bit_exact",
            Neden::OracleVerisiYok,
            ".cache/mc-p34-oracle yok; üretmek için: bash scripts/oracle/run_mc_oracle.sh",
        );
        return;
    };
    let Some(avx) = avx2_tablolari("mc_p34_oracle::p34_taramali_oracle_dav1d_c_ile_bit_exact")
    else {
        return;
    };
    let Some(sse) = ssse3_tablolari("mc_p34_oracle::p34_taramali_oracle_dav1d_c_ile_bit_exact")
    else {
        return;
    };
    // P39 ara karar 3: AVX512ICL kısmı isteğe bağlı; yoksa yalnız o kısım ön koşul
    // kaydıyla atlanır, AVX2/SSSE3/skaler taraması her x86_64'te koşar.
    let z = avx512_tablolari("mc_p34_oracle::p34_taramali_oracle_dav1d_c_ile_bit_exact[avx512icl]");
    let skaler = [McDsp::scalar::<10>(), McDsp::scalar::<12>()];
    let mut yollar = vec![("avx2", &avx), ("ssse3", &sse)];
    if let Some(z) = &z {
        yollar.push(("avx512icl", z));
    }
    yollar.push(("skaler", &skaler));
    assert_eq!(&girdi[..8], b"AV1OXMQ1");
    assert_eq!(&beklenen[..8], b"AV1OXMQ2");
    assert_eq!(u32le(&girdi, 8), SURUM, "girdi sürümü");
    assert_eq!(u32le(&beklenen, 8), SURUM, "beklenen sürümü");
    let ndz = u32le(&girdi, 12);
    let mut duzlemler = Vec::with_capacity(ndz);
    let mut o = 16usize;
    for _ in 0..ndz {
        let (bpc, w, h, ss, n) = (
            u32le(&girdi, o),
            u32le(&girdi, o + 4),
            u32le(&girdi, o + 8),
            u32le(&girdi, o + 12),
            u32le(&girdi, o + 20),
        );
        o += 24;
        let mut src = girdi[o..o + n].to_vec();
        src.resize(n + MC_KUYRUK, 0);
        o += n;
        duzlemler.push(Duzlem { bpc, w, h, ss, src });
    }
    assert_eq!(o, girdi.len(), "girdi uzunluğu");
    let nhucre = u32le(&beklenen, 12);
    assert_eq!(16 + nhucre * 32, beklenen.len(), "beklenen uzunluğu");
    let devir0 = skaler_devir_sayisi();
    let mut cagri = 0usize;
    // Warp bölümünün ilk hücresi (parametre seti sırası ondan sayılır).
    let mut warp_bas = None;
    // `emu_edge` bölümünün ilk hücresi (`e` = bit derinliği içindeki sıra).
    let mut emu_bas = None;
    // [işlev][bit derinliği] hücre sayısı
    let mut hucre = [[0usize; 2]; ISLEV_ADI.len()];
    for c in 0..nhucre {
        let b = 16 + c * 32;
        let (islev, bpc, w, h, a, bb) = (
            u32le(&beklenen, b),
            u32le(&beklenen, b + 4),
            u32le(&beklenen, b + 8),
            u32le(&beklenen, b + 12),
            u32le(&beklenen, b + 16),
            u32le(&beklenen, b + 20),
        );
        let want = u64::from_le_bytes(beklenen[b + 24..b + 32].try_into().expect("8 bayt"));
        let (bd, max, bi) = match bpc {
            10 => (BitDepth::Ten, 1023u16, 0usize),
            12 => (BitDepth::Twelve, 4095, 1),
            _ => panic!("hücre {c}: bpc={bpc}"),
        };
        assert!(
            islev < ISLEV_ADI.len(),
            "hücre {c}: bilinmeyen işlev {islev}"
        );
        for &(ad, tablo) in &yollar {
            let dsp = &tablo[bi];
            let (oz, n) = match islev {
                0 | 1 => {
                    let dz = &duzlemler[a];
                    assert_eq!((dz.bpc, dz.w, dz.h), (bpc, w, h), "hücre {c}: düzlem");
                    (put_prep(dsp, islev, dz, bb, bd, max), 256)
                }
                2..=7 => bilesik(dsp, islev, c, bi, w, h, a, bb, bd, max),
                8..=10 => karisim(dsp, islev, c, w, h, a, bb, bd, max),
                11 | 12 => {
                    let ps = (c - *warp_bas.get_or_insert(c)) / 3 % 27;
                    (warp(dsp, islev, c, ps, a, bb, bd, max), 1)
                }
                13 => {
                    let e = (c - *emu_bas.get_or_insert(c)) % 275;
                    (emu(dsp, c, e, w, h, a, bb, bd, max), 1)
                }
                _ => (olcekli(dsp, islev, c, w, h, a, bb, bd, max), 1),
            };
            assert_eq!(
                oz, want,
                "{ad} {bpc} bit {} {w}x{h} a={a} b={bb}: dav1d C özeti farklı (hücre {c})",
                ISLEV_ADI[islev]
            );
            cagri += n;
        }
        hucre[islev][bi] += 1;
    }
    assert_eq!(
        skaler_devir_sayisi(),
        devir0,
        "taramalı oracle'da skaler devir olmamalı"
    );
    let dokum: Vec<String> = ISLEV_ADI
        .iter()
        .zip(hucre)
        .map(|(ad, [a, b])| {
            assert!(a > 0 && b > 0, "{ad}: 10 ve 12 bit hücreleri ({a}, {b})");
            format!("{ad} {a}+{b}")
        })
        .collect();
    let kume = if z.is_some() {
        "AVX2 + SSSE3 + AVX512ICL + skaler"
    } else {
        "AVX2 + SSSE3 + skaler; AVX512ICL atlandı: ön koşul"
    };
    println!(
        "[P34 oracle] sürüm {SURUM} · {ndz} düzlem · {nhucre} hücre ({}) · {cagri} çağrı ({kume}) dav1d C ile bit-exact",
        dokum.join(", ")
    );
}
