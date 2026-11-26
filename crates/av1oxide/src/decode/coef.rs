//! Katsayı çözme ve dequant (S2b.3).
//!
//! dav1d `src/recon_tmpl.c:49-730` (`read_golomb`, `get_skip_ctx`,
//! `get_dc_sign_ctx`, `get_lo_ctx`, `decode_coefs`) C yolunun birebir
//! karşılığı. Katsayı **token**'ları MSAC ile çözülür, `dequant` AV1
//! §7.12.3'e göre uygulanır ve `cf` tamponu itx çekirdeğine hazırlanır.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! | Kural | dav1d satırı | AV1 § |
//! |---|---|---|
//! | `read_golomb` | `recon_tmpl.c:49-57` | 8.3.2 |
//! | `get_skip_ctx` | `:59-138` | 8.3.2 |
//! | `get_dc_sign_ctx` | `:140-296` | 8.3.2 |
//! | `get_lo_ctx` | `:298-319` | 8.3.2 |
//! | `decode_coefs` (eob, token, işaret) | `:321-728` | 5.11.39 |
//! | dequant (`dq_tbl`, `dq_shift`, `cf_max`) | `:558-720` | 7.12.3 |
//! | tarama tabloları | `src/scan.c` | 5.11.39 |
//! | varsayılan katsayı CDF'leri | `src/cdf.c:699` | 8.2.6 |
//!
//! ## Kuantizasyon matrisleri
//!
//! `using_qmatrix` dalı desteklenir (`CoefEnv::qm` seçimi; dav1d ayrı
//! çarpım `:583-665`). Bu modülde kare düzeyinde
//! [`crate::error::Error::Unsupported`] yolu yoktur (S2d.1 denetimi).

#![forbid(unsafe_code)]

use crate::entropy::CdfContext;
use crate::entropy::cdf_layout as layout;
use crate::entropy::msac::MsacContext;
use crate::frame::BitDepth;
use crate::tables::decode as t;

/// Bir katsayı bloğunun çözüm sonucu.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoefResult {
    /// Katsayı sonu; `-1` = bütünüyle sıfır (itx çağrılmaz).
    pub eob: i32,
    /// Tx türü indeksi (0..=16; 16 = WHT).
    pub txtp: u8,
    /// Komşu bağlamı için `cul_level | dc_sign_level`.
    pub res_ctx: u8,
}

/// `decode_coefs` çevre sabitleri (kare ve blok düzeyinden).
#[derive(Clone, Copy)]
pub(crate) struct CoefEnv<'a> {
    /// `segmentation.lossless[seg_id]`.
    pub lossless: bool,
    /// `reduced_txtp_set`.
    pub reduced_txtp: bool,
    /// Segmentin qidx'i (luma tx türü için `qidx == 0` testi).
    pub seg_qidx: u8,
    /// Bit derinliği (dequant `cf_max` ve depolama).
    pub bit_depth: BitDepth,
    /// 4:2:0 dikey alt örnekleme.
    pub ss_ver: bool,
    /// 4:2:0/4:2:2 yatay alt örnekleme.
    pub ss_hor: bool,
    /// Blok segmenti için dequant tablosu: `[plane][{dc, ac}]`.
    pub dq: &'a [[u16; 2]; 3],
    /// Kuantizasyon matrisi seçimi (varsa; `txtp < IDTX` ve qm etkin).
    pub qm: Option<QmSel>,
    /// Inter (intrabc) katsayı yolu: tx türü CDF'den okunur
    /// (`recon_tmpl.c:376-412`, `intra = 0`).
    pub inter: bool,
    /// Kroma için luma tx türü (`get_uv_inter_txtp`, `env.h:120-133`).
    ///
    /// Yalnız `inter && chroma` iken okunur; diğer yollarda 0.
    pub txtp_hint: u8,
}

/// Kuantizasyon matrisi seçimi (seviye + sınıf + tx boyutu).
#[derive(Clone, Copy, Debug)]
pub(crate) struct QmSel {
    /// `qm_y`/`qm_u`/`qm_v` (0–15; 15 = matris yok).
    pub level: u8,
    /// 0 = luma, 1 = chroma (`dav1d_qm_tbl[..][class]`).
    pub class: usize,
    /// Dikdörtgen tx boyutu indeksi (0–18).
    pub tx: u8,
}

impl QmSel {
    /// Matris dilimi; seviye 15 ise `None`.
    fn slice(self) -> Option<&'static [u8]> {
        if self.level >= 15 {
            return None;
        }
        let q = usize::from(self.level);
        let c = self.class.min(1);
        let base = match self.tx {
            0 => 0,
            1 => 4,
            2 => 9,
            3 | 4 | 11 | 12 => 13,
            5 => 3,
            6 => 1,
            7 => 8,
            8 => 5,
            9 => 12,
            10 => 10,
            13 => 7,
            14 => 2,
            15 => 11,
            16 => 6,
            // dav1d `qm.c:4699-4711`: tablo ataması **ters**tir (katsayılar
            // transpoze saklanır) ve 64'lük boyutlar 32'lik tabloyu devralır:
            //   RTX_16X64 = RTX_16X32 = qm_tbl_32x16,
            //   RTX_64X16 = RTX_32X16 = qm_tbl_16x32.
            // T30'a kadar 17/18 `10`/`12` kardeşleriyle aynı tabloya
            // düşüyordu; QM açık karelerde 16x64/64x16 katsayıları yanlış
            // ölçekleniyordu.
            17 => 12,
            18 => 10,
            _ => 13,
        };
        Some(match base {
            0 => &t::QM_4X4[q][c],
            1 => &t::QM_4X8[q][c],
            2 => &t::QM_4X16[q][c],
            3 => &t::QM_8X4[q][c],
            4 => &t::QM_8X8[q][c],
            5 => &t::QM_8X16[q][c],
            6 => &t::QM_8X32[q][c],
            7 => &t::QM_16X4[q][c],
            8 => &t::QM_16X8[q][c],
            9 => &t::QM_16X16[q][c],
            10 => &t::QM_16X32[q][c],
            11 => &t::QM_32X8[q][c],
            12 => &t::QM_32X16[q][c],
            _ => &t::QM_32X32[q][c],
        })
    }
}

/// `read_golomb` (`recon_tmpl.c:49-57`).
///
/// **Açık aritmetik (T30).** C gövdesi `unsigned val` üzerinde
/// `val = (val << 1) + bit` yazar (`recon_tmpl.c:53`). İlk döngü 32 sıfır bit
/// okuyabildiği için `val` 32 kez sola kaydırılır ve **işaretsiz 32-bit
/// sarması** zorunludur (`1u << 32` C'de 0'dır). Rust'ta `u32 << 1` taşan
/// biti sessizce atar (panik yok), fakat `+` hata ayıklamada **panikler**
/// (`attempt to add with overflow`). Bu yüzden toplama `wrapping_add`'tir;
/// sonuç `wrapping_sub(1)` C'nin `return val - 1` (işaretsiz) karşılığıdır.
/// Uç değer testleri: `tests::golomb_uc_degerler`.
fn read_golomb(msac: &mut MsacContext<'_>) -> u32 {
    let mut len = 0u32;
    let mut val = 1u32;
    while msac.decode_bool_equi() == 0 && len < 32 {
        len += 1;
    }
    while len > 0 {
        len -= 1;
        // dav1d `recon_tmpl.c:53`: işaretsiz 32-bit sarma.
        val = val.wrapping_shl(1).wrapping_add(msac.decode_bool_equi());
    }
    val.wrapping_sub(1)
}

/// `get_skip_ctx` (`recon_tmpl.c:59-138`) — SWAR sabit genişlikli yükleme.
#[inline(always)]
fn ctx_any_ne_0x40(s: &[u8], l: u8) -> usize {
    let ne = match l {
        0 => s[0] != 0x40,
        1 => {
            let s = &s[..2];
            u16::from_ne_bytes([s[0], s[1]]) != 0x4040
        }
        2 => {
            let s = &s[..4];
            u32::from_ne_bytes([s[0], s[1], s[2], s[3]]) != 0x4040_4040
        }
        _ => {
            let s = &s[..8];
            u64::from_ne_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
                != 0x4040_4040_4040_4040
        }
    };
    usize::from(ne)
}

#[inline(always)]
fn ctx_or_low6_clamped4(s: &[u8], l: u8) -> usize {
    let v: u32 = match l {
        0 => u32::from(s[0]),
        1 => {
            let s = &s[..2];
            let w = u32::from(u16::from_ne_bytes([s[0], s[1]]));
            w | (w >> 8)
        }
        2 => {
            let s = &s[..4];
            let d = u32::from_ne_bytes([s[0], s[1], s[2], s[3]]);
            let d = d | (d >> 16);
            d | (d >> 8)
        }
        3 => {
            let s = &s[..8];
            let q = u64::from_ne_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]);
            let d = (q as u32) | ((q >> 32) as u32);
            let d = d | (d >> 16);
            d | (d >> 8)
        }
        _ => {
            let s = &s[..16];
            let q0 = u64::from_ne_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]);
            let q1 = u64::from_ne_bytes([s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]]);
            let q = q0 | q1;
            let d = (q as u32) | ((q >> 32) as u32);
            let d = d | (d >> 16);
            d | (d >> 8)
        }
    };
    ((v & 0x3f) as usize).min(4)
}

#[inline(always)]
fn get_skip_ctx(
    tx: u8,
    bs: u8,
    a: &[u8],
    l: &[u8],
    chroma: bool,
    ss_hor: bool,
    ss_ver: bool,
) -> usize {
    let b_dim = &t::BLOCK_DIMENSIONS[bs as usize];
    let dim = &t::TXFM_DIMENSIONS[tx as usize];
    let lw = dim[2];
    let lh = dim[3];
    if chroma {
        let not_one_blk = u32::from(b_dim[2]) - u32::from(b_dim[2] != 0 && ss_hor) > u32::from(lw)
            || u32::from(b_dim[3]) - u32::from(b_dim[3] != 0 && ss_ver) > u32::from(lh);
        let ca = ctx_any_ne_0x40(a, lw);
        let cl = ctx_any_ne_0x40(l, lh);
        return 7 + usize::from(not_one_blk) * 3 + ca + cl;
    }
    if b_dim[2] == lw && b_dim[3] == lh {
        return 0;
    }
    let la = ctx_or_low6_clamped4(a, lw);
    let ll = ctx_or_low6_clamped4(l, lh);
    usize::from(t::SKIP_CTX[la][ll])
}

/// `get_dc_sign_ctx` (`recon_tmpl.c:140-296`) — SWAR üst-2-bit toplamı.
#[inline(always)]
fn sum_top2_bits(s: &[u8], l: u8) -> i32 {
    match l {
        0 => i32::from(s[0] >> 6),
        1 => {
            let s = &s[..2];
            i32::from(s[0] >> 6) + i32::from(s[1] >> 6)
        }
        2 => {
            let s = &s[..4];
            let v = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            ((((v & 0xC0C0_C0C0) >> 6).wrapping_mul(0x0101_0101)) >> 24) as i32
        }
        3 => {
            let s = &s[..8];
            let v = u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]);
            ((((v & 0xC0C0_C0C0_C0C0_C0C0) >> 6).wrapping_mul(0x0101_0101_0101_0101)) >> 56) as i32
        }
        _ => {
            let s = &s[..16];
            let v0 = u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]);
            let v1 = u64::from_le_bytes([s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]]);
            let t = ((v0 & 0xC0C0_C0C0_C0C0_C0C0) >> 6) + ((v1 & 0xC0C0_C0C0_C0C0_C0C0) >> 6);
            (t.wrapping_mul(0x0101_0101_0101_0101) >> 56) as i32
        }
    }
}

#[inline(always)]
fn get_dc_sign_ctx(tx: u8, a: &[u8], l: &[u8]) -> usize {
    let dim = &t::TXFM_DIMENSIONS[tx as usize];
    let total = sum_top2_bits(a, dim[2]) + sum_top2_bits(l, dim[3]);
    let s = total - i32::from(dim[0]) - i32::from(dim[1]);
    usize::from(s != 0) + usize::from(s > 0)
}

/// `get_lo_ctx` 2B sınıfı (`recon_tmpl.c:298-319`, `TX_CLASS_2D`), sabit `SLH` (`0..=3`).
#[inline(always)]
fn get_lo_ctx_2d_const<const SLH: usize>(
    levels: &[u8; 32 * 34 + 64],
    base: usize,
    x: usize,
    y: usize,
    lo_ctx_offsets: &[[u8; 5]; 5],
) -> (usize, u32) {
    let stride = 4usize << SLH;
    let base = base & 0x3ff;
    let mut mag = u32::from(levels[base + 1])
        + u32::from(levels[base + stride])
        + u32::from(levels[base + stride + 1]);
    let hi = mag;
    mag += u32::from(levels[base + 2]) + u32::from(levels[base + 2 * stride]);
    let offset = usize::from(lo_ctx_offsets[y.min(4)][x.min(4)]);
    let add = if mag > 512 {
        4
    } else {
        ((mag + 64) >> 7) as usize
    };
    (offset + add, hi)
}

/// 2B AC katsayı döngüsü (`recon_tmpl.c:486-527`, `TX_CLASS_2D`), sabit `SLH` (`0..=3`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn decode_ac_2d<const SLH: usize>(
    msac: &mut MsacContext<'_>,
    lo_cdf: &mut [[u16; 4]; 41],
    hi_cdf: &mut [[u16; 4]; 21],
    levels: &mut [u8; 32 * 34 + 64],
    cf: &mut [i32; 1024],
    scan: &[u16],
    lo_ctx_offsets: &[[u8; 5]; 5],
    eob_idx: usize,
    eob_tok: usize,
    #[allow(unused_variables)] eob: i32,
    #[allow(unused_variables)] lo_cdf_off: usize,
    #[allow(unused_variables)] hi_cdf_off: usize,
) -> usize {
    let shift = SLH + 2;
    let mask = (4usize << SLH) - 1;
    let scan = &scan[..=eob_idx];
    let mut rc = usize::from(scan[eob_idx]) & 0x3ff;
    let x = rc >> shift;
    let y = rc & mask;
    let mut tok = eob_tok + 1;
    let mut level_tok = tok * 0x41;
    if eob_tok == 2 {
        let hi_ctx = if x | y > 1 { 14 } else { 7 };
        #[cfg(feature = "msac-trace")]
        crate::trace::mark(
            "coef.hi_tok0",
            (hi_cdf_off + hi_ctx * 4) as i32,
            hi_ctx as i32,
            eob,
            0,
        );
        tok = msac.decode_hi_tok(&mut hi_cdf[hi_ctx]);
        level_tok = tok + (3 << 6);
    }
    cf[rc] = (tok << 11) as i32;
    levels[rc] = level_tok as u8;

    #[cfg(not(feature = "msac-trace"))]
    {
        #[allow(clippy::too_many_arguments)]
        #[inline(always)]
        fn run_ac_2d_loop<const SLH: usize>(
            msac: &mut MsacContext<'_>,
            lo_cdf: &mut [[u16; 4]; 41],
            hi_cdf: &mut [[u16; 4]; 21],
            levels: &mut [u8; 32 * 34 + 64],
            cf: &mut [i32; 1024],
            scan_rev: &[u16],
            lo_ctx_offsets: &[[u8; 5]; 5],
            mut rc: usize,
            mut dec_lo: impl FnMut(&mut MsacContext<'_>, &mut [u16; 4]) -> usize,
            mut dec_hi: impl FnMut(&mut MsacContext<'_>, &mut [u16; 4]) -> usize,
        ) -> usize {
            let shift = SLH + 2;
            let mask = (4usize << SLH) - 1;
            for &scan_rc in scan_rev.iter().rev() {
                let rc_i = usize::from(scan_rc) & 0x3ff;
                let x = rc_i >> shift;
                let mut y = rc_i & mask;
                let (lo_ctx, mag) = get_lo_ctx_2d_const::<SLH>(levels, rc_i, x, y, lo_ctx_offsets);
                y |= x;
                let mut tok = dec_lo(msac, &mut lo_cdf[lo_ctx]);
                if tok == 3 {
                    let mag = mag & 63;
                    let base_ctx = if y > 1 { 14 } else { 7 };
                    let hi_add = (if mag > 12 { 6 } else { (mag + 1) >> 1 }) as usize;
                    let hi_ctx = base_ctx + hi_add;
                    tok = dec_hi(msac, &mut hi_cdf[hi_ctx]);
                    levels[rc_i] = (tok + (3 << 6)) as u8;
                    cf[rc_i] = ((tok << 11) | rc) as i32;
                    rc = rc_i;
                } else {
                    let t = (tok as u32) * 0x17ff41;
                    levels[rc_i] = t as u8;
                    let v = (t >> 9) & (rc as u32).wrapping_add(!0x7ffu32);
                    if tok != 0 {
                        rc = rc_i;
                    }
                    cf[rc_i] = v as i32;
                }
            }
            rc
        }

        let scan_rev = &scan[1..eob_idx];
        #[cfg(target_arch = "x86_64")]
        {
            use crate::entropy::msac::MsacIsa;
            rc = match msac.isa() {
                MsacIsa::Sse2(t) | MsacIsa::Avx2(t, _) => run_ac_2d_loop::<SLH>(
                    msac,
                    lo_cdf,
                    hi_cdf,
                    levels,
                    cf,
                    scan_rev,
                    lo_ctx_offsets,
                    rc,
                    |m, c| crate::dsp::x86::msac::decode_symbol_adapt4_sse2(t, m, c, 3),
                    |m, c| crate::dsp::x86::msac::decode_hi_tok_sse2(t, m, c),
                ),
                MsacIsa::Scalar => run_ac_2d_loop::<SLH>(
                    msac,
                    lo_cdf,
                    hi_cdf,
                    levels,
                    cf,
                    scan_rev,
                    lo_ctx_offsets,
                    rc,
                    |m, c| m.decode_symbol_adapt_scalar(c, 3),
                    |m, c| m.decode_hi_tok_scalar(c),
                ),
            };
        }
        #[cfg(target_arch = "aarch64")]
        {
            use crate::entropy::msac::MsacIsa;
            rc = match msac.isa() {
                MsacIsa::Neon(t) => run_ac_2d_loop::<SLH>(
                    msac,
                    lo_cdf,
                    hi_cdf,
                    levels,
                    cf,
                    scan_rev,
                    lo_ctx_offsets,
                    rc,
                    |m, c| crate::dsp::arm::giris::msac_decode_symbol_adapt4(t, m, c, 3),
                    |m, c| crate::dsp::arm::giris::msac_decode_hi_tok(t, m, c),
                ),
                MsacIsa::Scalar => run_ac_2d_loop::<SLH>(
                    msac,
                    lo_cdf,
                    hi_cdf,
                    levels,
                    cf,
                    scan_rev,
                    lo_ctx_offsets,
                    rc,
                    |m, c| m.decode_symbol_adapt_scalar(c, 3),
                    |m, c| m.decode_hi_tok_scalar(c),
                ),
            };
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            rc = run_ac_2d_loop::<SLH>(
                msac,
                lo_cdf,
                hi_cdf,
                levels,
                cf,
                scan_rev,
                lo_ctx_offsets,
                rc,
                |m, c| m.decode_symbol_adapt_scalar(c, 3),
                |m, c| m.decode_hi_tok_scalar(c),
            );
        }
    }

    #[cfg(feature = "msac-trace")]
    for i in (1..eob_idx).rev() {
        let rc_i = usize::from(scan[i]) & 0x3ff;
        let x = rc_i >> shift;
        let mut y = rc_i & mask;
        let (lo_ctx, mag) = get_lo_ctx_2d_const::<SLH>(levels, rc_i, x, y, lo_ctx_offsets);
        y |= x;
        crate::trace::mark(
            "coef.lo_tok",
            (lo_cdf_off + lo_ctx * 4) as i32,
            lo_ctx as i32,
            i as i32,
            0,
        );
        tok = msac.decode_symbol_adapt4(&mut lo_cdf[lo_ctx], 3);
        if tok == 3 {
            let mag = mag & 63;
            let base_ctx = if y > 1 { 14 } else { 7 };
            let hi_add = (if mag > 12 { 6 } else { (mag + 1) >> 1 }) as usize;
            let hi_ctx = base_ctx + hi_add;
            crate::trace::mark(
                "coef.hi_tok",
                (hi_cdf_off + hi_ctx * 4) as i32,
                hi_ctx as i32,
                i as i32,
                mag as i32,
            );
            tok = msac.decode_hi_tok(&mut hi_cdf[hi_ctx]);
            levels[rc_i] = (tok + (3 << 6)) as u8;
            cf[rc_i] = ((tok << 11) | rc) as i32;
            rc = rc_i;
        } else {
            let t = (tok as u32) * 0x17ff41;
            levels[rc_i] = t as u8;
            let v = (t >> 9) & (rc as u32).wrapping_add(!0x7ffu32);
            if tok != 0 {
                rc = rc_i;
            }
            cf[rc_i] = v as i32;
        }
    }
    rc
}

/// `get_lo_ctx` 1B sınıfı (`recon_tmpl.c:298-319`, `TX_CLASS_H` / `TX_CLASS_V`, `stride == 16`).
#[inline(always)]
fn get_lo_ctx_1d(levels: &[u8; 32 * 34 + 64], base: usize, y: usize) -> (usize, u32) {
    let base = base & 0x3ff;
    let mut mag =
        u32::from(levels[base + 1]) + u32::from(levels[base + 16]) + u32::from(levels[base + 2]);
    let hi = mag;
    mag += u32::from(levels[base + 3]) + u32::from(levels[base + 4]);
    let offset = 26 + if y > 1 { 10 } else { y * 5 };
    let add = if mag > 512 {
        4
    } else {
        ((mag + 64) >> 7) as usize
    };
    (offset + add, hi)
}

/// Tüm büyük kare/dikdörtgen tarama tabloları (14 benzersiz).
fn scan_table(index: usize) -> &'static [u16] {
    match index {
        0 => &t::SCAN_4X4,
        1 => &t::SCAN_4X8,
        2 => &t::SCAN_4X16,
        3 => &t::SCAN_8X4,
        4 => &t::SCAN_8X8,
        5 => &t::SCAN_8X16,
        6 => &t::SCAN_8X32,
        7 => &t::SCAN_16X4,
        8 => &t::SCAN_16X8,
        9 => &t::SCAN_16X16,
        10 => &t::SCAN_16X32,
        11 => &t::SCAN_32X8,
        12 => &t::SCAN_32X16,
        _ => &t::SCAN_32X32,
    }
}

/// `get_uv_inter_txtp` (`src/env.h:120-133`): inter blokta kroma tx türü
/// luma tx türünden türetilir.
fn uv_inter_txtp(uv_dim: &[u8; 8], ytxtp: u8) -> u8 {
    if uv_dim[5] == 3 {
        // `uvt_dim->max == TX_32X32`: yalnız IDTX korunur.
        return if ytxtp == 9 { 9 } else { 0 };
    }
    // `uvt_dim->min == TX_16X16` ve ytxtp ∈ {H_FLIPADST, V_FLIPADST, H_ADST,
    // V_ADST} (12..=15) ise DCT_DCT.
    if uv_dim[4] == 2 && ytxtp >= 12 {
        return 0;
    }
    ytxtp
}

/// Katsayı bloğunu çözer (dav1d `decode_coefs`, `recon_tmpl.c:321-728`).
///
/// `a`/`l`, `lcoef`/`ccoef` bağlam dizilerinin blok konumundan başlayan
/// dilimidir. `cf` 1024 hücrelidir; yalnız çözülen bölge yazılır.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    unused_assignments
)]
pub(crate) fn decode_coefs(
    msac: &mut MsacContext<'_>,
    cdf: &mut CdfContext,
    env: CoefEnv<'_>,
    tx: u8,
    bs: u8,
    y_mode_nofilt: u8,
    uv_mode: u8,
    plane: usize,
    a: &[u8],
    l: &[u8],
    cf: &mut [i32],
    levels: &mut [u8],
) -> CoefResult {
    let chroma = plane != 0;
    let dim = &t::TXFM_DIMENSIONS[tx as usize];
    let ctx = usize::from(dim[7]);
    let (coef, mode) = cdf.cells_mut().split_at_mut(layout::COEF_CONTEXT_CELLS);

    let sctx = get_skip_ctx(tx, bs, a, l, chroma, env.ss_hor, env.ss_ver);
    #[cfg(feature = "msac-trace")]
    crate::trace::mark(
        "coef.skip",
        i32::from(tx),
        i32::from(bs),
        ctx as i32,
        sctx as i32,
    );
    let all_skip = msac.decode_bool_adapt(&mut coef[layout::COEF_skip + (ctx * 13 + sctx) * 2..]);
    if all_skip != 0 {
        return CoefResult {
            eob: -1,
            txtp: if env.lossless { 16 } else { 0 },
            res_ctx: 0x40,
        };
    }

    let txtp: u8 = if env.lossless {
        16
    } else if u32::from(dim[5]) + u32::from(!env.inter) >= 4 {
        0
    } else if chroma {
        if env.inter {
            uv_inter_txtp(dim, env.txtp_hint)
        } else {
            t::TXTP_FROM_UVMODE[usize::from(uv_mode)]
        }
    } else if env.seg_qidx == 0 {
        0
    } else if env.inter {
        if env.reduced_txtp || dim[5] == 3 {
            let idx = msac
                .decode_bool_adapt(&mut mode[layout::MODE_txtp_inter3 + usize::from(dim[4]) * 2..]);
            if idx != 0 { 0 } else { 9 }
        } else if dim[4] == 2 {
            let idx = msac.decode_symbol_adapt16(&mut mode[layout::MODE_txtp_inter2..], 11);
            t::TX_TYPES_PER_SET[idx + 12]
        } else {
            let idx = msac.decode_symbol_adapt16(
                &mut mode[layout::MODE_txtp_inter1 + usize::from(dim[4]) * 16..],
                15,
            );
            t::TX_TYPES_PER_SET[idx + 24]
        }
    } else if env.reduced_txtp || dim[4] == 2 {
        let idx = msac.decode_symbol_adapt8(
            &mut mode[layout::MODE_txtp_intra2
                + (usize::from(dim[4]) * 13 + usize::from(y_mode_nofilt)) * 8..],
            4,
        );
        t::TX_TYPES_PER_SET[idx]
    } else {
        let idx = msac.decode_symbol_adapt8(
            &mut mode[layout::MODE_txtp_intra1
                + (usize::from(dim[4]) * 13 + usize::from(y_mode_nofilt)) * 8..],
            6,
        );
        t::TX_TYPES_PER_SET[idx + 5]
    };

    let qm = env
        .qm
        .and_then(|sel| if txtp < 9 { sel.slice() } else { None });
    let slw = usize::from(dim[2].min(3));
    let slh = usize::from(dim[3].min(3));
    let tx2dszctx = slw + slh;
    let tx_class = t::TX_TYPE_CLASS[usize::from(txtp)];
    let lo_ctx_idx = if tx < 5 { 0 } else { 1 + (usize::from(tx) & 1) };
    let is_1d = usize::from(tx_class != 0);
    let eob_bin_off = match tx2dszctx {
        0 => layout::COEF_eob_bin_16 + (usize::from(chroma) * 2 + is_1d) * 8,
        1 => layout::COEF_eob_bin_32 + (usize::from(chroma) * 2 + is_1d) * 8,
        2 => layout::COEF_eob_bin_64 + (usize::from(chroma) * 2 + is_1d) * 8,
        3 => layout::COEF_eob_bin_128 + (usize::from(chroma) * 2 + is_1d) * 8,
        4 => layout::COEF_eob_bin_256 + (usize::from(chroma) * 2 + is_1d) * 16,
        5 => layout::COEF_eob_bin_512 + usize::from(chroma) * 16,
        _ => layout::COEF_eob_bin_1024 + usize::from(chroma) * 16,
    };
    #[cfg(feature = "msac-trace")]
    crate::trace::mark(
        "coef.eob_bin",
        eob_bin_off as i32,
        (4 + tx2dszctx) as i32,
        i32::from(chroma),
        i32::from(txtp),
    );
    let mut eob = if tx2dszctx <= 3 {
        msac.decode_symbol_adapt8(&mut coef[eob_bin_off..], 4 + tx2dszctx) as i32
    } else {
        msac.decode_symbol_adapt16(&mut coef[eob_bin_off..], 4 + tx2dszctx) as i32
    };
    if eob > 1 {
        let eob_bin = (eob - 2) as usize;
        let hi_off = layout::COEF_eob_hi_bit + ((ctx * 2 + usize::from(chroma)) * 9 + eob_bin) * 2;
        #[cfg(feature = "msac-trace")]
        crate::trace::mark("coef.eob_hi", hi_off as i32, eob, 0, 0);
        let hi = msac.decode_bool_adapt(&mut coef[hi_off..]);
        eob = (((hi | 2) << eob_bin) | msac.decode_bools(eob_bin as u32)) as i32;
    }

    let eob_cdf_off = layout::COEF_eob_base_tok + (ctx * 2 + usize::from(chroma)) * 4 * 4;
    let hi_cdf_off = layout::COEF_br_tok + (ctx.min(3) * 2 + usize::from(chroma)) * 21 * 4;
    let lo_cdf_off = layout::COEF_base_tok + (ctx * 2 + usize::from(chroma)) * 41 * 4;
    let mut rc = 0usize;
    let dc_tok;

    if eob > 0 {
        let levels: &mut [u8; 32 * 34 + 64] = (&mut levels[..32 * 34 + 64])
            .try_into()
            .expect("levels 1152");
        let cf: &mut [i32; 1024] = (&mut cf[..1024]).try_into().expect("cf 1024");
        let eob_idx = eob as usize;
        let eob_ctx = 1 + usize::from(eob > (2 << tx2dszctx)) + usize::from(eob > (4 << tx2dszctx));
        #[cfg(feature = "msac-trace")]
        crate::trace::mark(
            "coef.eob_tok",
            (eob_cdf_off + eob_ctx * 4) as i32,
            eob_ctx as i32,
            eob,
            0,
        );
        let eob_tok = msac.decode_symbol_adapt4(&mut coef[eob_cdf_off + eob_ctx * 4..], 2);
        let stride = if tx_class == 0 { 4usize << slh } else { 16 };
        let shift2 = slh + 2;
        let hi_rel_off = (ctx.min(3) * 2 + usize::from(chroma)) * 21 * 4;
        let (lo_part, hi_part) = coef.split_at_mut(layout::COEF_br_tok);
        let lo_cdf: &mut [[u16; 4]; 41] = lo_part[lo_cdf_off..lo_cdf_off + 41 * 4]
            .as_chunks_mut::<4>()
            .0
            .try_into()
            .expect("lo_cdf 41");
        let hi_cdf: &mut [[u16; 4]; 21] = hi_part[hi_rel_off..hi_rel_off + 21 * 4]
            .as_chunks_mut::<4>()
            .0
            .try_into()
            .expect("hi_cdf 21");

        if tx_class == 0 {
            levels[..stride * ((4usize << slw) + 2)].fill(0);
            let scan = scan_table(usize::from(t::SCAN_INDEX[usize::from(tx)]));
            let lo_ctx_offsets = &t::LO_CTX_OFFSETS[lo_ctx_idx];
            rc = match slh {
                0 => decode_ac_2d::<0>(
                    msac,
                    lo_cdf,
                    hi_cdf,
                    levels,
                    cf,
                    scan,
                    lo_ctx_offsets,
                    eob_idx,
                    eob_tok,
                    eob,
                    lo_cdf_off,
                    hi_cdf_off,
                ),
                1 => decode_ac_2d::<1>(
                    msac,
                    lo_cdf,
                    hi_cdf,
                    levels,
                    cf,
                    scan,
                    lo_ctx_offsets,
                    eob_idx,
                    eob_tok,
                    eob,
                    lo_cdf_off,
                    hi_cdf_off,
                ),
                2 => decode_ac_2d::<2>(
                    msac,
                    lo_cdf,
                    hi_cdf,
                    levels,
                    cf,
                    scan,
                    lo_ctx_offsets,
                    eob_idx,
                    eob_tok,
                    eob,
                    lo_cdf_off,
                    hi_cdf_off,
                ),
                _ => decode_ac_2d::<3>(
                    msac,
                    lo_cdf,
                    hi_cdf,
                    levels,
                    cf,
                    scan,
                    lo_ctx_offsets,
                    eob_idx,
                    eob_tok,
                    eob,
                    lo_cdf_off,
                    hi_cdf_off,
                ),
            };
        } else {
            let mut tok = eob_tok + 1;
            let mut level_tok = tok * 0x41;
            let is_h = tx_class == 1;
            let sl_1d = if is_h { slh } else { slw };
            levels[..16 * ((4usize << sl_1d) + 2)].fill(0);
            let mask_1d = (4usize << sl_1d) - 1;
            let shift_1d = sl_1d + 2;
            let x = eob_idx & mask_1d;
            let y = eob_idx >> shift_1d;
            rc = (if is_h { eob_idx } else { (x << shift2) | y }) & 0x3ff;
            let base = (x * 16 + y) & 0x3ff;
            if eob_tok == 2 {
                let hi_ctx = if y != 0 { 14 } else { 7 };
                #[cfg(feature = "msac-trace")]
                crate::trace::mark(
                    "coef.hi_tok0",
                    (hi_cdf_off + hi_ctx * 4) as i32,
                    hi_ctx as i32,
                    eob,
                    0,
                );
                tok = msac.decode_hi_tok(&mut hi_cdf[hi_ctx]);
                level_tok = tok + (3 << 6);
            }
            cf[rc] = (tok << 11) as i32;
            levels[base] = level_tok as u8;

            #[cfg(not(feature = "msac-trace"))]
            {
                #[allow(clippy::too_many_arguments)]
                #[inline(always)]
                fn run_ac_1d_loop<const IS_H: bool>(
                    msac: &mut MsacContext<'_>,
                    lo_cdf: &mut [[u16; 4]; 41],
                    hi_cdf: &mut [[u16; 4]; 21],
                    levels: &mut [u8; 32 * 34 + 64],
                    cf: &mut [i32; 1024],
                    eob_idx: usize,
                    mask_1d: usize,
                    shift_1d: usize,
                    shift2: usize,
                    mut rc: usize,
                    mut dec_lo: impl FnMut(&mut MsacContext<'_>, &mut [u16; 4]) -> usize,
                    mut dec_hi: impl FnMut(&mut MsacContext<'_>, &mut [u16; 4]) -> usize,
                ) -> usize {
                    for i in (1..eob_idx).rev() {
                        let x = i & mask_1d;
                        let y = i >> shift_1d;
                        let rc_i = (if IS_H { i } else { (x << shift2) | y }) & 0x3ff;
                        let base = (x * 16 + y) & 0x3ff;
                        let (lo_ctx, mag) = get_lo_ctx_1d(levels, base, y);
                        let mut tok = dec_lo(msac, &mut lo_cdf[lo_ctx]);
                        if tok == 3 {
                            let mag = mag & 63;
                            let base_ctx = if y > 0 { 14 } else { 7 };
                            let hi_add = (if mag > 12 { 6 } else { (mag + 1) >> 1 }) as usize;
                            let hi_ctx = base_ctx + hi_add;
                            tok = dec_hi(msac, &mut hi_cdf[hi_ctx]);
                            levels[base] = (tok + (3 << 6)) as u8;
                            cf[rc_i] = ((tok << 11) | rc) as i32;
                            rc = rc_i;
                        } else {
                            let t = (tok as u32) * 0x17ff41;
                            levels[base] = t as u8;
                            let v = (t >> 9) & (rc as u32).wrapping_add(!0x7ffu32);
                            if tok != 0 {
                                rc = rc_i;
                            }
                            cf[rc_i] = v as i32;
                        }
                    }
                    rc
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use crate::entropy::msac::MsacIsa;
                    rc = match (is_h, msac.isa()) {
                        (true, MsacIsa::Sse2(t) | MsacIsa::Avx2(t, _)) => run_ac_1d_loop::<true>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| crate::dsp::x86::msac::decode_symbol_adapt4_sse2(t, m, c, 3),
                            |m, c| crate::dsp::x86::msac::decode_hi_tok_sse2(t, m, c),
                        ),
                        (false, MsacIsa::Sse2(t) | MsacIsa::Avx2(t, _)) => run_ac_1d_loop::<false>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| crate::dsp::x86::msac::decode_symbol_adapt4_sse2(t, m, c, 3),
                            |m, c| crate::dsp::x86::msac::decode_hi_tok_sse2(t, m, c),
                        ),
                        (true, MsacIsa::Scalar) => run_ac_1d_loop::<true>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| m.decode_symbol_adapt_scalar(c, 3),
                            |m, c| m.decode_hi_tok_scalar(c),
                        ),
                        (false, MsacIsa::Scalar) => run_ac_1d_loop::<false>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| m.decode_symbol_adapt_scalar(c, 3),
                            |m, c| m.decode_hi_tok_scalar(c),
                        ),
                    };
                }
                #[cfg(target_arch = "aarch64")]
                {
                    use crate::entropy::msac::MsacIsa;
                    rc = match (is_h, msac.isa()) {
                        (true, MsacIsa::Neon(t)) => run_ac_1d_loop::<true>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| crate::dsp::arm::giris::msac_decode_symbol_adapt4(t, m, c, 3),
                            |m, c| crate::dsp::arm::giris::msac_decode_hi_tok(t, m, c),
                        ),
                        (false, MsacIsa::Neon(t)) => run_ac_1d_loop::<false>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| crate::dsp::arm::giris::msac_decode_symbol_adapt4(t, m, c, 3),
                            |m, c| crate::dsp::arm::giris::msac_decode_hi_tok(t, m, c),
                        ),
                        (true, MsacIsa::Scalar) => run_ac_1d_loop::<true>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| m.decode_symbol_adapt_scalar(c, 3),
                            |m, c| m.decode_hi_tok_scalar(c),
                        ),
                        (false, MsacIsa::Scalar) => run_ac_1d_loop::<false>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| m.decode_symbol_adapt_scalar(c, 3),
                            |m, c| m.decode_hi_tok_scalar(c),
                        ),
                    };
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    rc = if is_h {
                        run_ac_1d_loop::<true>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| m.decode_symbol_adapt_scalar(c, 3),
                            |m, c| m.decode_hi_tok_scalar(c),
                        )
                    } else {
                        run_ac_1d_loop::<false>(
                            msac,
                            lo_cdf,
                            hi_cdf,
                            levels,
                            cf,
                            eob_idx,
                            mask_1d,
                            shift_1d,
                            shift2,
                            rc,
                            |m, c| m.decode_symbol_adapt_scalar(c, 3),
                            |m, c| m.decode_hi_tok_scalar(c),
                        )
                    };
                }
            }
            #[cfg(feature = "msac-trace")]
            for i in (1..eob_idx).rev() {
                let x = i & mask_1d;
                let y = i >> shift_1d;
                let rc_i = (if is_h { i } else { (x << shift2) | y }) & 0x3ff;
                let base = (x * 16 + y) & 0x3ff;
                let (lo_ctx, mag) = get_lo_ctx_1d(levels, base, y);
                crate::trace::mark(
                    "coef.lo_tok",
                    (lo_cdf_off + lo_ctx * 4) as i32,
                    lo_ctx as i32,
                    i as i32,
                    i32::from(tx_class),
                );
                tok = msac.decode_symbol_adapt4(&mut lo_cdf[lo_ctx], 3);
                if tok == 3 {
                    let mag = mag & 63;
                    let base_ctx = if y > 0 { 14 } else { 7 };
                    let hi_add = (if mag > 12 { 6 } else { (mag + 1) >> 1 }) as usize;
                    let hi_ctx = base_ctx + hi_add;
                    crate::trace::mark(
                        "coef.hi_tok",
                        (hi_cdf_off + hi_ctx * 4) as i32,
                        hi_ctx as i32,
                        i as i32,
                        mag as i32,
                    );
                    tok = msac.decode_hi_tok(&mut hi_cdf[hi_ctx]);
                    levels[base] = (tok + (3 << 6)) as u8;
                    cf[rc_i] = ((tok << 11) | rc) as i32;
                    rc = rc_i;
                } else {
                    let t = (tok as u32) * 0x17ff41;
                    levels[base] = t as u8;
                    let v = (t >> 9) & (rc as u32).wrapping_add(!0x7ffu32);
                    if tok != 0 {
                        rc = rc_i;
                    }
                    cf[rc_i] = v as i32;
                }
            }
        }
        let dc_ctx = if tx_class == 0 {
            0
        } else {
            get_lo_ctx_1d(levels, 0, 0).0
        };
        #[cfg(feature = "msac-trace")]
        crate::trace::mark(
            "coef.dc_tok",
            (lo_cdf_off + dc_ctx * 4) as i32,
            dc_ctx as i32,
            eob,
            i32::from(tx_class),
        );
        dc_tok = msac.decode_symbol_adapt4(&mut lo_cdf[dc_ctx], 3);
        let mut dc_tok = dc_tok;
        if dc_tok == 3 {
            let mag = if tx_class == 0 {
                (u32::from(levels[1]) + u32::from(levels[stride]) + u32::from(levels[stride + 1]))
                    & 63
            } else {
                get_lo_ctx_1d(levels, 0, 0).1 & 63
            };
            let hi_ctx = if mag > 12 { 6 } else { (mag as usize + 1) >> 1 };
            #[cfg(feature = "msac-trace")]
            crate::trace::mark(
                "coef.dc_hi_tok",
                (hi_cdf_off + hi_ctx * 4) as i32,
                hi_ctx as i32,
                eob,
                0,
            );
            dc_tok = msac.decode_hi_tok(&mut hi_cdf[hi_ctx]);
        }
        // dc işareti (C: :600-607)
        let (dc_sign, dc_sign_level) = if dc_tok == 0 {
            (0u32, 1u32 << 6)
        } else {
            let off = dc_sign_off(chroma, tx, a, l);
            #[cfg(feature = "msac-trace")]
            crate::trace::mark(
                "coef.dc_sign",
                off as i32,
                i32::from(chroma),
                i32::from(tx),
                dc_tok as i32,
            );
            let sign = msac.decode_bool_adapt(&mut coef[off..]);
            (sign, sign.wrapping_sub(1) & (2 << 6))
        };
        return dequant_block(
            msac,
            env,
            plane,
            cf,
            dq_shift_of(ctx),
            cf_max_of(env),
            eob,
            txtp,
            rc,
            dc_tok,
            dc_sign,
            dc_sign_level,
            qm,
        );
    }

    // dc-only (`eob == 0`: `levels` hiç okunmaz ve sıfırlanmaz)
    #[cfg(feature = "msac-trace")]
    crate::trace::mark("coef.dc_only", eob_cdf_off as i32, 0, 0, 0);
    let tok_br = msac.decode_symbol_adapt4(&mut coef[eob_cdf_off..], 2);
    let mut dc_tok = 1 + tok_br;
    if tok_br == 2 {
        #[cfg(feature = "msac-trace")]
        crate::trace::mark("coef.dc_only_hi", hi_cdf_off as i32, 0, 0, 0);
        dc_tok = msac.decode_hi_tok(&mut coef[hi_cdf_off..]);
    }
    let (dc_sign, dc_sign_level) = if dc_tok == 0 {
        (0u32, 1u32 << 6)
    } else {
        let off = dc_sign_off(chroma, tx, a, l);
        #[cfg(feature = "msac-trace")]
        crate::trace::mark(
            "coef.dc_sign",
            off as i32,
            i32::from(chroma),
            i32::from(tx),
            dc_tok as i32,
        );
        let sign = msac.decode_bool_adapt(&mut coef[off..]);
        (sign, sign.wrapping_sub(1) & (2 << 6))
    };
    dequant_block(
        msac,
        env,
        plane,
        cf,
        dq_shift_of(ctx),
        cf_max_of(env),
        eob,
        txtp,
        0,
        dc_tok,
        dc_sign,
        dc_sign_level,
        qm,
    )
}

#[inline(always)]
fn dq_shift_of(ctx: usize) -> u32 {
    ctx.saturating_sub(2) as u32
}

#[inline(always)]
fn cf_max_of(env: CoefEnv<'_>) -> u32 {
    let bpc = env.bit_depth.bits();
    !(!127u32 << if bpc == 8 { 8 } else { bpc })
}

#[inline(always)]
fn dc_sign_off(chroma: bool, tx: u8, a: &[u8], l: &[u8]) -> usize {
    let ctx = get_dc_sign_ctx(tx, a, l);
    layout::COEF_dc_sign + (usize::from(chroma) * 3 + ctx) * 2
}

/// Dequant + işaret + `cul_level` — dav1d `recon_tmpl.c:596-728`.
///
/// ## Açık aritmetik (T30)
///
/// dav1d burada **işaretsiz 32-bit sarmasına** dayanır; Rust karşılıkları
/// bilinçli olarak `wrapping_*`'tir. Her satırın kaynağı:
///
/// | Bu işlevdeki işlem | dav1d satırı | C türü | Seçilen işlem |
/// |---|---|---|---|
/// | `dq_tbl[0]`, `dq_tbl[1]` okuma | `:616`, `:645` | `uint16_t` | `u32::from` (daraltma yok) |
/// | qm çarpımı `dq*qm+16 >> 5` | `:620`, `:636` | `int` | düz `*`/`+` — aralık: `dq ≤ 65535`, `qm ≤ 255` → `16 711 441 < 2^32` |
/// | `read_golomb() + 15` | `:623`, `:650`, `:673`, `:700` | `unsigned` | **`wrapping_add`** (golomb 0xFFFF_FFFF dönebilir) |
/// | `& 0xfffff` | `:628`, `:678` | `unsigned` | düz maske |
/// | `(dq * tok) & 0xffffff` | `:629`, `:656`, `:679`, `:709` | `unsigned`/`int` | **`wrapping_mul`** — aralık `522 239 × 1 048 575 ≈ 5,5e11 > 2^32` |
/// | `dq_sat = umin(dq, cf_max + sign)` | `:636`, `:664`, `:680`, `:710` | `unsigned` | düz `+` — aralık: `cf_max ≤ 0x7f_ffff`, `sign ≤ 1` |
/// | `cul_level += tok` | `:662`, `:717` | `unsigned` | düz `+` — aralık: en çok 1 024 × `0xfffff ≈ 1,07e9 < 2^32` |
/// | `dq min cf_max` sonrası `cf[..]` | `:647`, `:671` | `(coef)` daraltma | `dq ≤ cf_max+1`; 8-bit'te `cf_max = 32767` (`coef = int16_t`) sınırı yorumla belgelenir |
/// | `res_ctx = umin(cul_level,63) \| dc_sign_level` | `:726` | `unsigned` | `min(63)` |
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dequant_block(
    msac: &mut MsacContext<'_>,
    env: CoefEnv<'_>,
    plane: usize,
    cf: &mut [i32],
    dq_shift: u32,
    cf_max: u32,
    eob: i32,
    txtp: u8,
    rc_start: usize,
    dc_tok: usize,
    dc_sign: u32,
    dc_sign_level: u32,
    qm: Option<&[u8]>,
) -> CoefResult {
    let cf: &mut [i32; 1024] = (&mut cf[..1024]).try_into().expect("cf 1024");
    let dq_shift = dq_shift & 3;
    // `recon_tmpl.c:616,645`: `dq_tbl` `uint16_t`; u32'ye genişletme kayıpsız.
    let (dc_dq, ac_dq) = (u32::from(env.dq[plane][0]), u32::from(env.dq[plane][1]));
    let mut cul_level: u32;
    let mut rc = rc_start & 0x3ff;
    // `recon_tmpl.c:636,664,680,710`: `cf_max + sign` — taşma yok
    // (`cf_max ≤ 0x7f_ffff`, `sign ∈ {0,1}`).
    let cf_max_signed = cf_max.wrapping_add(dc_sign);
    match qm {
        None => {
            if dc_tok == 0 {
                cul_level = 0;
            } else {
                let dq = if dc_tok == 15 {
                    let tok = read_golomb(msac).wrapping_add(15) & 0xfffff;
                    cul_level = tok;
                    ((dc_dq.wrapping_mul(tok) & 0xffffff) >> dq_shift).min(cf_max_signed)
                } else {
                    cul_level = dc_tok as u32;
                    ((dc_dq * dc_tok as u32) >> dq_shift).min(cf_max_signed)
                };
                cf[0] = if dc_sign != 0 {
                    -(dq as i32)
                } else {
                    dq as i32
                };
            }
            #[cfg(not(feature = "msac-trace"))]
            {
                #[inline(always)]
                fn run_dequant_no_qm(
                    msac: &mut MsacContext<'_>,
                    cf: &mut [i32; 1024],
                    mut rc: usize,
                    ac_dq: u32,
                    dq_shift: u32,
                    cf_max: u32,
                    mut cul_level: u32,
                    mut dec_equi: impl FnMut(&mut MsacContext<'_>) -> u32,
                ) -> u32 {
                    while rc != 0 {
                        let sign = dec_equi(msac);
                        let rc_tok = cf[rc] as u32;
                        let dq = if rc_tok >= (15 << 11) {
                            let tok = read_golomb(msac).wrapping_add(15) & 0xfffff;
                            cul_level += tok;
                            ((ac_dq.wrapping_mul(tok) & 0xffffff) >> dq_shift)
                                .min(cf_max.wrapping_add(sign))
                        } else {
                            let tok = rc_tok >> 11;
                            cul_level += tok;
                            (ac_dq * tok) >> dq_shift
                        };
                        cf[rc] = if sign != 0 { -(dq as i32) } else { dq as i32 };
                        rc = (rc_tok & 0x3ff) as usize;
                    }
                    cul_level
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use crate::entropy::msac::MsacIsa;
                    cul_level = match msac.isa() {
                        MsacIsa::Sse2(t) | MsacIsa::Avx2(t, _) => run_dequant_no_qm(
                            msac,
                            cf,
                            rc,
                            ac_dq,
                            dq_shift,
                            cf_max,
                            cul_level,
                            |m| crate::dsp::x86::msac::decode_bool_equi_sse2(t, m),
                        ),
                        MsacIsa::Scalar => run_dequant_no_qm(
                            msac,
                            cf,
                            rc,
                            ac_dq,
                            dq_shift,
                            cf_max,
                            cul_level,
                            |m| m.decode_bool_equi_scalar(),
                        ),
                    };
                }
                #[cfg(target_arch = "aarch64")]
                {
                    use crate::entropy::msac::MsacIsa;
                    cul_level = match msac.isa() {
                        MsacIsa::Neon(t) => run_dequant_no_qm(
                            msac,
                            cf,
                            rc,
                            ac_dq,
                            dq_shift,
                            cf_max,
                            cul_level,
                            |m| crate::dsp::arm::giris::msac_decode_bool_equi(t, m),
                        ),
                        MsacIsa::Scalar => run_dequant_no_qm(
                            msac,
                            cf,
                            rc,
                            ac_dq,
                            dq_shift,
                            cf_max,
                            cul_level,
                            |m| m.decode_bool_equi_scalar(),
                        ),
                    };
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    cul_level =
                        run_dequant_no_qm(msac, cf, rc, ac_dq, dq_shift, cf_max, cul_level, |m| {
                            m.decode_bool_equi_scalar()
                        });
                }
            }
            #[cfg(feature = "msac-trace")]
            while rc != 0 {
                let sign = msac.decode_bool_equi();
                let rc_tok = cf[rc] as u32;
                let dq = if rc_tok >= (15 << 11) {
                    let tok = read_golomb(msac).wrapping_add(15) & 0xfffff;
                    cul_level += tok;
                    ((ac_dq.wrapping_mul(tok) & 0xffffff) >> dq_shift)
                        .min(cf_max.wrapping_add(sign))
                } else {
                    let tok = rc_tok >> 11;
                    cul_level += tok;
                    (ac_dq * tok) >> dq_shift
                };
                cf[rc] = if sign != 0 { -(dq as i32) } else { dq as i32 };
                rc = (rc_tok & 0x3ff) as usize;
            }
        }
        Some(qm) => {
            if dc_tok == 0 {
                cul_level = 0;
            } else {
                let base_dq = (dc_dq * u32::from(qm[0]) + 16) >> 5;
                let dq = if dc_tok == 15 {
                    let tok = read_golomb(msac).wrapping_add(15) & 0xfffff;
                    cul_level = tok;
                    ((base_dq.wrapping_mul(tok) & 0xffffff) >> dq_shift).min(cf_max_signed)
                } else {
                    cul_level = dc_tok as u32;
                    ((base_dq * dc_tok as u32) >> dq_shift).min(cf_max_signed)
                };
                cf[0] = if dc_sign != 0 {
                    -(dq as i32)
                } else {
                    dq as i32
                };
            }
            while rc != 0 {
                let sign = msac.decode_bool_equi();
                let rc_tok = cf[rc] as u32;
                let base_dq = (ac_dq * u32::from(qm[rc]) + 16) >> 5;
                let cf_max_sign = cf_max.wrapping_add(sign);
                let dq = if rc_tok >= (15 << 11) {
                    let tok = read_golomb(msac).wrapping_add(15) & 0xfffff;
                    cul_level += tok;
                    ((base_dq.wrapping_mul(tok) & 0xffffff) >> dq_shift).min(cf_max_sign)
                } else {
                    let tok = rc_tok >> 11;
                    cul_level += tok;
                    ((base_dq * tok) >> dq_shift).min(cf_max_sign)
                };
                cf[rc] = if sign != 0 { -(dq as i32) } else { dq as i32 };
                rc = (rc_tok & 0x3ff) as usize;
            }
        }
    }
    CoefResult {
        eob,
        txtp,
        res_ctx: (cul_level.min(63) | dc_sign_level) as u8,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn golomb_c_ile_ayni() {
        // dav1d `read_golomb`: unary len + len bit; val-1.
        // Ayrıntılı test MSAC oracle koşusuna dayanır; burada yalnız
        // fonksiyonun sabitleri korunur.
        assert_eq!(read_golomb(&mut MsacContext::init(&[0xff; 8], true)), 0);
    }

    #[test]
    fn dc_sign_baglami_skaler_esdeger() {
        // 4x4: iki komşu, ikisi de negatif (>>6 == 2) → s = 4-2 = 2 → 2.
        let mut a = [0u8; 32];
        let mut l = [0u8; 32];
        a[0] = 0x80;
        a[1] = 0x80;
        l[0] = 0x80;
        l[1] = 0x80;
        assert_eq!(get_dc_sign_ctx(0, &a, &l), 2);
        // Hepsi pozitif → s = 0 → 0.
        a[0] = 0x40;
        a[1] = 0x40;
        l[0] = 0x40;
        l[1] = 0x40;
        assert_eq!(get_dc_sign_ctx(0, &a, &l), 0);
        // Yarısı pozitif yarısı negatif → s = 0 → 0.
        a[0] = 0x40;
        a[1] = 0x80;
        l[0] = 0x40;
        l[1] = 0x80;
        assert_eq!(get_dc_sign_ctx(0, &a, &l), 0);
    }

    /// `read_golomb`'un **64-bit referans** eşdeğeri (sarma yok).
    ///
    /// C gövdesi (`recon_tmpl.c:51-56`) `unsigned val` ile 32 bit sarmalıdır;
    /// bu referans `u64` ile hesaplar ve sonucu 32 bite indirir. İki taraf
    /// bütün bit dizilerinde aynı olmalıdır.
    fn golomb_reference(msac: &mut MsacContext<'_>) -> u32 {
        let mut len = 0u32;
        let mut val = 1u64;
        while msac.decode_bool_equi() == 0 && len < 32 {
            len += 1;
        }
        while len > 0 {
            len -= 1;
            val = (val << 1) + u64::from(msac.decode_bool_equi());
        }
        (val as u32).wrapping_sub(1)
    }

    #[test]
    fn golomb_uc_degerler() {
        // --- 32 sıfır bit yolu: `len` 32'ye ulaşır, `val` 32 kez kaydırılır.
        // 0x00 baytları `dif`'i 0xff ile doldurur → `decode_bool_equi` 0 verir
        // (bkz. `MsacContext::ctx_refill` tümleyen semantiği).
        let mut s = MsacContext::init(&[0x00; 32], true);
        assert_eq!(
            read_golomb(&mut s),
            u32::MAX,
            "32 sıfır bit: val = 1u32 << 32 = 0, 0 - 1 sarmalı = 0xFFFF_FFFF"
        );

        // --- `+ 15` taşması: C'de `unsigned` sarma, Rust'ta `+` paniklerdi.
        let mut s = MsacContext::init(&[0x00; 32], true);
        let golomb = read_golomb(&mut s);
        assert!(
            golomb.checked_add(15).is_none(),
            "u32::MAX + 15 taşar — düz `+` hata ayıklamada paniklerdi"
        );
        assert_eq!(
            golomb.wrapping_add(15),
            14,
            "0xFFFF_FFFF + 15 = 0x1_0000_000E → 32 bitte 14"
        );
        // `recon_tmpl.c:628` maskesi: `dc_tok &= 0xfffff` (14 < 0xfffff).
        assert_eq!(golomb.wrapping_add(15) & 0xfffff, 14);

        // --- Golomb "1" biti: `len = 0`, `val = 1` → 0.
        let mut s = MsacContext::init(&[0xff; 16], true);
        assert_eq!(read_golomb(&mut s), 0);

        // --- `dq * tok` maskesi: çarpım 2^32'yi aşsa da **alt 24 bit** aynıdır.
        // dav1d `recon_tmpl.c:629,656,679,709`: `(dq * tok) & 0xffffff`.
        let dq = 0xffffu32; // `dq_tbl` en büyük `uint16_t`
        let tok = 0xfffffu32; // `& 0xfffff` sonrası en büyük token
        assert!(
            dq.checked_mul(tok).is_none(),
            "düz `*` taşar (6,9e10 > 2^32)"
        );
        let wrapped = dq.wrapping_mul(tok) & 0xffffff;
        let exact = ((u64::from(dq) * u64::from(tok)) & 0xffffff) as u32;
        assert_eq!(wrapped, exact, "maske alt 24 biti korur");
        // Sarmasız çarpım ile sarmalı çarpım maskeden **sonra** aynıdır;
        // fark yalnız taşma davranışındadır (C: sarma, Rust düz `*`: panik).
    }

    #[test]
    fn golomb_referansla_ayni() {
        // Deterministik LCG ile 2 000 rastgele tampon: `read_golomb` (u32 sarma)
        // ile 64-bit referans **her** durumda aynı olmalı.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for case in 0..2_000 {
            let mut buf = vec![0u8; 1 + (case % 48)];
            for b in &mut buf {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *b = (state >> 33) as u8;
            }
            let mut a = MsacContext::init(&buf, true);
            let mut b = MsacContext::init(&buf, true);
            assert_eq!(
                read_golomb(&mut a),
                golomb_reference(&mut b),
                "tampon {case}: sarmalı ve referans golomb ayrıştı"
            );
            assert_eq!(a.buf_pos(), b.buf_pos(), "tüketilen bayt aynı olmalı");
        }
    }
}

#[cfg(all(test, feature = "std"))]
pub(crate) fn coef_label(off: usize) -> String {
    use layout::*;
    let range = |start: usize, row: usize| {
        let rel = off - start;
        (rel / row, rel % row)
    };
    if off < COEF_eob_bin_32 {
        let (r, c) = range(COEF_eob_bin_16, 8);
        format!("eob16[{r}][{c}]")
    } else if off < COEF_eob_bin_64 {
        let (r, c) = range(COEF_eob_bin_32, 8);
        format!("eob32[{r}][{c}]")
    } else if off < COEF_eob_bin_128 {
        let (r, c) = range(COEF_eob_bin_64, 8);
        format!("eob64[{r}][{c}]")
    } else if off < COEF_eob_bin_256 {
        let (r, c) = range(COEF_eob_bin_128, 8);
        format!("eob128[{r}][{c}]")
    } else if off < COEF_eob_bin_512 {
        let (r, c) = range(COEF_eob_bin_256, 16);
        format!("eob256[{r}][{c}]")
    } else if off < COEF_eob_bin_1024 {
        let (r, c) = range(COEF_eob_bin_512, 16);
        format!("eob512[{r}][{c}]")
    } else if off < COEF_eob_base_tok {
        let (r, c) = range(COEF_eob_bin_1024, 16);
        format!("eob1024[{r}][{c}]")
    } else if off < COEF_base_tok {
        let (r, c) = range(COEF_eob_base_tok, 16);
        format!("eobtok[{r}][{c}]")
    } else if off < COEF_br_tok {
        let (r, c) = range(COEF_base_tok, 328);
        format!("basetok[{r}][{c}]")
    } else if off < COEF_eob_hi_bit {
        let (r, c) = range(COEF_br_tok, 168);
        format!("brtok[{r}][{c}]")
    } else if off < COEF_skip {
        let (r, c) = range(COEF_eob_hi_bit, 36);
        format!("eobhi[{r}][{c}]")
    } else if off < COEF_dc_sign {
        let (r, c) = range(COEF_skip, 26);
        format!("skip[{r}][{c}]")
    } else {
        let (r, c) = range(COEF_dc_sign, 6);
        format!("dcsign[{r}][{c}]")
    }
}

#[cfg(all(test, feature = "std"))]
mod oracle_diff {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn byte(&mut self) -> u8 {
            (self.next() & 0xff) as u8
        }
    }

    fn hex(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }

    fn fnv(cf: &[i32]) -> u64 {
        let mut hash = 14_695_981_039_346_656_037u64;
        for &v in cf {
            hash ^= v as i64 as u64;
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
        hash
    }

    #[test]
    #[ignore = "harici C oracle (scripts/oracle/coef_oracle.c) gerektirir"]
    fn katsayi_cozme_c_ile_ayni() {
        let oracle = std::env::var("COEF_ORACLE")
            .unwrap_or_else(|_| ".cache/coef-oracle/coef_oracle".to_string());
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let bd_bits = std::env::var("COEF_BD")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(8);
        let bit_depth = match bd_bits {
            10 => BitDepth::Ten,
            12 => BitDepth::Twelve,
            _ => BitDepth::Eight,
        };
        let mut cases = Vec::new();
        for idx in 0..4000 {
            let forced_tx = std::env::var("COEF_TX")
                .ok()
                .and_then(|v| v.parse::<usize>().ok());
            let lossless = forced_tx.is_none() && rng.below(4) == 0;
            let plane = if forced_tx.is_some() { 0 } else { rng.below(3) };
            let tx = if let Some(tx) = forced_tx {
                tx
            } else if lossless {
                0
            } else if plane != 0 {
                rng.below(4)
            } else {
                rng.below(19)
            };
            let bs = rng.below(22);
            let layout = 1 + rng.below(3);
            let ss_hor = usize::from(layout != 3);
            let ss_ver = usize::from(layout == 1);
            let reduced = rng.below(2) as u8;
            let segq0 = rng.below(2) as u8;
            let seg = rng.below(8);
            let y_mode = if rng.below(8) == 0 {
                13
            } else {
                rng.below(13) as u8
            };
            let y_angle = if y_mode == 13 { rng.below(5) as i32 } else { 0 };
            let uv_mode = rng.below(14) as u8;
            let qidx = [0u8, 21, 61, 121, 200][rng.below(5)];
            let mut a = [0u8; 32];
            let mut l = [0u8; 32];
            for v in &mut a {
                *v = [0x40u8, 0x00, 0x7f, 0x80][rng.below(4)];
            }
            for v in &mut l {
                *v = [0x40u8, 0x00, 0x7f, 0x80][rng.below(4)];
            }
            let dq_dc = (rng.below(64) + 1) as u16;
            let dq_ac = (rng.below(64) + 1) as u16;
            let msac_len = 1 + rng.below(80);
            let mut msac = vec![0u8; msac_len];
            for b in &mut msac {
                *b = rng.byte();
            }
            let disable = rng.below(2) as u8;
            let line = format!(
                "{idx} {tx} {bs} {plane} {} {reduced} {segq0} {y_mode} {uv_mode} {y_angle} {seg} {ss_hor} {ss_ver} {layout} {bd_bits} {disable} {qidx} {} {} {dq_dc} {dq_ac} {}\n",
                u8::from(lossless),
                hex(&a),
                hex(&l),
                hex(&msac)
            );
            cases.push((
                line, tx, bs, plane, lossless, reduced, segq0, y_mode, uv_mode, y_angle, seg,
                ss_hor, ss_ver, layout, disable, qidx, a, l, dq_dc, dq_ac, msac,
            ));
        }
        if let Ok(only) = std::env::var("COEF_CASE_ONLY") {
            let only: usize = only.parse().expect("COEF_CASE_ONLY");
            let case = cases[only].clone();
            cases = vec![case];
        }
        let mut child = Command::new(&oracle)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("coef oracle çalıştırılamadı (COEF_ORACLE)");
        let stdin = child.stdin.take().expect("stdin");
        let mut input = String::new();
        for case in &cases {
            input.push_str(&case.0);
        }
        let writer = std::thread::spawn(move || {
            let mut stdin = stdin;
            let _ = stdin.write_all(input.as_bytes());
        });
        let stdout = child.stdout.take().expect("stdout");
        let mut lines = BufReader::new(stdout).lines();
        let mut mismatches = 0usize;
        for (idx, case) in cases.iter().enumerate() {
            let line = lines.next().expect("oracle satırı yok").expect("okuma");
            let fields: Vec<&str> = line.split_whitespace().collect();
            let c_eob: i32 = fields[0].parse().unwrap();
            let c_txtp: u8 = fields[1].parse().unwrap();
            let c_res: u8 = fields[2].parse().unwrap();
            let c_rng: u32 = fields[3].parse().unwrap();
            let c_dif: u64 = fields[4].parse().unwrap();
            let c_cnt: i32 = fields[5].parse().unwrap();
            let c_pos: usize = fields[6].parse().unwrap();
            let c_hash = u64::from_str_radix(fields[7], 16).unwrap();

            let (
                tx,
                bs,
                plane,
                lossless,
                reduced,
                segq0,
                y_mode,
                uv_mode,
                y_angle,
                seg,
                ss_hor,
                ss_ver,
                layout,
                disable,
                qidx,
                a,
                l,
                dq_dc,
                dq_ac,
                msac,
            ) = (
                case.1,
                case.2,
                case.3,
                case.4,
                case.5,
                case.6,
                case.7,
                case.8,
                case.9,
                case.10,
                case.11,
                case.12,
                case.13,
                case.14,
                case.15,
                case.16,
                case.17,
                case.18,
                case.19,
                case.20.clone(),
            );
            let mut cdf = CdfContext::from_default(crate::entropy::cdf::qcat_for_qidx(qidx));
            let cdf_initial = cdf.clone();
            let mut mc = MsacContext::init(&msac, disable != 0);
            let env = CoefEnv {
                lossless,
                reduced_txtp: reduced != 0,
                seg_qidx: if segq0 != 0 { 0 } else { 1 },
                bit_depth,
                ss_ver: ss_ver != 0,
                ss_hor: ss_hor != 0,
                dq: &[[dq_dc, dq_ac]; 3],
                qm: None,
                // Bu differential testi **intra** yolu üretir (C oracle
                // `coef_oracle.c` intra tx türü CDF'lerini kullanır); inter
                // tx türü yolu kare differential'ında doğrulanır.
                inter: false,
                txtp_hint: 0,
            };
            let mut cf = [0i32; 1024];
            let mut levels = [0u8; 32 * 34 + 64];
            let y_mode_nofilt = if y_mode == 13 {
                t::FILTER_MODE_TO_Y_MODE[y_angle.clamp(0, 4) as usize]
            } else {
                y_mode
            };
            let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                decode_coefs(
                    &mut mc,
                    &mut cdf,
                    env,
                    tx as u8,
                    bs as u8,
                    y_mode_nofilt,
                    uv_mode,
                    plane,
                    &a,
                    &l,
                    &mut cf,
                    &mut levels,
                )
            }));
            let result = match decoded {
                Ok(result) => result,
                Err(_) => {
                    mismatches += 1;
                    if mismatches <= 8 {
                        eprintln!(
                            "PANIK vaka {idx}: tx={tx} bs={bs} plane={plane} ll={lossless} red={reduced} segq0={segq0} ym={y_mode} uvm={uv_mode} seg={seg} ss=({ss_hor},{ss_ver}) lay={layout} dis={disable} q={qidx}"
                        );
                        eprintln!("  C: {:?} cf {:?}", (c_eob, c_txtp, c_res), &fields[8..16]);
                    }
                    continue;
                }
            };
            let c_id: usize = line
                .split("id=")
                .nth(1)
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse().ok())
                .unwrap_or(9999);
            if c_id != idx {
                eprintln!("KIMLIK vaka {idx}: C id={c_id}");
            }
            let c_tx: u32 = line
                .split("tx=")
                .nth(1)
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse().ok())
                .unwrap_or(999);
            if c_tx != tx as u32 {
                eprintln!("HIZALAMA vaka {idx}: C tx={c_tx} Rust tx={tx}");
            }
            let r = (
                result.eob,
                result.txtp,
                result.res_ctx,
                mc.rng(),
                mc.dif(),
                mc.cnt(),
                mc.buf_pos(),
                fnv(&cf),
            );
            let expected = (c_eob, c_txtp, c_res, c_rng, c_dif, c_cnt, c_pos, c_hash);
            let mut ok = r == expected;
            if ok {
                for i in 0..128 {
                    let c: i32 = fields[8 + i].parse().unwrap();
                    if cf[i] != c {
                        ok = false;
                        break;
                    }
                }
            }
            // Adapte olmuş katsayı CDF hücrelerini karşılaştır (ilk 64 fark).
            let mut cdf_diff: Vec<(usize, String, u16)> = Vec::new();
            for i in 0..crate::entropy::cdf::CDF_COEF_CELL_COUNT {
                let before = cdf_initial.coef()[i];
                let after = cdf.coef()[i];
                if before != after {
                    cdf_diff.push((i, coef_label(i), after));
                }
            }
            if let Some(pipe) = line.find('|') {
                let tail = &line[pipe + 1..];
                let mut c_diff: Vec<(usize, String, u16)> = Vec::new();
                for token in tail.split_whitespace() {
                    let (off, rest) = token.split_once(':').unwrap();
                    let (name, val) = rest.split_once('=').unwrap();
                    c_diff.push((off.parse().unwrap(), name.to_string(), val.parse().unwrap()));
                }
                let c_off: Vec<(usize, u16)> = c_diff.iter().map(|(o, _, v)| (*o, *v)).collect();
                let r_off: Vec<(usize, u16)> = cdf_diff.iter().map(|(o, _, v)| (*o, *v)).collect();
                if c_off != r_off {
                    ok = false;
                    if mismatches < 40 {
                        let first = c_diff
                            .iter()
                            .zip(&cdf_diff)
                            .position(|(a, b)| a != b)
                            .unwrap_or(c_diff.len().min(cdf_diff.len()));
                        eprintln!(
                            "  cdf farkı ilk {first} (C ctx={:?}): C {:?} R {:?}",
                            line.split("ctx=")
                                .nth(1)
                                .and_then(|v| v.split_whitespace().next()),
                            &c_diff[first..(first + 8).min(c_diff.len())],
                            &cdf_diff[first..(first + 8).min(cdf_diff.len())]
                        );
                    }
                }
            }
            if !ok {
                mismatches += 1;
                if mismatches <= 20 {
                    eprintln!(
                        "VAKA {idx}: tx={tx} bs={bs} plane={plane} ll={lossless} red={reduced} segq0={segq0} ym={y_mode} uvm={uv_mode} ya={y_angle} seg={seg} ss=({ss_hor},{ss_ver}) lay={layout} dis={disable} q={qidx} c_ctx={:?} c_dis={:?}",
                        line.split("ctx=")
                            .nth(1)
                            .and_then(|v| v.split_whitespace().next()),
                        line.split("dis=")
                            .nth(1)
                            .and_then(|v| v.split_whitespace().next())
                    );
                    eprintln!("  C   : {:?}", expected);
                    eprintln!("  Rust: {:?}", r);
                    eprintln!("  cf C : {:?}", &fields[8..40]);
                    eprintln!("  cf R : {:?}", &cf[..32]);
                }
            }
            let _ = layout;
        }
        let _ = writer.join();
        let _ = child.wait();
        assert_eq!(mismatches, 0, "{mismatches}/{} vaka uyuşmadı", cases.len());
    }
}
