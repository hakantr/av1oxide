//! Intra kenar hazırlama (`prepare_intra_edges`) — S2b.3.
//!
//! dav1d `src/ipred_prepare_tmpl.c` (32–204) C yolunun birebir karşılığı.
//! Komşu pikselleri, kullanılabilirlik bayraklarına göre tek bir kenar
//! tamponuna toplar; yönlü modlarda açıyı mutlak açıya çevirir, kenar
//! filtresi/yumuşak blok bayraklarını çağıran ekler.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`)
//!
//! | Kural | dav1d satırı | AV1 § |
//! |---|---|---|
//! | mod dönüşümü `av1_mode_conv` | `ipred_prepare_tmpl.c:39-53` | 7.11.2 |
//! | açı eşlemesi `av1_mode_to_angle_map` | `:55-57` | 7.11.2 |
//! | mod başına kenar gereksinimi | `:59-77` | 7.11.2 |
//! | yönlü mod → Z1/Z2/Z3/VERT/HOR | `:112-130` | 7.11.2 |
//! | sol/alt-sol kenar toplama + uzatma | `:139-176` | 7.11.2 |
//! | üst/üst-sağ kenar toplama + uzatma | `:178-208` | 7.11.2 |
//! | sol-üst ve Z2 kenar filtresi | `:210-223` | 7.11.2 |
//!
//! ## Bellek sözleşmesi
//!
//! `edge` tamponu `[topleft ± 2*64]` piksel tutar; `topleft` bayt ofseti
//! `128 * bpp`'dir. `dst` düzlem baytlarıdır; `dst_off` blok sol-üst
//! pikselinin bayt ofsetidir. Negatif yön okumaları yalnız `have_left` /
//! `have_top` / bayrak koşullarında yapılır; blok dışına taşma yoktur.

#![forbid(unsafe_code)]

use crate::decode::edge_tree::{
    EDGE_I420_LEFT_HAS_BOTTOM, EDGE_I420_TOP_HAS_RIGHT, EDGE_I422_LEFT_HAS_BOTTOM,
    EDGE_I422_TOP_HAS_RIGHT, EDGE_I444_LEFT_HAS_BOTTOM, EDGE_I444_TOP_HAS_RIGHT,
};
use crate::frame::BitDepth;

/// Kenar gereksinimi bayrakları (`av1_intra_prediction_edges`).
#[derive(Clone, Copy, Default)]
struct Edges {
    left: bool,
    top: bool,
    topleft: bool,
    topright: bool,
    bottomleft: bool,
}

const fn edges(left: bool, top: bool, topleft: bool, topright: bool, bottomleft: bool) -> Edges {
    Edges {
        left,
        top,
        topleft,
        topright,
        bottomleft,
    }
}

/// `av1_intra_prediction_edges` (14 uygulama modu).
const EDGE_NEEDS: [Edges; 14] = [
    edges(true, true, false, false, false),   // DC
    edges(false, true, false, false, false),  // VERT
    edges(true, false, false, false, false),  // HOR
    edges(true, false, false, false, false),  // LEFT_DC (DIAG_DOWN_LEFT)
    edges(false, true, false, false, false),  // TOP_DC (DIAG_DOWN_RIGHT)
    edges(false, false, false, false, false), // DC_128
    edges(false, true, true, true, false),    // Z1
    edges(true, true, true, false, false),    // Z2
    edges(true, false, true, false, true),    // Z3
    edges(true, true, false, false, false),   // SMOOTH
    edges(true, true, false, false, false),   // SMOOTH_V
    edges(true, true, false, false, false),   // SMOOTH_H
    edges(true, true, true, false, false),    // PAETH
    edges(true, true, true, false, false),    // FILTER
];

/// `av1_mode_to_angle_map`: `VERT..VERT_LEFT` için temel açılar.
const MODE_TO_ANGLE: [i32; 8] = [90, 180, 45, 135, 113, 157, 203, 67];

/// Bayt ofsetinden piksel okur.
#[inline]
pub(crate) fn px_get(bytes: &[u8], off: usize, bd: BitDepth) -> u16 {
    if bd.bytes_per_sample() == 1 {
        u16::from(bytes[off])
    } else {
        u16::from_le_bytes([bytes[off], bytes[off + 1]])
    }
}

/// Bayt ofsetine piksel yazar.
#[inline]
pub(crate) fn px_set(bytes: &mut [u8], off: usize, value: u16, bd: BitDepth) {
    if bd.bytes_per_sample() == 1 {
        bytes[off] = value as u8;
    } else {
        bytes[off..off + 2].copy_from_slice(&value.to_le_bytes());
    }
}

/// Kenar tamponunu `count` piksel aynı değerle doldurur.
#[inline(always)]
fn px_fill(bytes: &mut [u8], off: usize, value: u16, count: usize, bd: BitDepth) {
    if bd.bytes_per_sample() == 1 {
        bytes[off..off + count].fill(value as u8);
    } else {
        let pair = value.to_le_bytes();
        bytes[off..off + count * 2]
            .as_chunks_mut::<2>()
            .0
            .fill(pair);
    }
}

/// `prepare_intra_edges` dönüşü.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Prepared {
    /// DSP `intra_pred` indeksi (0..13).
    pub mode: u8,
    /// Mutlak açı (yönlü) ya da FILTER_PRED filtre indeksi.
    pub angle: i32,
}

/// Blok döngüsü boyunca sabit kalan intra kenar hazırlama bağlamı (`5.D.1`).
#[derive(Clone, Copy)]
pub(crate) struct EdgeCtx {
    pub col_start: usize,
    pub row_start: usize,
    pub col_end: usize,
    pub row_end: usize,
    pub stride: usize,
    pub mode_in: u8,
    pub angle_in: i32,
    pub tw: usize,
    pub th: usize,
    pub filter_edge: bool,
    pub topleft_off: usize,
    pub bd: BitDepth,
}

/// Blok döngüsünde düşük çağrı maliyetli `prepare_intra_edges` girişi.
#[inline(always)]
pub(crate) fn prepare_intra_edges_ctx(
    ctx: &EdgeCtx,
    dst: &[u8],
    x: usize,
    y: usize,
    edge_flags: u8,
    dst_off: usize,
    edge: &mut [u8],
) -> Prepared {
    if ctx.bd.bytes_per_sample() == 1 {
        prepare_intra_edges_ctx_bpp::<1>(ctx, dst, x, y, edge_flags, dst_off, edge)
    } else {
        prepare_intra_edges_ctx_bpp::<2>(ctx, dst, x, y, edge_flags, dst_off, edge)
    }
}

#[inline(never)]
fn prepare_intra_edges_ctx_bpp<const BPP: usize>(
    ctx: &EdgeCtx,
    dst: &[u8],
    x: usize,
    y: usize,
    edge_flags: u8,
    dst_off: usize,
    edge: &mut [u8],
) -> Prepared {
    prepare_intra_edges_core::<BPP>(
        x,
        x > ctx.col_start,
        y,
        y > ctx.row_start,
        ctx.col_end,
        ctx.row_end,
        edge_flags,
        dst,
        dst_off,
        ctx.stride,
        ctx.mode_in,
        ctx.angle_in,
        ctx.tw,
        ctx.th,
        ctx.filter_edge,
        edge,
        ctx.topleft_off,
        ctx.bd,
    )
}

/// `dav1d_prepare_intra_edges` (C yolu).
///
/// - `x`, `y`: blok konumu (4px birimi), `have_left` = `x > tile_col_start`,
///   `have_top` = `y > tile_row_start`,
/// - `w`, `h`: karo sınırları (4px birimi, sağ/alt hariç),
/// - `dst`, `dst_off`, `stride`: düzlem baytları ve blok sol-üstü,
/// - `edge`, `topleft_off`: kenar tamponu ve `[0]` pikselinin bayt ofseti.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn prepare_intra_edges(
    x: usize,
    have_left: bool,
    y: usize,
    have_top: bool,
    w: usize,
    h: usize,
    edge_flags: u8,
    dst: &[u8],
    dst_off: usize,
    stride: usize,
    mode_in: u8,
    angle_in: i32,
    tw: usize,
    th: usize,
    filter_edge: bool,
    edge: &mut [u8],
    topleft_off: usize,
    bd: BitDepth,
) -> Prepared {
    if bd.bytes_per_sample() == 1 {
        prepare_intra_edges_bpp::<1>(
            x,
            have_left,
            y,
            have_top,
            w,
            h,
            edge_flags,
            dst,
            dst_off,
            stride,
            mode_in,
            angle_in,
            tw,
            th,
            filter_edge,
            edge,
            topleft_off,
            bd,
        )
    } else {
        prepare_intra_edges_bpp::<2>(
            x,
            have_left,
            y,
            have_top,
            w,
            h,
            edge_flags,
            dst,
            dst_off,
            stride,
            mode_in,
            angle_in,
            tw,
            th,
            filter_edge,
            edge,
            topleft_off,
            bd,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prepare_intra_edges_bpp<const BPP: usize>(
    x: usize,
    have_left: bool,
    y: usize,
    have_top: bool,
    w: usize,
    h: usize,
    edge_flags: u8,
    dst: &[u8],
    dst_off: usize,
    stride: usize,
    mode_in: u8,
    angle_in: i32,
    tw: usize,
    th: usize,
    filter_edge: bool,
    edge: &mut [u8],
    topleft_off: usize,
    bd: BitDepth,
) -> Prepared {
    prepare_intra_edges_core::<BPP>(
        x,
        have_left,
        y,
        have_top,
        w,
        h,
        edge_flags,
        dst,
        dst_off,
        stride,
        mode_in,
        angle_in,
        tw,
        th,
        filter_edge,
        edge,
        topleft_off,
        bd,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn prepare_intra_edges_core<const BPP: usize>(
    x: usize,
    have_left: bool,
    y: usize,
    have_top: bool,
    w: usize,
    h: usize,
    edge_flags: u8,
    dst: &[u8],
    dst_off: usize,
    stride: usize,
    mode_in: u8,
    angle_in: i32,
    tw: usize,
    th: usize,
    filter_edge: bool,
    edge: &mut [u8],
    topleft_off: usize,
    bd: BitDepth,
) -> Prepared {
    let bpp = BPP;
    let bitdepth_max = bd.max_sample() as u16;
    let mid = (bitdepth_max as i32 + 1) >> 1;
    let mut mode = mode_in;
    let mut angle = angle_in;

    match mode {
        1..=8 => {
            angle = MODE_TO_ANGLE[(mode - 1) as usize] + 3 * angle;
            mode = if angle <= 90 {
                if angle < 90 && have_top {
                    6 // Z1
                } else {
                    1 // VERT
                }
            } else if angle < 180 {
                7 // Z2
            } else if angle > 180 && have_left {
                8 // Z3
            } else {
                2 // HOR
            };
        }
        0 | 12 => {
            // DC / PAETH: av1_mode_conv[mode][have_left][have_top]
            mode = if mode == 0 {
                [[5, 4], [3, 0]][usize::from(have_left)][usize::from(have_top)]
            } else {
                [[5, 1], [2, 12]][usize::from(have_left)][usize::from(have_top)]
            };
        }
        _ => {}
    }

    if mode == 5 {
        return Prepared { mode, angle };
    }

    let need = EDGE_NEEDS[mode as usize];
    let dst_top_off = dst_off.saturating_sub(stride);

    if need.left {
        let sz = th << 2;
        let left_off = topleft_off - sz * bpp;
        if have_left {
            let px_have = sz.min(h.saturating_sub(y) << 2);
            if BPP == 1 {
                let s = dst_off.saturating_sub(1);
                if px_have == 4 && sz == 4 {
                    let s1 = s + stride;
                    let s2 = s1 + stride;
                    let s3 = s2 + stride;
                    let bytes = [dst[s3], dst[s2], dst[s1], dst[s]];
                    edge[left_off..left_off + 4].copy_from_slice(&bytes);
                } else if px_have == 8 && sz == 8 {
                    let s1 = s + stride;
                    let s2 = s1 + stride;
                    let s3 = s2 + stride;
                    let s4 = s3 + stride;
                    let s5 = s4 + stride;
                    let s6 = s5 + stride;
                    let s7 = s6 + stride;
                    let bytes = [
                        dst[s7], dst[s6], dst[s5], dst[s4], dst[s3], dst[s2], dst[s1], dst[s],
                    ];
                    edge[left_off..left_off + 8].copy_from_slice(&bytes);
                } else {
                    let mut si = s;
                    for out in edge[left_off + (sz - px_have)..left_off + sz]
                        .iter_mut()
                        .rev()
                    {
                        *out = dst[si];
                        si += stride;
                    }
                }
            } else {
                let s = dst_off.saturating_sub(2);
                if px_have == 4 && sz == 4 {
                    let s1 = s + stride;
                    let s2 = s1 + stride;
                    let s3 = s2 + stride;
                    let bytes = [
                        dst[s3],
                        dst[s3 + 1],
                        dst[s2],
                        dst[s2 + 1],
                        dst[s1],
                        dst[s1 + 1],
                        dst[s],
                        dst[s + 1],
                    ];
                    edge[left_off..left_off + 8].copy_from_slice(&bytes);
                } else {
                    let mut si = s;
                    for out in edge[left_off + (sz - px_have) * 2..left_off + sz * 2]
                        .as_chunks_mut::<2>()
                        .0
                        .iter_mut()
                        .rev()
                    {
                        *out = [dst[si], dst[si + 1]];
                        si += stride;
                    }
                }
            }
            if px_have < sz {
                let value = px_get(edge, left_off + (sz - px_have) * bpp, bd);
                px_fill(edge, left_off, value, sz - px_have, bd);
            }
        } else {
            let value = if have_top {
                px_get(dst, dst_top_off, bd)
            } else {
                (mid + 1) as u16
            };
            px_fill(edge, left_off, value, sz, bd);
        }

        if need.bottomleft {
            let have_bottomleft =
                have_left && y + th < h && edge_flags & EDGE_I444_LEFT_HAS_BOTTOM != 0;
            if have_bottomleft {
                let px_have = sz.min((h - y - th) << 2);
                if BPP == 1 {
                    let mut s = (dst_off + sz * stride).saturating_sub(1);
                    for out in edge[left_off - px_have..left_off].iter_mut().rev() {
                        *out = dst[s];
                        s += stride;
                    }
                } else {
                    let mut s = (dst_off + sz * stride).saturating_sub(2);
                    for out in edge[left_off - px_have * 2..left_off]
                        .as_chunks_mut::<2>()
                        .0
                        .iter_mut()
                        .rev()
                    {
                        *out = [dst[s], dst[s + 1]];
                        s += stride;
                    }
                }
                if px_have < sz {
                    let value = px_get(edge, left_off - px_have * bpp, bd);
                    px_fill(edge, left_off - sz * bpp, value, sz - px_have, bd);
                }
            } else {
                let value = px_get(edge, left_off, bd);
                px_fill(edge, left_off - sz * bpp, value, sz, bd);
            }
        }
    }

    if need.top {
        let sz = tw << 2;
        let top_off = topleft_off + bpp;
        if have_top {
            let px_have = sz.min(w.saturating_sub(x) << 2);
            let nbytes = px_have * bpp;
            if BPP == 1 && nbytes == 4 {
                let b = [
                    dst[dst_top_off],
                    dst[dst_top_off + 1],
                    dst[dst_top_off + 2],
                    dst[dst_top_off + 3],
                ];
                edge[top_off..top_off + 4].copy_from_slice(&b);
            } else if BPP == 1 && nbytes == 8 {
                let b: [u8; 8] = dst[dst_top_off..dst_top_off + 8].try_into().unwrap();
                edge[top_off..top_off + 8].copy_from_slice(&b);
            } else {
                edge[top_off..top_off + nbytes]
                    .copy_from_slice(&dst[dst_top_off..dst_top_off + nbytes]);
            }
            if px_have < sz {
                let src = if px_have == 0 {
                    topleft_off
                } else {
                    top_off + (px_have - 1) * bpp
                };
                let value = px_get(edge, src, bd);
                px_fill(edge, top_off + px_have * bpp, value, sz - px_have, bd);
            }
        } else {
            let value = if have_left {
                px_get(dst, dst_off.saturating_sub(bpp), bd)
            } else {
                (mid - 1) as u16
            };
            px_fill(edge, top_off, value, sz, bd);
        }

        if need.topright {
            let have_topright = have_top && x + tw < w && edge_flags & EDGE_I444_TOP_HAS_RIGHT != 0;
            if have_topright {
                let px_have = sz.min((w - x - tw) << 2);
                let off_add = sz * bpp;
                let nbytes = px_have * bpp;
                edge[top_off + off_add..top_off + off_add + nbytes]
                    .copy_from_slice(&dst[dst_top_off + off_add..dst_top_off + off_add + nbytes]);
                if px_have < sz {
                    let value = px_get(edge, top_off + (sz + px_have - 1) * bpp, bd);
                    px_fill(
                        edge,
                        top_off + (sz + px_have) * bpp,
                        value,
                        sz - px_have,
                        bd,
                    );
                }
            } else {
                let value = px_get(edge, top_off + (sz - 1) * bpp, bd);
                px_fill(edge, top_off + sz * bpp, value, sz, bd);
            }
        }
    }

    if need.topleft {
        let value = if have_left {
            if have_top {
                px_get(dst, dst_top_off.saturating_sub(bpp), bd)
            } else {
                px_get(dst, dst_off.saturating_sub(bpp), bd)
            }
        } else if have_top {
            px_get(dst, dst_top_off, bd)
        } else {
            mid as u16
        };
        px_set(edge, topleft_off, value, bd);
        if mode == 7 && tw + th >= 6 && filter_edge {
            let l = px_get(edge, topleft_off - bpp, bd) as i32;
            let t = px_get(edge, topleft_off + bpp, bd) as i32;
            let c = px_get(edge, topleft_off, bd) as i32;
            let filtered = ((l + t) * 5 + c * 6 + 8) >> 4;
            px_set(edge, topleft_off, filtered as u16, bd);
        }
    }

    Prepared { mode, angle }
}

/// Kroma için `EDGE_I4xx_*` bayrağı seçer (`recon_tmpl.c:1430-1437`).
pub(crate) fn chroma_edge_masks(layout: u8) -> (u8, u8) {
    // layout: 1 = I420, 2 = I422, 3 = I444 (dav1d `enum Dav1dPixelLayout`)
    let idx = layout.saturating_sub(1) as u32;
    (
        EDGE_I420_TOP_HAS_RIGHT >> idx,
        EDGE_I420_LEFT_HAS_BOTTOM >> idx,
    )
}

/// Menzil dışı layout'lar için derleme zamanı doğrulaması (kullanılmayan sabitler).
const _: () = {
    assert!(EDGE_I444_TOP_HAS_RIGHT != 0);
    assert!(EDGE_I422_TOP_HAS_RIGHT != 0);
    assert!(EDGE_I420_TOP_HAS_RIGHT != 0);
    assert!(EDGE_I444_LEFT_HAS_BOTTOM != 0);
    assert!(EDGE_I422_LEFT_HAS_BOTTOM != 0);
    assert!(EDGE_I420_LEFT_HAS_BOTTOM != 0);
};

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
    }

    fn fill(seed: u32) -> [u8; 64 * 64] {
        let mut buf = [0u8; 64 * 64];
        let mut s = seed;
        for b in &mut buf {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }
        buf
    }

    /// Alt-sol (Z3) kenar yerleşimi: dav1d `ipred_prepare_tmpl.c:151-161`.
    ///
    /// `left[-(i+1)]` **sol kenar dizisinin altına** yazılır; `topleft[-(i+1)]`
    /// yazılırsa sol kenar bozulur (T27'de düzeltilen gerçek hata). Bu test
    /// C oracle'ı olmadan o yerleşimi sabitler.
    #[test]
    fn alt_sol_kenar_yerlesimi_dav1d_gibi() {
        // 64×64 8-bit düzlem: dst[y*64 + x] = 10*y + x.
        let mut dst = [0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                dst[y * 64 + x] = (10 * y + x) as u8;
            }
        }
        let mut edge = [0xAAu8; 257];
        // x=2 (4px birimi) → dst x=2; y=0 → üst yok; geometri 8×8 blok.
        // HOR_UP (7) + açı 0 → 203° → have_left ile Z3.
        let prepared = prepare_intra_edges(
            2,
            true,
            0,
            false,
            8,
            8,
            super::EDGE_I444_LEFT_HAS_BOTTOM,
            &dst,
            2, // dst_off: (x=2, y=0)
            64,
            7,
            0,
            1,
            1,
            false,
            &mut edge,
            128,
            BitDepth::Eight,
        );
        assert_eq!(prepared.mode, 8, "Z3 beklenir");
        assert_eq!(prepared.angle, 203);
        // Sol kenar: left[3-i] = dst[y=i, x=1] → 1, 11, 21, 31.
        assert_eq!(&edge[124..128], &[31, 21, 11, 1], "sol kenar (left[0..4])");
        // Alt-sol: left[-(i+1)] = dst[y=4+i, x=1] → 41, 51, 61, 71.
        assert_eq!(&edge[120..124], &[71, 61, 51, 41], "alt-sol (left[-4..-1])");
        // Sol üst köşe: have_left && !have_top → dst[-1] = dst[x=1, y=0] = 1.
        assert_eq!(edge[128], 1, "topleft");
    }

    #[test]
    #[ignore = "harici C oracle (scripts/oracle/ipred_edge_oracle.c) gerektirir"]
    fn kenar_hazirlama_c_ile_ayni() {
        let oracle = std::env::var("IPRED_EDGE_ORACLE").unwrap_or_else(|_| {
            "/home/hakan/github/av1oxide/.cache/ipred-edge-oracle/ipred_edge_oracle".into()
        });
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut cases = Vec::new();
        for _ in 0..3000 {
            let seed = rng.next() as u32;
            let dst_x = rng.below(5) * 4;
            let dst_y = rng.below(5) * 4;
            let max_tw = (64 - dst_x) / 4;
            let max_th = (64 - dst_y) / 4;
            let tw = 1 + rng.below(max_tw.min(16));
            let th = 1 + rng.below(max_th.min(16));
            let x = dst_x / 4;
            let y = dst_y / 4;
            let have_left = usize::from(x > 0 && rng.below(2) == 0);
            let have_top = usize::from(y > 0 && rng.below(2) == 0);
            // `w`/`h` karo sınırıdır; blok kenarından **büyük** seçilirse
            // `px_have > 0` yolları (komşu piksel okuma, alt-sol/üst-sağ
            // kırpma) tetiklenir. Eski üretici `w = x + tw` sabitlediği için
            // `px_have` hep 0 kalıyordu ve bu yollar hiç sınanmıyordu
            // (T27'de bulunan `left[-(i+1)]` hatası bu yüzden kaçtı).
            let extra_w = rng.below(3);
            let extra_h = rng.below(3);
            let w = (x + tw + extra_w).min(x + (64 - dst_x) / 4);
            let h = (y + th + extra_h).min(y + (64 - dst_y) / 4);
            let edge_flags = [0u8, 1, 4, 8, 15, 32, 39][rng.below(7)];
            let mode = rng.below(14) as u8;
            let angle = if mode == 13 {
                rng.below(5) as i32
            } else {
                rng.below(7) as i32 - 3
            };
            let filter_edge = rng.below(2);
            let line = format!(
                "{seed} {x} {have_left} {y} {have_top} {w} {h} {edge_flags} {mode} {angle} {tw} {th} {filter_edge} {dst_x} {dst_y}\n"
            );
            cases.push((
                line,
                seed,
                x,
                have_left,
                y,
                have_top,
                w,
                h,
                edge_flags,
                mode,
                angle,
                tw,
                th,
                filter_edge,
                dst_x,
                dst_y,
            ));
        }
        let mut child = Command::new(&oracle)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("ipred edge oracle çalıştırılamadı");
        let stdin = child.stdin.take().unwrap();
        let mut input = String::new();
        for c in &cases {
            input.push_str(&c.0);
        }
        let writer = std::thread::spawn(move || {
            let mut stdin = stdin;
            let _ = stdin.write_all(input.as_bytes());
        });
        let stdout = child.stdout.take().unwrap();
        let mut lines = BufReader::new(stdout).lines();
        let mut mismatches = 0usize;
        for case in &cases {
            let line = lines.next().expect("oracle satırı yok").unwrap();
            let fields: Vec<i32> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let buf = fill(case.1);
            let mut edge = [0xAAu8; 257];
            let prepared = prepare_intra_edges(
                case.2,
                case.3 != 0,
                case.4,
                case.5 != 0,
                case.6,
                case.7,
                case.8,
                &buf,
                case.15 * 64 + case.14,
                64,
                case.9,
                case.10,
                case.11,
                case.12,
                case.13 != 0,
                &mut edge,
                128,
                BitDepth::Eight,
            );
            let mut ok = prepared.mode as i32 == fields[0] && prepared.angle == fields[1];
            if ok {
                for i in 0..257 {
                    if i32::from(edge[i]) != fields[2 + i] {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "VAKA mode={} ang={} x={} hl={} y={} ht={} w={} h={} flags={} tw={} th={} fe={}",
                        case.9,
                        case.10,
                        case.2,
                        case.3,
                        case.4,
                        case.5,
                        case.6,
                        case.7,
                        case.8,
                        case.11,
                        case.12,
                        case.13
                    );
                    eprintln!(
                        "  C: mode={} ang={} edge={:?}",
                        fields[0],
                        fields[1],
                        &fields[2..20]
                    );
                    eprintln!(
                        "  R: mode={} ang={} edge={:?}",
                        prepared.mode,
                        prepared.angle,
                        &edge[..18]
                    );
                }
            }
        }
        let _ = writer.join();
        let _ = child.wait();
        assert_eq!(mismatches, 0, "{mismatches}/{} vaka uyuşmadı", cases.len());
    }
}
