//! Wedge ve interintra maske tabloları — dav1d `src/wedge.c` birebir karşılığı.
//!
//! ## Sorumluluk
//!
//! Compound tahminin **wedge** ve **fark (seg) maskeleri** ile interintra
//! birleştirme maskelerini üretir. Tablolar dav1d'de süreç başında bir kez
//! kurulur (`dav1d_init_ii_wedge_masks`, `src/wedge.c:207-299`); burada da
//! [`Masks`] tembel tek seferlik kurulumla (`OnceLock`) üretilir.
//!
//! ## Upstream karşılığı (dav1d @ `99f3354`, salt-okunur)
//!
//! | Rust | dav1d satırı |
//! |---|---|
//! | [`Masks`] / [`Masks::build`] | `src/wedge.c:207-299` (`dav1d_init_ii_wedge_masks`) |
//! | [`Masks::wedge`] | `src/wedge.h:103-105` (`WEDGE_MASK`) |
//! | [`Masks::interintra`] | `src/wedge.h:97-102` (`II_MASK`) |
//! | `insert_border` | `src/wedge.c:81-88` |
//! | `transpose` / `hflip` | `src/wedge.c:90-101` |
//! | `copy2d` | `src/wedge.c:103-119` |
//! | `init_chroma` | `src/wedge.c:122-136` |
//! | `fill2d_16x2` | `src/wedge.c:138-176` |
//! | `build_nondc_ii_masks` | `src/wedge.c:178-199` |
//! | `wedge_codebook_16_*` | `src/wedge.c:48-68` |
//!
//! ## Ölçü birimleri
//!
//! Maske boyutları **piksel**tir (4×4 birimi değil). Blok boyutu indeksi
//! (`bs`) dav1d `enum BlockSize` diskriminantıdır; tablo yalnız
//! `BS_32x32..=BS_8x8` aralığını kapsar ([`FIRST_BS`], [`N_BS_ENTRIES`]) —
//! wedge ve interintra yalnız bu boyutlarda geçerlidir (dav1d
//! `src/tables.h:86-105` `wedge_allowed_mask`, `interintra_allowed_mask`).
//!
//! ## Bilinçli fark
//!
//! dav1d maske baytlarını `Dav1dMasks` yapısının içine, `uint16_t` **8 baytlık
//! ofsetlerle** yerleştirir (`MASK_OFFSET`); burada aynı baytlar tek bir
//! `Vec<u8>` arenasında tutulur ve `(layout, bs, tür, indeks)` ile erişilir.
//! **Bayt içerikleri birebir aynıdır**; yalnız bellek düzeni farklıdır
//! (oracle karşılaştırması kanonik sırayla bayt bayt yapılır).

#![forbid(unsafe_code)]

use alloc::vec::Vec;

use crate::sync::OnceLock;

/// İlk maske tablosu blok boyutu: dav1d `BS_32x32` = 7.
pub(crate) const FIRST_BS: usize = 7;
/// Maske tablosu kapsamındaki blok boyutu sayısı (`BS_8x8 - BS_32x32 + 1`).
pub(crate) const N_BS_ENTRIES: usize = 11;
/// Kroma düzeni indeksi sayısı: 4:4:4, 4:2:2, 4:2:0.
pub(crate) const N_LAYOUTS: usize = 3;
/// Wedge indeksi sayısı (CDF 16 sembol).
pub(crate) const N_WEDGE_IDX: usize = 16;
/// Interintra mod sayısı (`N_INTER_INTRA_PRED_MODES`).
pub(crate) const N_II_MODES: usize = 4;

/// `enum WedgeDirectionType` (`src/wedge.c:35-42`).
const WEDGE_HORIZONTAL: u8 = 0;
const WEDGE_VERTICAL: u8 = 1;
const WEDGE_OBLIQUE27: u8 = 2;
const WEDGE_OBLIQUE63: u8 = 3;
const WEDGE_OBLIQUE117: u8 = 4;
const WEDGE_OBLIQUE153: u8 = 5;
const N_WEDGE_DIRECTIONS: usize = 6;

/// `wedge_code_type`.
#[derive(Clone, Copy)]
struct WedgeCode {
    direction: u8,
    x_offset: u8,
    y_offset: u8,
}

const fn wc(direction: u8, x_offset: u8, y_offset: u8) -> WedgeCode {
    WedgeCode {
        direction,
        x_offset,
        y_offset,
    }
}

/// `wedge_codebook_16_hgtw` (`src/wedge.c:48-55`).
static WEDGE_CODEBOOK_16_HGTW: [WedgeCode; 16] = [
    wc(WEDGE_OBLIQUE27, 4, 4),
    wc(WEDGE_OBLIQUE63, 4, 4),
    wc(WEDGE_OBLIQUE117, 4, 4),
    wc(WEDGE_OBLIQUE153, 4, 4),
    wc(WEDGE_HORIZONTAL, 4, 2),
    wc(WEDGE_HORIZONTAL, 4, 4),
    wc(WEDGE_HORIZONTAL, 4, 6),
    wc(WEDGE_VERTICAL, 4, 4),
    wc(WEDGE_OBLIQUE27, 4, 2),
    wc(WEDGE_OBLIQUE27, 4, 6),
    wc(WEDGE_OBLIQUE153, 4, 2),
    wc(WEDGE_OBLIQUE153, 4, 6),
    wc(WEDGE_OBLIQUE63, 2, 4),
    wc(WEDGE_OBLIQUE63, 6, 4),
    wc(WEDGE_OBLIQUE117, 2, 4),
    wc(WEDGE_OBLIQUE117, 6, 4),
];

/// `wedge_codebook_16_hltw` (`src/wedge.c:57-64`).
static WEDGE_CODEBOOK_16_HLTW: [WedgeCode; 16] = [
    wc(WEDGE_OBLIQUE27, 4, 4),
    wc(WEDGE_OBLIQUE63, 4, 4),
    wc(WEDGE_OBLIQUE117, 4, 4),
    wc(WEDGE_OBLIQUE153, 4, 4),
    wc(WEDGE_VERTICAL, 2, 4),
    wc(WEDGE_VERTICAL, 4, 4),
    wc(WEDGE_VERTICAL, 6, 4),
    wc(WEDGE_HORIZONTAL, 4, 4),
    wc(WEDGE_OBLIQUE27, 4, 2),
    wc(WEDGE_OBLIQUE27, 4, 6),
    wc(WEDGE_OBLIQUE153, 4, 2),
    wc(WEDGE_OBLIQUE153, 4, 6),
    wc(WEDGE_OBLIQUE63, 2, 4),
    wc(WEDGE_OBLIQUE63, 6, 4),
    wc(WEDGE_OBLIQUE117, 2, 4),
    wc(WEDGE_OBLIQUE117, 6, 4),
];

/// `wedge_codebook_16_heqw` (`src/wedge.c:66-73`).
static WEDGE_CODEBOOK_16_HEQW: [WedgeCode; 16] = [
    wc(WEDGE_OBLIQUE27, 4, 4),
    wc(WEDGE_OBLIQUE63, 4, 4),
    wc(WEDGE_OBLIQUE117, 4, 4),
    wc(WEDGE_OBLIQUE153, 4, 4),
    wc(WEDGE_HORIZONTAL, 4, 2),
    wc(WEDGE_HORIZONTAL, 4, 6),
    wc(WEDGE_VERTICAL, 2, 4),
    wc(WEDGE_VERTICAL, 6, 4),
    wc(WEDGE_OBLIQUE27, 4, 2),
    wc(WEDGE_OBLIQUE27, 4, 6),
    wc(WEDGE_OBLIQUE153, 4, 2),
    wc(WEDGE_OBLIQUE153, 4, 6),
    wc(WEDGE_OBLIQUE63, 2, 4),
    wc(WEDGE_OBLIQUE63, 6, 4),
    wc(WEDGE_OBLIQUE117, 2, 4),
    wc(WEDGE_OBLIQUE117, 6, 4),
];

/// `ii_weights_1d` (`src/wedge.c:180-183`).
static II_WEIGHTS_1D: [u8; 32] = [
    60, 52, 45, 39, 34, 30, 26, 22, 19, 17, 15, 13, 11, 10, 8, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 2, 2,
    1, 1, 1, 1, 1,
];

/// `wedge_master_border` (`src/wedge.c:215-219`).
static WEDGE_MASTER_BORDER: [[u8; 8]; 3] = [
    [1, 2, 6, 18, 37, 53, 60, 63],
    [1, 4, 11, 27, 46, 58, 62, 63],
    [0, 2, 7, 21, 43, 57, 62, 64],
];

/// `fill` tablosu satırı: (bs, w, h, 422 w/h, 420 w/h, codebook, signs).
type WedgeFill = (
    usize,
    usize,
    usize,
    (usize, usize),
    (usize, usize),
    usize,
    u32,
);

/// Bir maske: arena içindeki ofset + piksel boyutları.
#[derive(Clone, Copy, Debug, Default)]
struct MaskRef {
    off: u32,
    w: u8,
    h: u8,
}

/// Wedge + interintra maske tabloları.
pub(crate) struct Masks {
    arena: Vec<u8>,
    /// `[layout][bs][sign][idx]`.
    wedge: [[[[MaskRef; N_WEDGE_IDX]; 2]; N_BS_ENTRIES]; N_LAYOUTS],
    /// `[layout][bs][mode]`.
    ii: [[[MaskRef; N_II_MODES]; N_BS_ENTRIES]; N_LAYOUTS],
}

impl Masks {
    /// dav1d `dav1d_init_ii_wedge_masks` (`src/wedge.c:207-299`).
    fn build() -> Self {
        let mut arena = Vec::with_capacity(256 * 1024);
        let mut wedge = [[[[MaskRef::default(); N_WEDGE_IDX]; 2]; N_BS_ENTRIES]; N_LAYOUTS];
        let mut ii = [[[MaskRef::default(); N_II_MODES]; N_BS_ENTRIES]; N_LAYOUTS];

        // --- master şablonlar (6 × 64×64) --------------------------------
        let mut master = [[0u8; 64 * 64]; N_WEDGE_DIRECTIONS];
        for y in 0..64 {
            let off = y * 64;
            insert_border(
                &mut master[usize::from(WEDGE_VERTICAL)][off..off + 64],
                &WEDGE_MASTER_BORDER[2],
                32,
            );
        }
        // `src/wedge.c:224-231`: çift/tek satırlar 2'şer satır adımla, `ctr`
        // 48'den 1 azalarak.
        let mut ctr = 48i32;
        let mut y = 0usize;
        while y < 64 {
            let off = y * 64;
            let even = &WEDGE_MASTER_BORDER[1];
            let odd = &WEDGE_MASTER_BORDER[0];
            insert_border(
                &mut master[usize::from(WEDGE_OBLIQUE63)][off..off + 64],
                even,
                ctr,
            );
            insert_border(
                &mut master[usize::from(WEDGE_OBLIQUE63)][off + 64..off + 128],
                odd,
                ctr - 1,
            );
            y += 2;
            ctr -= 1;
        }
        let ob63 = master[usize::from(WEDGE_OBLIQUE63)];
        let vertical = master[usize::from(WEDGE_VERTICAL)];
        transpose(&mut master[usize::from(WEDGE_OBLIQUE27)], &ob63);
        transpose(&mut master[usize::from(WEDGE_HORIZONTAL)], &vertical);
        // `OBLIQUE153`, **transpose edilmiş** `OBLIQUE27`'nin yatay aynasıdır
        // (`src/wedge.c:233-236` sırası önemlidir).
        let ob27 = master[usize::from(WEDGE_OBLIQUE27)];
        hflip(&mut master[usize::from(WEDGE_OBLIQUE117)], &ob63);
        hflip(&mut master[usize::from(WEDGE_OBLIQUE153)], &ob27);

        // --- wedge tabloları (9 blok boyutu) ------------------------------
        // (bs, w, h, 422 w/h, 420 w/h, codebook, signs)
        let fills: [WedgeFill; 9] = [
            (7, 32, 32, (16, 32), (16, 16), 2, 0x7bfb),
            (8, 32, 16, (16, 16), (16, 8), 1, 0x7beb),
            (9, 32, 8, (16, 8), (16, 4), 1, 0x6beb),
            (11, 16, 32, (8, 32), (8, 16), 0, 0x7beb),
            (12, 16, 16, (8, 16), (8, 8), 2, 0x7bfb),
            (13, 16, 8, (8, 8), (8, 4), 1, 0x7beb),
            (15, 8, 32, (4, 32), (4, 16), 0, 0x7aeb),
            (16, 8, 16, (4, 16), (4, 8), 0, 0x7beb),
            (17, 8, 8, (4, 8), (4, 4), 2, 0x7bfb),
        ];
        for (bs, w, h, w422, w420, cb, mut signs) in fills {
            let codebook: &[WedgeCode; 16] = match cb {
                0 => &WEDGE_CODEBOOK_16_HGTW,
                1 => &WEDGE_CODEBOOK_16_HLTW,
                _ => &WEDGE_CODEBOOK_16_HEQW,
            };
            let bi = bs - FIRST_BS;
            for n in 0..N_WEDGE_IDX {
                let sign = signs & 1;
                // 4:4:4 — tek maske, iki işaret de aynı baytları gösterir.
                let off444 = arena.len();
                copy2d(
                    &mut arena,
                    &master[usize::from(codebook[n].direction)],
                    sign,
                    w,
                    h,
                    32 - ((w * usize::from(codebook[n].x_offset)) >> 3),
                    32 - ((h * usize::from(codebook[n].y_offset)) >> 3),
                );
                let r444 = MaskRef {
                    off: off444 as u32,
                    w: w as u8,
                    h: h as u8,
                };
                wedge[0][bi][0][n] = r444;
                wedge[0][bi][1][n] = r444;

                // 4:2:2 — işaret 0 ve 1 ayrı maske.
                let luma = arena[off444..off444 + w * h].to_vec();
                let off422_0 = arena.len();
                init_chroma(&mut arena, &luma, 0, w, h, 0, false);
                wedge[1][bi][0][n] = MaskRef {
                    off: off422_0 as u32,
                    w: w422.0 as u8,
                    h: w422.1 as u8,
                };
                let off422_1 = arena.len();
                init_chroma(&mut arena, &luma, 1, w, h, 0, false);
                wedge[1][bi][1][n] = MaskRef {
                    off: off422_1 as u32,
                    w: w422.0 as u8,
                    h: w422.1 as u8,
                };

                // 4:2:0
                let off420_0 = arena.len();
                init_chroma(&mut arena, &luma, 0, w, h, 1, true);
                wedge[2][bi][0][n] = MaskRef {
                    off: off420_0 as u32,
                    w: w420.0 as u8,
                    h: w420.1 as u8,
                };
                let off420_1 = arena.len();
                init_chroma(&mut arena, &luma, 1, w, h, 1, true);
                wedge[2][bi][1][n] = MaskRef {
                    off: off420_1 as u32,
                    w: w420.0 as u8,
                    h: w420.1 as u8,
                };

                signs >>= 1;
            }
        }

        // --- interintra DC maskesi ---------------------------------------
        let off_dc = arena.len();
        arena.resize(off_dc + 32 * 32, 32);

        // --- interintra DC dışı maskeler ---------------------------------
        // (ad, w, h, step) — `src/wedge.c:265-273`.
        let ii_shapes: [(usize, usize, usize); 9] = [
            (32, 32, 1),
            (16, 32, 1),
            (16, 16, 2),
            (8, 32, 1),
            (8, 16, 2),
            (8, 8, 4),
            (4, 16, 2),
            (4, 8, 4),
            (4, 4, 8),
        ];
        // Her şekil için [v, h, sm] üçlüsü — `build_nondc_ii_masks`.
        let mut nondc: Vec<(usize, usize, [usize; 3])> = Vec::new();
        for (w, h, step) in ii_shapes {
            let off = arena.len();
            build_nondc_ii_masks(&mut arena, w, h, step);
            nondc.push((w, h, [off, off + w * h, off + 2 * w * h]));
        }
        let shape = |w: usize, h: usize| -> usize {
            ii_shapes
                .iter()
                .position(|&(sw, sh, _)| sw == w && sh == h)
                .expect("II şekli tanımlı")
        };
        // `ASSIGN_NONDC_II_OFFSET` (`src/wedge.c:277-294`).
        let assign: [(usize, [usize; 6]); 7] = [
            (7, [32, 32, 16, 32, 16, 16]),
            (8, [32, 32, 16, 16, 16, 16]),
            (11, [16, 32, 8, 32, 8, 16]),
            (12, [16, 16, 8, 16, 8, 8]),
            (13, [16, 16, 8, 8, 8, 8]),
            (16, [8, 16, 4, 16, 4, 8]),
            (17, [8, 8, 4, 8, 4, 4]),
        ];
        // DC maskesi: dav1d'de bütün boyutlar `ii_dc`'ye (32×32, hepsi 32)
        // işaret eder; tüketilen boyut layout/blok boyutuna göredir.
        for &(bs, dims) in &assign {
            for (c, slot) in ii.iter_mut().enumerate() {
                let (w, h) = layout_dims(dims, c);
                slot[bs - FIRST_BS][0] = MaskRef {
                    off: off_dc as u32,
                    w: w as u8,
                    h: h as u8,
                };
            }
        }
        for (bs, dims) in assign {
            for p in 0..3usize {
                for (c, slot) in ii.iter_mut().enumerate() {
                    let (w, h) = layout_dims(dims, c);
                    let si = shape(w, h);
                    slot[bs - FIRST_BS][p + 1] = MaskRef {
                        off: nondc[si].2[p] as u32,
                        w: w as u8,
                        h: h as u8,
                    };
                }
            }
        }

        Self { arena, wedge, ii }
    }

    /// `WEDGE_MASK(c, bs, sign, idx)` — `c`: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0.
    #[must_use]
    pub(crate) fn wedge(&self, c: usize, bs: usize, sign: usize, idx: usize) -> &[u8] {
        let r = self.wedge[c][bs - FIRST_BS][sign][idx];
        &self.arena[r.off as usize..r.off as usize + usize::from(r.w) * usize::from(r.h)]
    }

    /// `II_MASK(c, bs, ...)` — interintra birleştirme maskesi.
    ///
    /// `blend` doğruysa `mode` dalı (`II_DC/II_VERT/II_HOR/II_SMOOTH`), yanlışsa
    /// wedge dalı (`idx`) kullanılır (dav1d `src/wedge.h:97-102`).
    #[must_use]
    pub(crate) fn interintra(
        &self,
        c: usize,
        bs: usize,
        blend: bool,
        mode: usize,
        idx: usize,
    ) -> &[u8] {
        let r = if blend {
            self.ii[c][bs - FIRST_BS][mode]
        } else {
            self.wedge[c][bs - FIRST_BS][0][idx]
        };
        &self.arena[r.off as usize..r.off as usize + usize::from(r.w) * usize::from(r.h)]
    }

    /// Kare başına **tek** kurulum (dav1d: süreç başına bir kez).
    #[must_use]
    pub(crate) fn get() -> &'static Self {
        static MASKS: OnceLock<Masks> = OnceLock::new();
        MASKS.get_or_init(Masks::build)
    }
}

/// Kanonik (oracle) sırasıyla bütün wedge + interintra maske baytları.
///
/// Sıra: `layout` (444, 422, 420) → blok boyutu (artan) → sign → idx; sonra
/// `layout` → bs → ii modu. Oracle aynı sırayı kullanır (bkz.
/// `scripts/oracle/wedge_masks_oracle.c`).
#[must_use]
pub(crate) fn canonical_bytes(m: &Masks) -> Vec<u8> {
    let mut out = Vec::new();
    for c in 0..N_LAYOUTS {
        for bi in 0..N_BS_ENTRIES {
            for sign in 0..2 {
                for idx in 0..N_WEDGE_IDX {
                    out.extend_from_slice(m.wedge(c, bi + FIRST_BS, sign, idx));
                }
            }
        }
    }
    for c in 0..N_LAYOUTS {
        for bi in 0..N_BS_ENTRIES {
            for mode in 0..N_II_MODES {
                let r = m.ii[c][bi][mode];
                if r.w == 0 {
                    continue;
                }
                out.extend_from_slice(m.interintra(c, bi + FIRST_BS, true, mode, 0));
            }
        }
    }
    out
}

/// `ASSIGN_NONDC_II_OFFSET` boyut üçlüsünden layout boyutları.
#[inline]
fn layout_dims(dims: [usize; 6], c: usize) -> (usize, usize) {
    match c {
        0 => (dims[0], dims[1]),
        1 => (dims[2], dims[3]),
        _ => (dims[4], dims[5]),
    }
}

/// Kanonik sırayla bütün wedge + interintra maske baytları (oracle için).
///
/// Sıra `scripts/oracle/wedge_masks_oracle.c` ile birebir aynıdır; oracle
/// çıktısıyla **bayt bayt** karşılaştırılır (oracle doğrulama ölçütü).
#[must_use]
pub fn canonical_mask_bytes() -> Vec<u8> {
    canonical_bytes(Masks::get())
}

/// `insert_border` (`src/wedge.c:81-88`).
fn insert_border(dst: &mut [u8], src: &[u8; 8], ctr: i32) {
    let ctr = ctr as usize;
    if ctr > 4 {
        dst[..ctr - 4].fill(0);
    }
    let dst_off = ctr.max(4) - 4;
    let src_off = 4usize.saturating_sub(ctr);
    let len = (64 - ctr).min(8).min(8 - src_off);
    dst[dst_off..dst_off + len].copy_from_slice(&src[src_off..src_off + len]);
    if ctr < 64 - 4 {
        dst[ctr + 4..64 - 4].fill(64);
    }
}

/// `transpose` (`src/wedge.c:90-95`).
fn transpose(dst: &mut [u8], src: &[u8]) {
    for y in 0..64 {
        for x in 0..64 {
            dst[x * 64 + y] = src[y * 64 + x];
        }
    }
}

/// `hflip` (`src/wedge.c:97-101`).
fn hflip(dst: &mut [u8], src: &[u8]) {
    for y in 0..64 {
        for x in 0..64 {
            dst[y * 64 + 63 - x] = src[y * 64 + x];
        }
    }
}

/// `copy2d` (`src/wedge.c:103-119`) — `dst` arenanın sonuna eklenir.
fn copy2d(
    dst: &mut Vec<u8>,
    src: &[u8],
    sign: u32,
    w: usize,
    h: usize,
    x_off: usize,
    y_off: usize,
) {
    let base = y_off * 64 + x_off;
    if sign != 0 {
        for y in 0..h {
            for x in 0..w {
                dst.push(64 - src[base + y * 64 + x]);
            }
        }
    } else {
        for y in 0..h {
            dst.extend_from_slice(&src[base + y * 64..base + y * 64 + w]);
        }
    }
}

/// `init_chroma` (`src/wedge.c:122-136`).
fn init_chroma(
    dst: &mut Vec<u8>,
    luma: &[u8],
    sign: i32,
    w: usize,
    h: usize,
    ss_ver: usize,
    is_420: bool,
) {
    let _ = is_420;
    let step = 1 + ss_ver;
    let mut row = 0usize;
    let mut out_len = 0usize;
    while row < h {
        let l = &luma[row * w..row * w + w];
        for x in (0..w).step_by(2) {
            let mut sum = i32::from(l[x]) + i32::from(l[x + 1]) + 1;
            if ss_ver != 0 {
                let l2 = &luma[(row + 1) * w..(row + 1) * w + w];
                sum += i32::from(l2[x]) + i32::from(l2[x + 1]) + 1;
            }
            let v = (sum - sign) >> (1 + ss_ver);
            dst.push(v as u8);
            out_len += 1;
        }
        row += step;
    }
    debug_assert_eq!(out_len, (w / 2) * (h / step.max(1)));
}

/// `build_nondc_ii_masks` (`src/wedge.c:178-199`) — `[v, h, sm]` sırasıyla.
fn build_nondc_ii_masks(dst: &mut Vec<u8>, w: usize, h: usize, step: usize) {
    let base = dst.len();
    dst.resize(base + 3 * w * h, 0);
    // mask_v
    for y in 0..h {
        let v = II_WEIGHTS_1D[y * step];
        dst[base + y * w..base + y * w + w].fill(v);
    }
    // mask_h ve mask_sm
    for y in 0..h {
        for x in 0..w {
            dst[base + w * h + y * w + x] = II_WEIGHTS_1D[x * step];
            dst[base + 2 * w * h + y * w + x] = II_WEIGHTS_1D[x.min(y) * step];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maske_tablolari_kurulur_ve_boyutlar_dogru() {
        let m = Masks::get();
        // 4:4:4 32×32 wedge maske = 1024 bayt.
        assert_eq!(m.wedge(0, 7, 0, 0).len(), 32 * 32);
        // 4:2:2 32×32 → 16×32.
        assert_eq!(m.wedge(1, 7, 0, 0).len(), 16 * 32);
        // 4:2:0 32×32 → 16×16.
        assert_eq!(m.wedge(2, 7, 0, 0).len(), 16 * 16);
        // 4:4:4 8×8.
        assert_eq!(m.wedge(0, 17, 1, 15).len(), 8 * 8);
        // Interintra DC maskesi 32×32 ve tamamı 32.
        let dc = m.interintra(0, 7, true, 0, 0);
        assert_eq!(dc.len(), 32 * 32);
        assert!(dc.iter().all(|&v| v == 32));
        // II smooth maskesi köşede en büyük ağırlıkla başlar.
        let sm = m.interintra(0, 7, true, 3, 0);
        assert_eq!(sm[0], II_WEIGHTS_1D[0]);
    }

    #[test]
    fn wedge_maske_degerleri_0_64_araliginda() {
        let m = Masks::get();
        for c in 0..N_LAYOUTS {
            for bs in FIRST_BS..FIRST_BS + 11 {
                for sign in 0..2 {
                    for idx in 0..N_WEDGE_IDX {
                        assert!(m.wedge(c, bs, sign, idx).iter().all(|&v| v <= 64));
                    }
                }
            }
        }
    }

    #[test]
    fn kanonik_baytlar_kararli() {
        let a = canonical_bytes(Masks::get());
        let b = canonical_bytes(Masks::get());
        assert_eq!(a, b);
        assert!(a.len() > 150_000, "kanonik boyut {}", a.len());
    }
}
