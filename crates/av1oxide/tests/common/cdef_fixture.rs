//! CDEF oracle fixture okuyucusu ve koşum yardımcıları — S15.4 pilotu.
//!
//! Fixture biçimi `scripts/oracle/cdef_oracle.c` başlığındadır (küçük sonlu).
//! Bu modül **hem** oracle karşılaştırmasını (7(a)) **hem** AVX2 ↔ skaler
//! karşılaştırmasını (7(b)) aynı kodla koşturur: tek fark verilen çekirdek
//! işaretçisidir.
//!
//! ## Neden `unsafe` yok?
//!
//! Test yardımcıları da `#![forbid(unsafe_code)]` taşır; 16bpc düzlemler
//! `Vec<u8>` + `to_ne_bytes` ile kurulur. Bu, üretim API'sinin bayt yönelimli
//! sözleşmesiyle aynıdır ve test kodunda `unsafe` gerektirmez.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use av1oxide::dsp::cdef::{
    CdefBitDepth, CdefBlockSize, CdefDirFn, CdefEdgeFlags, CdefFilterFn, CdefFilterParams,
};

/// Girdi dosyası büyüsü.
pub const MAGIC_IN: &[u8; 8] = b"AV1OCDEF";
/// Beklenen çıktı dosyası büyüsü.
pub const MAGIC_EXPECT: &[u8; 8] = b"AV1OCDEO";

/// Oracle fixture'ındaki tek bir kayıt (girdi + dav1d C beklenen çıktısı).
#[derive(Clone, Debug)]
pub struct Case {
    /// 0 = dir, 1 = filter.
    pub kind: u8,
    /// 8 / 10 / 12.
    pub bit_depth: u8,
    /// 0 = 8x8, 1 = 4x8, 2 = 4x4 (`dir` için daima 0).
    pub block_size: u8,
    /// Kenar bayrakları (bit maskesi).
    pub edges: u8,
    /// Öncelikli güç (bit derinliği ölçeği uygulanmış).
    pub pri: u8,
    /// İkincil güç (bit derinliği ölçeği uygulanmış).
    pub sec: u8,
    /// Yön 0..7.
    pub dir: u8,
    /// Sönümleme.
    pub damping: u8,
    /// Satır adımı (bayt).
    pub stride: usize,
    /// Girdi yükü.
    pub input: Vec<u8>,
    /// Dav1d C beklenen çıktısı.
    pub expect: Vec<u8>,
}

fn rd_u32(b: &[u8], off: usize) -> usize {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]) as usize
}

/// Girdi ve beklenen dosyalarını okuyup kayıt kayıt eşler.
///
/// # Panics
///
/// Büyü/sürüm uyuşmazlığı, kesik dosya veya kayıt sayısı farkında panikler
/// (test yardımcısı sözleşmesi: bozuk fixture sessizce geçilmez).
#[must_use]
pub fn read_cases(input: &[u8], expect: &[u8]) -> Vec<Case> {
    assert_eq!(&input[..8], MAGIC_IN, "girdi büyüsü");
    assert_eq!(&expect[..8], MAGIC_EXPECT, "beklenen büyüsü");
    assert_eq!(rd_u32(input, 8), 1, "girdi sürümü");
    assert_eq!(rd_u32(expect, 8), 1, "beklenen sürümü");
    let count = rd_u32(input, 12);
    assert_eq!(count, rd_u32(expect, 12), "kayıt sayısı eşleşmeli");

    let mut cases = Vec::with_capacity(count);
    let mut ip = 16usize;
    let mut ep = 16usize;
    for index in 0..count {
        let h = &input[ip..ip + 20];
        let in_len = rd_u32(input, ip + 12);
        let out_len = rd_u32(input, ip + 16);
        let input_payload = input[ip + 20..ip + 20 + in_len].to_vec();
        ip += 20 + in_len;

        let exp_len = rd_u32(expect, ep);
        assert_eq!(exp_len, out_len, "kayıt {index}: out_len eşleşmeli");
        let expect_payload = expect[ep + 4..ep + 4 + exp_len].to_vec();
        ep += 4 + exp_len;

        cases.push(Case {
            kind: h[0],
            bit_depth: h[1],
            block_size: h[2],
            edges: h[3],
            pri: h[4],
            sec: h[5],
            dir: h[6],
            damping: h[7],
            stride: rd_u32(h, 8),
            input: input_payload,
            expect: expect_payload,
        });
    }
    assert_eq!(ip, input.len(), "girdi tamamen tüketilmeli");
    assert_eq!(ep, expect.len(), "beklenen dosya tamamen tüketilmeli");
    cases
}

/// Bit derinliği kodu → tip.
#[must_use]
pub fn bit_depth(bd: u8) -> CdefBitDepth {
    match bd {
        8 => CdefBitDepth::B8,
        10 => CdefBitDepth::B10,
        12 => CdefBitDepth::B12,
        other => panic!("bilinmeyen bit derinliği: {other}"),
    }
}

/// Blok boyutu kodu → tip.
#[must_use]
pub fn block_size(bs: u8) -> CdefBlockSize {
    match bs {
        0 => CdefBlockSize::W8H8,
        1 => CdefBlockSize::W4H8,
        2 => CdefBlockSize::W4H4,
        other => panic!("bilinmeyen blok boyutu: {other}"),
    }
}

/// Kaydın kenar bayrakları.
#[must_use]
pub fn edges(case: &Case) -> CdefEdgeFlags {
    CdefEdgeFlags::from_bits(case.edges).expect("kenar bitleri 4 bit olmalı")
}

/// Kaydın filtre parametreleri.
#[must_use]
pub fn params(case: &Case) -> CdefFilterParams {
    CdefFilterParams {
        bit_depth: bit_depth(case.bit_depth),
        size: block_size(case.block_size),
        pri_strength: case.pri,
        sec_strength: case.sec,
        dir: case.dir,
        damping: case.damping,
        edges: edges(case),
    }
}

/// Kaydın girdi yerleşimi: (dst, left, top, bottom) dilimleri.
#[must_use]
pub fn slices(case: &Case) -> (&[u8], &[u8], &[u8], &[u8]) {
    let bpp = bit_depth(case.bit_depth).bytes();
    let (w, h) = (
        block_size(case.block_size).w(),
        block_size(case.block_size).h(),
    );
    let dst_len = h * case.stride;
    let left_len = h * 2 * bpp;
    let top_len = 2 * case.stride;
    let bottom_len = 2 * case.stride;
    let total = dst_len + left_len + top_len + bottom_len;
    assert_eq!(
        case.input.len(),
        total,
        "girdi uzunluğu yerleşimle uyuşmalı"
    );
    let left = &case.input[dst_len..dst_len + left_len];
    let top = &case.input[dst_len + left_len..dst_len + left_len + top_len];
    let bottom = &case.input[dst_len + left_len + top_len..];
    let _ = w;
    (&case.input[..dst_len], left, top, bottom)
}

/// Bir `dir` kaydını verilen çekirdekle koşar ve 8 baytlık çıktı zarfını üretir.
#[must_use]
pub fn run_dir(case: &Case, kernel: CdefDirFn) -> Vec<u8> {
    let result = kernel(&case.input, case.stride, bit_depth(case.bit_depth));
    let mut out = vec![0u8; 8];
    out[0] = result.dir;
    out[1..5].copy_from_slice(&result.var.to_le_bytes());
    out
}

/// F3c: kaydın bit derinliği tablosundaki skaler `dir` yuvası.
#[must_use]
#[allow(dead_code)]
pub fn scalar_dir(case: &Case) -> CdefDirFn {
    av1oxide::dsp::cdef::CdefDsp::scalar_for(bit_depth(case.bit_depth)).dir
}

/// F3c: kaydın bit derinliği tablosundaki skaler `fb[blok boyutu]` yuvası.
#[must_use]
#[allow(dead_code)]
pub fn scalar_filter(case: &Case) -> CdefFilterFn {
    av1oxide::dsp::cdef::CdefDsp::scalar_for(bit_depth(case.bit_depth)).fb
        [block_size(case.block_size).index()]
}

/// Bir `filter` kaydını verilen çekirdekle koşar ve sıkışık çıktıyı üretir.
///
/// Girdi `dst` kopyalanır (çekirdek yerinde yazar), çıktı yalnız yazılan
/// `w` sütunu ve `h` satırı kapsar — oracle zarfı da böyledir.
#[must_use]
pub fn run_filter(case: &Case, kernel: CdefFilterFn) -> Vec<u8> {
    let bpp = bit_depth(case.bit_depth).bytes();
    let (w, h) = (
        block_size(case.block_size).w(),
        block_size(case.block_size).h(),
    );
    let (dst_in, left, top, bottom) = slices(case);
    let mut dst = dst_in.to_vec();
    kernel(&mut dst, case.stride, left, top, bottom, params(case));
    let mut out = Vec::with_capacity(h * w * bpp);
    for y in 0..h {
        out.extend_from_slice(&dst[y * case.stride..y * case.stride + w * bpp]);
    }
    out
}

/// Bir kaydı verilen iki çekirdekle koşup **ilk farkı** döndürür (`None` = bit-aynı).
#[must_use]
pub fn first_diff(case: &Case, a: CdefFilterFn, b: CdefFilterFn) -> Option<(usize, u8, u8)> {
    let left_out = run_filter(case, a);
    let right_out = run_filter(case, b);
    left_out
        .iter()
        .zip(right_out.iter())
        .position(|(x, y)| x != y)
        .map(|i| (i, left_out[i], right_out[i]))
}

/// Fixture hücre sayımı: (kind, bit derinliği, blok boyutu) → kayıt sayısı.
#[must_use]
pub fn cell_counts(cases: &[Case]) -> Vec<((u8, u8, u8), usize)> {
    let mut counts: Vec<((u8, u8, u8), usize)> = Vec::new();
    for case in cases {
        let key = (case.kind, case.bit_depth, case.block_size);
        match counts.iter_mut().find(|(k, _)| *k == key) {
            Some((_, n)) => *n += 1,
            None => counts.push((key, 1)),
        }
    }
    counts.sort_by_key(|(k, _)| *k);
    counts
}

/// Fixture'ın **kapsam** özeti (7. maddedeki kapsam şartlarının denetimi için).
#[derive(Clone, Copy, Debug, Default)]
pub struct Coverage {
    /// Görülen yönler.
    pub dirs: [bool; 8],
    /// Görülen kenar birleşimleri (16).
    pub edges: [bool; 16],
    /// Görülen blok boyutları.
    pub sizes: [bool; 3],
    /// Görülen bit derinlikleri (8/10/12).
    pub depths: [bool; 3],
    /// Görülen **temel** öncelikli güçler (ölçeksiz, 0..15).
    pub pri_base: [bool; 16],
    /// Görülen **temel** ikincil güçler (0..4).
    pub sec_base: [bool; 5],
    /// Görülen sönümlemeler (0..7).
    pub damping: [bool; 8],
    /// En büyük piksel değerinin geçtiği bit derinlikleri.
    pub max_pixel_seen: [bool; 3],
    /// P29: MINMAX alt sınır kırpmasının etkin olduğu piksel sayısı (8/10/12).
    pub alt_kirpma: [usize; 3],
}

impl Coverage {
    /// Kayıt kümesinden kapsam çıkarır.
    #[must_use]
    pub fn of(cases: &[Case]) -> Self {
        let mut cov = Self::default();
        for case in cases {
            let bd = bit_depth(case.bit_depth);
            let shift = bd.min_8();
            cov.dirs[usize::from(case.dir & 7)] = true;
            cov.edges[usize::from(case.edges & 0b1111)] = true;
            cov.damping[usize::from(case.damping & 7)] = true;
            cov.depths[match case.bit_depth {
                8 => 0,
                10 => 1,
                _ => 2,
            }] = true;
            if case.kind == 1 {
                cov.sizes[usize::from(case.block_size.min(2))] = true;
                cov.pri_base[usize::from(case.pri >> shift)] = true;
                cov.sec_base[usize::from(case.sec >> shift)] = true;
            }
            let bi = match case.bit_depth {
                8 => 0,
                10 => 1,
                _ => 2,
            };
            if has_max_pixel(case) {
                cov.max_pixel_seen[bi] = true;
            }
            cov.alt_kirpma[bi] += alt_kirpma_pikselleri(case);
        }
        cov
    }

    /// İnsan okunur eksik listesi (boş = tam).
    #[must_use]
    pub fn missing(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.dirs.iter().filter(|x| **x).count() != 8 {
            out.push("8 yönün hepsi".into());
        }
        if self.edges.iter().filter(|x| **x).count() != 16 {
            out.push("16 kenar birleşiminin hepsi".into());
        }
        if self.sizes.iter().filter(|x| **x).count() != 3 {
            out.push("üç blok boyutu".into());
        }
        if self.depths.iter().filter(|x| **x).count() != 3 {
            out.push("8/10/12-bit".into());
        }
        if self.pri_base.iter().filter(|x| **x).count() != 16 {
            out.push("öncelikli güç 0..15".into());
        }
        if !(self.sec_base[0] && self.sec_base[1] && self.sec_base[2] && self.sec_base[4]) {
            out.push("ikincil güç {0,1,2,4}".into());
        }
        if !(self.damping[3] && self.damping[6]) {
            out.push("sönümleme uçları {3,6}".into());
        }
        if self.max_pixel_seen.iter().filter(|x| **x).count() != 3 {
            out.push("8/10/12-bit en büyük piksel".into());
        }
        if self.alt_kirpma.contains(&0) {
            out.push(format!(
                "8/10/12-bit MINMAX alt sınır kırpması (piksel {:?})",
                self.alt_kirpma
            ));
        }
        out
    }
}

/// dav1d `dav1d_cdef_directions` (`src/tables.c:400-413`), 12 adımlı ara tampon
/// ofsetleri; indeks `dir + 2` = yön, `dir + 4` = yön + 2, `dir` = yön − 2.
const YONLER: [[isize; 2]; 12] = [
    [12, 24],
    [12, 23],
    [-11, -22],
    [1, -10],
    [1, 2],
    [1, 14],
    [13, 26],
    [12, 25],
    [12, 24],
    [12, 23],
    [-11, -22],
    [1, -10],
];

/// P29 — kayıtta MINMAX **alt** sınır kırpmasının etkin olduğu piksel sayısı.
///
/// dav1d `cdef_filter_block_c` (`src/cdef_tmpl.c:56-169`) aritmetiğinin üretim
/// kodundan bağımsız bir kopyasıyla hesaplanır: yalnız `pri && sec` kolunda,
/// kırpılmamış değer `px + ((sum − (sum < 0) + 8) >> 4)` tap'ların en küçüğünden
/// (ve `px`'ten) küçükse sayılır. Bu piksellerde alt sınırı atlayan bir gövde
/// oracle'dan kaçamaz. Eksik kenar (dav1d `INT16_MIN` dolgusu) `constrain`'e 0
/// katar ve `min`'i değiştirmez; burada o tap atlanır.
#[must_use]
pub fn alt_kirpma_pikselleri(case: &Case) -> usize {
    if case.kind != 1 || case.pri == 0 || case.sec == 0 {
        return 0;
    }
    let bd = bit_depth(case.bit_depth);
    let bpp = bd.bytes();
    let (w, h) = (
        block_size(case.block_size).w(),
        block_size(case.block_size).h(),
    );
    let (dst, left, top, bottom) = slices(case);
    let oku = |buf: &[u8], i: usize| -> i32 {
        if bpp == 1 {
            i32::from(buf[i])
        } else {
            i32::from(u16::from_ne_bytes([buf[2 * i], buf[2 * i + 1]]))
        }
    };
    let sp = case.stride / bpp;
    let e = case.edges;
    let (sol, sag, ust, alt) = (e & 1 != 0, e & 2 != 0, e & 4 != 0, e & 8 != 0);
    // 12 × 12 ara tampon (dav1d `tmp_buf`), `None` = kenar yok.
    let mut tmp = [None::<i32>; 144];
    for ty in 0..h + 4 {
        for tx in 0..w + 4 {
            let (y, x) = (ty as isize - 2, tx as isize - 2);
            if (x < 0 && !sol)
                || (x >= w as isize && !sag)
                || (y < 0 && !ust)
                || (y >= h as isize && !alt)
            {
                continue;
            }
            let (xu, yu) = ((x + 2) as usize, y as usize);
            tmp[ty * 12 + tx] = Some(if y < 0 {
                oku(top, (y + 2) as usize * sp + xu)
            } else if y >= h as isize {
                oku(bottom, (y - h as isize) as usize * sp + xu)
            } else if x < 0 {
                oku(left, yu * 2 + xu)
            } else {
                oku(dst, yu * sp + x as usize)
            });
        }
    }
    let ulog2 = |v: i32| 31 - v.leading_zeros() as i32;
    let pri = i32::from(case.pri);
    let sec = i32::from(case.sec);
    let damping = i32::from(case.damping);
    let pri_tap = 4 - ((pri >> bd.min_8()) & 1);
    let pri_shift = (damping - ulog2(pri)).max(0);
    let sec_shift = (damping - ulog2(sec)).max(0);
    let constrain = |diff: i32, thr: i32, sh: i32| {
        let a = diff.abs();
        a.min((thr - (a >> sh)).max(0)) * diff.signum()
    };
    let d = usize::from(case.dir & 7);
    let mut n = 0;
    for y in 0..h {
        for x in 0..w {
            let base = ((y + 2) * 12 + x + 2) as isize;
            let at = |off: isize| tmp[(base + off) as usize];
            let px = at(0).expect("blok pikseli");
            let (mut sum, mut mn) = (0, px);
            let mut tap = pri_tap;
            let yon = YONLER[d + 2].into_iter().zip(YONLER[d + 4]).zip(YONLER[d]);
            for (k, ((o, o2), o3)) in yon.enumerate() {
                for p in [at(o), at(-o)].into_iter().flatten() {
                    sum += tap * constrain(p - px, pri, pri_shift);
                    mn = mn.min(p);
                }
                tap = (tap & 3) | 2;
                let sec_tap = 2 - k as i32;
                for o in [o2, o3] {
                    for s in [at(o), at(-o)].into_iter().flatten() {
                        sum += sec_tap * constrain(s - px, sec, sec_shift);
                        mn = mn.min(s);
                    }
                }
            }
            if px + ((sum - i32::from(sum < 0) + 8) >> 4) < mn {
                n += 1;
            }
        }
    }
    n
}

/// Kaydın **okunan** bölgesinde en büyük piksel değeri var mı?
fn has_max_pixel(case: &Case) -> bool {
    let bd = bit_depth(case.bit_depth);
    let max = bd.max_pixel();
    let bpp = bd.bytes();
    let bytes = &case.input;
    let mut i = 0;
    while i + bpp <= bytes.len() {
        let v = if bpp == 1 {
            u16::from(bytes[i])
        } else {
            u16::from_ne_bytes([bytes[i], bytes[i + 1]])
        };
        if v == max {
            return true;
        }
        i += bpp;
    }
    false
}

/// Sabit tohumlu, Python üreteciyle **aynı** xorshift64* PRNG'i (test içi üretim).
pub struct Rng(u64);

impl Rng {
    /// Tohumla kurar (0 tohumu sabit bir sabitle değiştirilir).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// 64-bit sözde rastgele değer.
    pub fn u64(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// `[0, n)` aralığında değer.
    pub fn below(&mut self, n: usize) -> usize {
        (self.u64() % n as u64) as usize
    }

    /// `[0, n]` aralığında değer.
    pub fn upto(&mut self, n: usize) -> usize {
        (self.u64() % (n as u64 + 1)) as usize
    }
}

/// Test için rastgele bir `Case` üretir (oracle'sız 7(b)/7(c) karşılaştırmaları).
///
/// Üretilen parametreler gerçek AV1 sözleşmesine uyar: `damping >= ulog2(sec)` ve
/// `!(pri == 0 && sec == 0)`.
#[must_use]
pub fn random_case(rng: &mut Rng, kind: u8, bd_code: u8, size_code: u8, pattern: u8) -> Case {
    let bd = bit_depth(bd_code);
    let bpp = bd.bytes();
    let max = usize::from(bd.max_pixel());
    let (w, h) = if kind == 0 {
        (8, 8)
    } else {
        let size = block_size(size_code);
        (size.w(), size.h())
    };
    let stride = if kind == 0 { 8 * bpp } else { (w + 4) * bpp };
    let edge_bits = if kind == 0 { 0 } else { rng.below(16) as u8 };

    let (pri, sec, damping) = if kind == 0 {
        (0u8, 0u8, 0u8)
    } else {
        let base_sec = [0usize, 1, 2, 4][rng.below(4)];
        let mut base_pri = rng.below(16);
        if base_pri == 0 && base_sec == 0 {
            base_pri = 1;
        }
        let sec = (base_sec << bd.min_8()) as u8;
        let pri = (base_pri << bd.min_8()) as u8;
        let ulog = 31 - (sec as u32).max(1).leading_zeros();
        let candidates: Vec<u8> = (3u8..=6).filter(|d| u32::from(*d) >= ulog).collect();
        let damping = candidates[rng.below(candidates.len())];
        (pri, sec, damping)
    };

    // Piksel desenleri: 0 = rastgele, 1 = {0,max}, 2 = yürüyüş, 3 = en büyük.
    let fill = |rng: &mut Rng, n: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(n * bpp);
        let mut value = rng.upto(max);
        for _ in 0..n {
            let v = match pattern {
                1 => {
                    if rng.below(2) == 0 {
                        0
                    } else {
                        max
                    }
                }
                2 => {
                    value = if rng.below(2) == 0 {
                        value.saturating_add(1).min(max)
                    } else {
                        value.saturating_sub(1)
                    };
                    value
                }
                3 => max,
                _ => rng.upto(max),
            };
            out.extend_from_slice(&(v as u16).to_ne_bytes()[..bpp]);
        }
        out
    };

    let mut payload = fill(rng, h * stride / bpp);
    payload.extend(fill(rng, h * 2));
    payload.extend(fill(rng, 2 * stride / bpp));
    payload.extend(fill(rng, 2 * stride / bpp));

    Case {
        kind,
        bit_depth: bd_code,
        block_size: if kind == 0 { 0 } else { size_code },
        edges: edge_bits,
        pri,
        sec,
        dir: rng.below(8) as u8,
        damping,
        stride,
        input: payload,
        expect: Vec::new(),
    }
}

/// P30 — bir ISA düzeyinin bu makinede çağrılabilir CDEF gövdeleri (bir bit
/// derinliği için): `dir` ve `fb` (üç blok boyutu tek gövdededir).
#[derive(Clone, Copy)]
pub struct IsaGovde {
    /// Uygulama adı (`ssse3`, `sse4`, `avx2`, `avx512icl`; dispatch matrisi `tier`).
    pub ad: &'static str,
    /// `dir` gövdesi (bu düzeyde dav1d kaydı yoksa `None`).
    pub dir: Option<CdefDirFn>,
    /// `fb` gövdesi (bu düzeyde dav1d kaydı yoksa `None`).
    pub fb: Option<CdefFilterFn>,
}

/// P30 — `bd` bit derinliğinin ISA gövdeleri, adaylardan (varsayılan olup
/// olmadığına bakılmaz; gövde ölçümle atanmasa da sınanır). İkinci değer
/// adayı olup belirteci üretilmeyen (bu makinede ya da maskede kapalı)
/// düzeylerdir.
#[must_use]
pub fn isa_govdeleri(bd: u8) -> (Vec<IsaGovde>, Vec<&'static str>) {
    use av1oxide::dsp::{DspDispatch, ISA_CANDIDATES, IsaEntry};
    let dsp = DspDispatch::select(av1oxide::cpu::CpuFlags::detect());
    let mut var = Vec::new();
    let mut yok = Vec::new();
    for ad in ["ssse3", "sse4", "avx2", "avx512icl"] {
        let adaylar: Vec<_> = ISA_CANDIDATES
            .iter()
            .filter(|c| c.family == "cdef" && c.bit_depth == bd && c.implementation == ad)
            .collect();
        if adaylar.is_empty() {
            continue;
        }
        let mut g = IsaGovde {
            ad,
            dir: None,
            fb: None,
        };
        let mut eksik = false;
        for c in adaylar {
            match dsp.candidate_entry(c) {
                Some(IsaEntry::CdefDir(f)) => g.dir = Some(f),
                Some(IsaEntry::CdefFb(_, f)) => g.fb = Some(f),
                Some(_) => {}
                None => eksik = true,
            }
        }
        if eksik {
            yok.push(ad);
        } else {
            var.push(g);
        }
    }
    (var, yok)
}
