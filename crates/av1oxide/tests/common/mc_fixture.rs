//! MC oracle fixture okuyucusu ve koşum yardımcıları — S15.5 pilotu.
//!
//! Fixture biçimi `scripts/oracle/mc_oracle.c` başlığındadır (küçük sonlu).
//! Bu modül **hem** oracle karşılaştırmasını (8(a)) **hem** AVX2 ↔ skaler
//! karşılaştırmasını (8(b)) aynı kodla koşturur: tek fark verilen çekirdek
//! işaretçisidir.
//!
//! ## Neden `unsafe` yok?
//!
//! Test yardımcıları da `#![forbid(unsafe_code)]` taşır; 16bpc düzlemler
//! `Vec<u8>` + `to_le_bytes` ile kurulur. Bu, üretim API'sinin bayt yönelimli
//! sözleşmesiyle aynıdır ve test kodunda `unsafe` gerektirmez.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use av1oxide::dsp::mc::{MC_MAX_H, MC_MAX_W, McBitDepth, McFn};

/// Girdi dosyası büyüsü.
pub const MAGIC_IN: &[u8; 8] = b"AV1OMC__";
/// Beklenen çıktı dosyası büyüsü.
pub const MAGIC_EXPECT: &[u8; 8] = b"AV1OMCEO";

/// dav1d blok boyutu tablosundan genişlikler (`src/levels.h:154-177`).
pub const WIDTHS: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];

/// Her genişlik için dav1d'nin desteklediği yükseklikler.
///
/// `w = 2`, 4-genişlikli blokların 4:2:0 kroma karşılığıdır; diğerleri
/// `src/levels.h:154-177` tablosundan birebir alınmıştır.
pub const HEIGHTS: [&[usize]; 7] = [
    &[2, 4, 8],
    &[4, 8, 16],
    &[4, 8, 16, 32],
    &[4, 8, 16, 32, 64],
    &[8, 16, 32, 64],
    &[16, 32, 64, 128],
    &[64, 128],
];

/// Oracle fixture'ındaki tek bir kayıt (girdi + dav1d C beklenen çıktısı).
#[derive(Clone, Debug)]
pub struct Case {
    /// 8 / 10 / 12.
    pub bit_depth: u8,
    /// Blok genişliği.
    pub w: usize,
    /// Blok yüksekliği.
    pub h: usize,
    /// Yatay alt piksel konumu (1/16).
    pub mx: u8,
    /// Dikey alt piksel konumu (1/16).
    pub my: u8,
    /// Kaynak satır adımı (bayt).
    pub src_stride: usize,
    /// Hedef satır adımı (bayt).
    pub dst_stride: usize,
    /// Dolgu hücresinden (satır −3, sütun −3) başlayan kaynak.
    pub src: Vec<u8>,
    /// Dav1d C beklenen çıktısı (`w*h*bpp`).
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
    assert!(
        input.len() >= 16 && expect.len() >= 16,
        "fixture başlığı kısa"
    );
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
        assert!(ip + 24 <= input.len(), "kayıt {index}: başlık taşması");
        let head = &input[ip..ip + 24];
        let src_stride = rd_u32(head, 8);
        let dst_stride = rd_u32(head, 12);
        let src_len = rd_u32(head, 16);
        let out_len = rd_u32(head, 20);
        assert!(
            ip + 24 + src_len <= input.len(),
            "kayıt {index}: girdi taşması"
        );
        let src = input[ip + 24..ip + 24 + src_len].to_vec();
        ip += 24 + src_len;

        assert!(ep + 4 <= expect.len(), "kayıt {index}: beklenen taşması");
        let exp_len = rd_u32(expect, ep);
        assert_eq!(exp_len, out_len, "kayıt {index}: out_len eşleşmeli");
        assert!(
            ep + 4 + exp_len <= expect.len(),
            "kayıt {index}: beklenen taşması"
        );
        let expect_payload = expect[ep + 4..ep + 4 + exp_len].to_vec();
        ep += 4 + exp_len;

        cases.push(Case {
            bit_depth: head[0],
            w: usize::from(head[1]),
            h: usize::from(head[2]),
            mx: head[3],
            my: head[4],
            src_stride,
            dst_stride,
            src,
            expect: expect_payload,
        });
    }
    assert_eq!(ip, input.len(), "girdi tamamen tüketilmeli");
    assert_eq!(ep, expect.len(), "beklenen dosya tamamen tüketilmeli");
    cases
}

/// Bit derinliği kodu → tip.
#[must_use]
pub fn bit_depth(bd: u8) -> McBitDepth {
    McBitDepth::from_depth(u32::from(bd))
        .unwrap_or_else(|| panic!("bilinmeyen bit derinliği: {bd}"))
}

/// Kaydın hedef dilim uzunluğu: `(h−1)·dst_stride + w·bpp`.
#[must_use]
pub fn dst_len(case: &Case) -> usize {
    let bpp = bit_depth(case.bit_depth).bytes_per_pixel();
    (case.h - 1) * case.dst_stride + case.w * bpp
}

/// Çekirdeği tek bir kayıt üzerinde koşturur ve yalnız `w×h` blok baytlarını
/// (satır satır) döner.
#[must_use]
pub fn run_case(kernel: McFn, case: &Case) -> Vec<u8> {
    let bd = bit_depth(case.bit_depth);
    let bpp = bd.bytes_per_pixel();
    let mut dst = vec![0u8; dst_len(case)];
    // P32: kaynak kuyruğu (`MC_KUYRUK`) test tarafında eklenir; böylece oracle
    // kayıtları (tam sözleşme uzunluğunda) da SIMD yolundan geçer.
    let mut src = case.src.clone();
    src.resize(src.len() + av1oxide::dsp::mc::MC_KUYRUK, 0);
    kernel(
        &mut dst,
        case.dst_stride,
        &src,
        case.src_stride,
        case.w,
        case.h,
        case.mx,
        case.my,
        bd,
    );
    let mut out = Vec::with_capacity(case.w * case.h * bpp);
    for y in 0..case.h {
        let start = y * case.dst_stride;
        out.extend_from_slice(&dst[start..start + case.w * bpp]);
    }
    out
}

/// Hücre anahtarı: (bit derinliği, genişlik).
#[must_use]
pub fn cell_key(case: &Case) -> (u8, usize) {
    (case.bit_depth, case.w)
}

/// Hücre başına kayıt sayısı.
#[must_use]
pub fn cell_counts(cases: &[Case]) -> std::collections::BTreeMap<(u8, usize), usize> {
    let mut map = std::collections::BTreeMap::new();
    for case in cases {
        *map.entry(cell_key(case)).or_insert(0) += 1;
    }
    map
}

/// Fixture kapsamı (8(a)/8(b) kapsam şartlarının ölçümü).
#[derive(Clone, Debug, Default)]
pub struct Coverage {
    /// Bit derinlikleri (8/10/12) görüldü mü.
    pub depths: [bool; 3],
    /// Her genişlik görüldü mü.
    pub widths: [bool; 7],
    /// Her genişlik için görülen en küçük/en büyük yükseklik.
    pub heights: [(usize, usize); 7],
    /// Görülen yükseklik sayısı (genişlik başına).
    pub height_seen: [usize; 7],
    /// mx 0..15 görüldü mü.
    pub mx: [bool; 16],
    /// my 0..15 görüldü mü.
    pub my: [bool; 16],
    /// Yollar: 0 = kopya, 1 = yalnız yatay, 2 = yalnız dikey, 3 = iki boyutlu.
    pub paths: [bool; 4],
    /// Her bit derinliğinde en büyük piksel değeri görüldü mü.
    pub max_pixel: [bool; 3],
    /// Kaynak ve hedef stride'ı farklı bir kayıt var mı.
    pub strides_differ: bool,
}

impl Coverage {
    /// Kayıt kümesinin kapsamını ölçer.
    #[must_use]
    pub fn of(cases: &[Case]) -> Self {
        let mut cov = Self {
            heights: [(usize::MAX, 0); 7],
            ..Self::default()
        };
        for case in cases {
            let depth_index = match case.bit_depth {
                8 => 0,
                10 => 1,
                12 => 2,
                other => panic!("bilinmeyen bit derinliği: {other}"),
            };
            cov.depths[depth_index] = true;
            let width_index = WIDTHS
                .iter()
                .position(|&w| w == case.w)
                .unwrap_or_else(|| panic!("bilinmeyen genişlik: {}", case.w));
            cov.widths[width_index] = true;
            let entry = &mut cov.heights[width_index];
            entry.0 = entry.0.min(case.h);
            entry.1 = entry.1.max(case.h);
            cov.height_seen[width_index] += 1;
            cov.mx[usize::from(case.mx)] = true;
            cov.my[usize::from(case.my)] = true;
            let path = match (case.mx == 0, case.my == 0) {
                (true, true) => 0,
                (false, true) => 1,
                (true, false) => 2,
                (false, false) => 3,
            };
            cov.paths[path] = true;
            if case.src_stride != case.dst_stride {
                cov.strides_differ = true;
            }
            if !cov.max_pixel[depth_index] {
                let bd = bit_depth(case.bit_depth);
                let max = bd.max_pixel();
                let bpp = bd.bytes_per_pixel();
                let seen = case.src.chunks_exact(bpp).any(|p| {
                    if bpp == 1 {
                        i32::from(p[0]) == max
                    } else {
                        i32::from(u16::from_le_bytes([p[0], p[1]])) == max
                    }
                });
                cov.max_pixel[depth_index] = seen;
            }
        }
        cov
    }

    /// Eksik kapsam şartları (boş = tam).
    #[must_use]
    pub fn missing(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, seen) in self.depths.iter().enumerate() {
            if !seen {
                out.push(format!("bit derinliği {}", [8, 10, 12][i]));
            }
        }
        for (i, seen) in self.widths.iter().enumerate() {
            if !seen {
                out.push(format!("genişlik {}", WIDTHS[i]));
            } else {
                let expected_min = HEIGHTS[i][0];
                let expected_max = HEIGHTS[i].last().copied().unwrap_or(0);
                let (min_seen, max_seen) = self.heights[i];
                if min_seen != expected_min {
                    out.push(format!(
                        "w={} en küçük yükseklik: {} (beklenen {expected_min})",
                        WIDTHS[i], min_seen
                    ));
                }
                if max_seen != expected_max {
                    out.push(format!(
                        "w={} en büyük yükseklik: {} (beklenen {expected_max})",
                        WIDTHS[i], max_seen
                    ));
                }
            }
        }
        for (i, seen) in self.mx.iter().enumerate() {
            if !seen {
                out.push(format!("mx={i}"));
            }
        }
        for (i, seen) in self.my.iter().enumerate() {
            if !seen {
                out.push(format!("my={i}"));
            }
        }
        for (i, seen) in self.paths.iter().enumerate() {
            if !seen {
                out.push(
                    ["kopya yolu", "yalnız yatay", "yalnız dikey", "iki boyutlu"][i].to_string(),
                );
            }
        }
        for (i, seen) in self.max_pixel.iter().enumerate() {
            if !seen {
                out.push(format!("{}‑bit en büyük piksel", [8, 10, 12][i]));
            }
        }
        if !self.strides_differ {
            out.push("kaynak/hedef stride farkı".to_string());
        }
        out
    }
}

/// xorshift64* — Python üreteciyle aynı aile, Rust tarafında bağımsız tohum.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Sabit tohumla yeni üreteç.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// 64-bit değer.
    pub fn u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// `0..n` aralığında değer.
    pub fn below(&mut self, n: usize) -> usize {
        (self.u64() % n as u64) as usize
    }
}

/// 8(b) için sabit tohumlu rastgele blok üretir (fixture'sız, doğrudan).
/// Kaynak dilimi okuma bölgesinin sonundan sonra `MC_KUYRUK` bayt sürer (P32).
///
/// Kaynak satır adımı blok genişliğinden **farklı** seçilir; 16bpc'de piksel
/// değerleri `bitdepth_max`'ı aşmaz ve her blokta en az bir en büyük piksel
/// bulunur (uç değer kapsamı).
#[must_use]
pub fn random_case(rng: &mut Rng, bd: McBitDepth, w: usize, h: usize, mx: u8, my: u8) -> Case {
    let bpp = bd.bytes_per_pixel();
    let maxpix = bd.max_pixel() as u16;
    let src_stride = (w + 7) * bpp + rng.below(5) * bpp;
    let dst_stride = w * bpp + rng.below(5) * bpp;
    // P32: kopyasız SIMD sözleşmesi bölge sonundan sonra `MC_KUYRUK` bayt ister
    // (yoksa AVX2 girişi skalere devreder ve AVX2 yolu sınanmaz).
    let src_len = (h + 6) * src_stride + (w + 7) * bpp + av1oxide::dsp::mc::MC_KUYRUK;
    let mut src = vec![0u8; src_len];
    let mut i = 0usize;
    while i < src_len {
        let v = (rng.u64() % (u64::from(maxpix) + 1)) as u16;
        if bpp == 1 {
            src[i] = v as u8;
        } else {
            src[i..i + 2].copy_from_slice(&v.to_le_bytes());
        }
        i += bpp;
    }
    // Uç değerler: en az bir 0 ve bir max piksel (blok penceresi içinde).
    let row = 3usize;
    let col = 3usize;
    let at = row * src_stride + col * bpp;
    if bpp == 1 {
        src[at] = 0;
        src[at + 1] = maxpix as u8;
    } else {
        src[at..at + 2].copy_from_slice(&0u16.to_le_bytes());
        src[at + 2..at + 4].copy_from_slice(&maxpix.to_le_bytes());
    }
    Case {
        bit_depth: bd.depth() as u8,
        w,
        h,
        mx,
        my,
        src_stride,
        dst_stride,
        src,
        expect: Vec::new(),
    }
}

/// Sözleşme dışı boyut üretimini engelleyen derleme zamanı denetimi.
const _: () = assert!(MC_MAX_W == 128 && MC_MAX_H == 128);
