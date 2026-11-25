//! S2b.4 — MC aile **sentinel / over-read** testleri (kabul ölçütü 4).
//!
//! Üç araç (SMD-07, `tests/common/sentinel.rs`):
//!
//! 1. [`SentinelBuffer`] — hedef tamponun önüne/arkasına koruyucu koyar;
//!    çekirdek **hedef dışına yazmamalıdır**.
//! 2. [`OverReadProbe`] — kaynak, **belgelenen okuma penceresi kadar** bir
//!    veri bölgesi olarak verilir ve iki farklı padding ile koşturulur;
//!    pencere dışından bir bayt okunursa çıktı değişir. Pencere tanımı
//!    `crates/av1oxide/src/dsp/scalar/mc.rs` modül belgesindedir (test (b)
//!    tablosu).
//! 3. Negatif kontrol — **kasıtlı** bir bayt taşan sarmalayıcı, aynı düzenekle
//!    yakalanır; bu, ölçümün duyarlı olduğunun kanıtıdır.

#![forbid(unsafe_code)]
#![allow(clippy::chunks_exact_to_as_chunks, clippy::while_let_loop)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::McDsp;
use av1oxide::frame::BitDepth;
use common::sentinel::{OverReadProbe, SentinelBuffer};

const BPP_BD: [BitDepth; 3] = [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve];

fn bdmax(bd: BitDepth) -> u16 {
    ((1u32 << bd.bits()) - 1) as u16
}

fn bpp(bd: BitDepth) -> usize {
    bd.bytes_per_sample()
}

/// 16bpc'de piksel yüklerini `bitdepth_max` aralığına indirger ve maske
/// baytlarını 0..64 aralığına çeker (sözleşme: geçerli girdi).
fn sanitize(spec: &Spec, s: &mut [u8]) {
    let b = bpp(spec.bd);
    let n = spec.w * spec.h;
    let pixels_len = match spec.fam {
        FAM_PUT | FAM_PREP | FAM_PUT_SCALED | FAM_PREP_SCALED | FAM_WARP | FAM_WARP_T
        | FAM_EMU_EDGE => s.len(),
        FAM_BLEND => 2 * n * b,
        FAM_BLEND_V | FAM_BLEND_H => 2 * n * b,
        _ => 0,
    };
    if b == 2 {
        for chunk in s[..pixels_len].chunks_exact_mut(2) {
            let v = u16::from_le_bytes([chunk[0], chunk[1]]) & bdmax(spec.bd);
            chunk.copy_from_slice(&v.to_le_bytes());
        }
    }
    match spec.fam {
        FAM_MASK => {
            for byte in &mut s[n * 4..n * 4 + n] {
                *byte %= 65;
            }
        }
        FAM_BLEND => {
            for byte in &mut s[2 * n * b..2 * n * b + n] {
                *byte %= 65;
            }
        }
        FAM_W_MASK => {} // maske çıktıdır, girdi değil
        _ => {}
    }
}

/// Bir aile çağrısının tam sözleşmesi (tek yapı, tekrar yok).
#[derive(Clone, Copy)]
struct Spec {
    fam: u32,
    sub: u32,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    aux: [i32; 4],
    bd: BitDepth,
}

const FAM_PUT: u32 = 0;
const FAM_PUT_SCALED: u32 = 1;
const FAM_PREP: u32 = 2;
const FAM_PREP_SCALED: u32 = 3;
const FAM_AVG: u32 = 4;
const FAM_W_AVG: u32 = 5;
const FAM_MASK: u32 = 6;
const FAM_W_MASK: u32 = 7;
const FAM_BLEND: u32 = 8;
const FAM_BLEND_V: u32 = 9;
const FAM_BLEND_H: u32 = 10;
const FAM_WARP: u32 = 11;
const FAM_WARP_T: u32 = 12;
const FAM_EMU_EDGE: u32 = 13;

/// Aile adı (hata mesajları için).
fn name(fam: u32) -> &'static str {
    match fam {
        0 => "put",
        1 => "put_scaled",
        2 => "prep",
        3 => "prep_scaled",
        4 => "avg",
        5 => "w_avg",
        6 => "mask",
        7 => "w_mask",
        8 => "blend",
        9 => "blend_v",
        10 => "blend_h",
        11 => "warp8x8",
        12 => "warp8x8t",
        _ => "emu_edge",
    }
}

impl Spec {
    /// Ölçekli kaynak penceresinin sütun sayısı (belgelenen pencere).
    fn scaled_cols(&self) -> usize {
        ((self.mx + (self.w as i32 - 1) * self.dx) >> 10) as usize + 8
    }

    /// **Belgelenen okuma penceresi** (bayt) — çekirdeğin dokunmasına izin
    /// verilen en küçük kaynak dilimi. Modül belgesindeki tabloyla birebir.
    fn src_len(&self) -> usize {
        let b = bpp(self.bd);
        match self.fam {
            // satır 0..h+6, sütun 0..w+6 → (h+6)·stride + (w+7)·bpp
            FAM_PUT | FAM_PREP => {
                let stride = (self.w + 7) * b;
                (self.h + 6) * stride + (self.w + 7) * b
            }
            // satır 0..((my+(h−1)·dy)>>10)+7, sütun 0..((mx+(w−1)·dx)>>10)+8
            FAM_PUT_SCALED | FAM_PREP_SCALED => {
                // Satır 0..base+7 **dâhil** → base+8 satır.
                let rows = ((self.my + (self.h as i32 - 1) * self.dy) >> 10) as usize + 8;
                rows * self.scaled_cols() * b
            }
            // satır 0..h+7, sütun 0..w+7
            FAM_WARP | FAM_WARP_T => 15 * 15 * b,
            FAM_EMU_EDGE => {
                let iw = self.aux[0] as usize;
                let ih = self.aux[1] as usize;
                (ih - 1) * iw * b + iw * b
            }
            _ => 0,
        }
    }

    /// Çekirdeğe verilen **tek bitişik kaynak yükü** (bayt). Üç aile için
    /// birden fazla düzlem içerir (`avg`: tmp1‖tmp2, `mask`: tmp1‖tmp2‖maske,
    /// `blend`: dst‖tmp‖maske).
    fn payload_len(&self) -> usize {
        let b = bpp(self.bd);
        let n = self.w * self.h;
        match self.fam {
            FAM_PUT | FAM_PREP | FAM_PUT_SCALED | FAM_PREP_SCALED | FAM_WARP | FAM_WARP_T
            | FAM_EMU_EDGE => self.src_len(),
            FAM_AVG | FAM_W_AVG | FAM_W_MASK => n * 2 * 2,
            FAM_MASK => n * 2 * 2 + n,
            FAM_BLEND => 2 * n * b + n,
            FAM_BLEND_V | FAM_BLEND_H => 2 * n * b,
            _ => 0,
        }
    }

    fn dst_stride(&self) -> usize {
        (self.w + 3) * bpp(self.bd)
    }

    /// Çıktı baytı: `dst` satırları (+ `w_mask` maskesi).
    fn out_len(&self) -> usize {
        let b = bpp(self.bd);
        match self.fam {
            FAM_PREP | FAM_PREP_SCALED => self.w * self.h * 2,
            FAM_WARP_T => 8 * 8 * 2,
            FAM_W_MASK => {
                let ss_hor = self.sub != 0;
                let ss_ver = self.sub == 2;
                self.w * self.h * b
                    + (self.w >> usize::from(ss_hor)) * (self.h >> usize::from(ss_ver))
            }
            _ => self.w * self.h * b,
        }
    }

    /// Çekirdeği koşturur: `src` belgelenen pencere, `out` çıktı.
    fn run(&self, src: &[u8], out: &mut [u8]) {
        let bd = self.bd;
        let dsp = McDsp::scalar_for(bd);
        let max = bdmax(bd);
        let b = bpp(bd);
        let w = self.w;
        let h = self.h;
        let stride = self.dst_stride();
        match self.fam {
            FAM_PUT | FAM_PUT_SCALED => {
                let mut dst = vec![0u8; (h - 1) * stride + w * b];
                if self.fam == FAM_PUT {
                    (dsp.mc[self.sub as usize])(
                        &mut dst,
                        stride,
                        src,
                        (w + 7) * b,
                        w,
                        h,
                        self.mx as u8,
                        self.my as u8,
                        bd,
                        max,
                    );
                } else {
                    let cols = self.scaled_cols();
                    (dsp.mc_scaled[self.sub as usize])(
                        &mut dst,
                        stride,
                        src,
                        cols * b,
                        w,
                        h,
                        self.mx,
                        self.my,
                        self.dx,
                        self.dy,
                        bd,
                        max,
                    );
                }
                for y in 0..h {
                    out[y * w * b..(y + 1) * w * b]
                        .copy_from_slice(&dst[y * stride..y * stride + w * b]);
                }
            }
            FAM_PREP | FAM_PREP_SCALED => {
                let mut tmp = vec![0i16; w * h];
                // Kaynak satır adımı: 8-tap için (w+7)·bpp, scaled için
                // pencere genişliği (dav1d `src_stride` parametresi).
                let sstride = if self.fam == FAM_PREP {
                    (w + 7) * b
                } else {
                    self.scaled_cols() * b
                };
                if self.fam == FAM_PREP {
                    (dsp.mct[self.sub as usize])(
                        &mut tmp,
                        src,
                        sstride,
                        w,
                        h,
                        self.mx as u8,
                        self.my as u8,
                        bd,
                        max,
                    );
                } else {
                    (dsp.mct_scaled[self.sub as usize])(
                        &mut tmp, src, sstride, w, h, self.mx, self.my, self.dx, self.dy, bd, max,
                    );
                }
                for (i, v) in tmp.iter().enumerate() {
                    out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            FAM_AVG | FAM_W_AVG | FAM_MASK | FAM_W_MASK => {
                let n = w * h;
                let tmp1: Vec<i16> = src[..n * 2]
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let tmp2: Vec<i16> = src[n * 2..n * 4]
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let mut dst = vec![0u8; (h - 1) * stride + w * b];
                match self.fam {
                    FAM_AVG => {
                        (dsp.avg)(&mut dst, stride, &tmp1, &tmp2, w, h, bd, max);
                    }
                    FAM_W_AVG => {
                        (dsp.w_avg)(&mut dst, stride, &tmp1, &tmp2, w, h, 7, bd, max);
                    }
                    FAM_MASK => {
                        let mask = &src[n * 4..n * 4 + n];
                        (dsp.mask)(&mut dst, stride, &tmp1, &tmp2, w, h, mask, bd, max);
                    }
                    _ => {
                        let ss_hor = self.sub != 0;
                        let ss_ver = self.sub == 2;
                        let mw = w >> usize::from(ss_hor);
                        let mh = h >> usize::from(ss_ver);
                        let mut mask = vec![0u8; mw * mh];
                        (dsp.w_mask[self.sub as usize])(
                            &mut dst,
                            stride,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            &mut mask,
                            self.aux[0],
                            ss_hor,
                            ss_ver,
                            bd,
                            max,
                        );
                        for y in 0..h {
                            out[y * w * b..(y + 1) * w * b]
                                .copy_from_slice(&dst[y * stride..y * stride + w * b]);
                        }
                        let off = w * h * b;
                        out[off..off + mw * mh].copy_from_slice(&mask);
                        return;
                    }
                }
                for y in 0..h {
                    out[y * w * b..(y + 1) * w * b]
                        .copy_from_slice(&dst[y * stride..y * stride + w * b]);
                }
            }
            FAM_BLEND | FAM_BLEND_V | FAM_BLEND_H => {
                // Kaynak: dst (w·h) ++ tmp (w·h) ++ maske (w·h).
                let n = w * h;
                let mut dst = vec![0u8; (h - 1) * stride + w * b];
                for y in 0..h {
                    let s = y * w * b;
                    dst[y * stride..y * stride + w * b].copy_from_slice(&src[s..s + w * b]);
                }
                let tmp = &src[n * b..2 * n * b];
                match self.fam {
                    FAM_BLEND => {
                        let mask = &src[2 * n * b..2 * n * b + n];
                        (dsp.blend)(&mut dst, stride, tmp, w, h, mask, bd, max);
                    }
                    FAM_BLEND_V => (dsp.blend_v)(&mut dst, stride, tmp, w, h, bd, max),
                    _ => (dsp.blend_h)(&mut dst, stride, tmp, w, h, bd, max),
                }
                for y in 0..h {
                    out[y * w * b..(y + 1) * w * b]
                        .copy_from_slice(&dst[y * stride..y * stride + w * b]);
                }
            }
            FAM_WARP | FAM_WARP_T => {
                let abcd = [
                    self.aux[0] as i16,
                    self.aux[1] as i16,
                    self.aux[2] as i16,
                    self.aux[3] as i16,
                ];
                if self.fam == FAM_WARP {
                    let mut dst = vec![0u8; (h - 1) * stride + w * b];
                    (dsp.warp8x8)(
                        &mut dst,
                        stride,
                        src,
                        15 * b,
                        &abcd,
                        self.mx,
                        self.my,
                        bd,
                        max,
                    );
                    for y in 0..8 {
                        out[y * 8 * b..(y + 1) * 8 * b]
                            .copy_from_slice(&dst[y * stride..y * stride + 8 * b]);
                    }
                } else {
                    let mut tmp = vec![0i16; 64];
                    (dsp.warp8x8t)(&mut tmp, 8, src, 15 * b, &abcd, self.mx, self.my, bd, max);
                    for (i, v) in tmp.iter().enumerate() {
                        out[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
            FAM_EMU_EDGE => {
                let iw = self.aux[0] as usize;
                let mut dst = vec![0u8; (h - 1) * stride + w * b];
                (dsp.emu_edge)(
                    w,
                    h,
                    iw,
                    self.aux[1] as usize,
                    self.aux[2],
                    self.aux[3],
                    &mut dst,
                    stride,
                    src,
                    iw * b,
                    bd,
                    max,
                );
                for y in 0..h {
                    out[y * w * b..(y + 1) * w * b]
                        .copy_from_slice(&dst[y * stride..y * stride + w * b]);
                }
            }
            other => panic!("aile {other}"),
        }
    }
}

/// Test edilecek aile kümesi (her aile en az bir kez, iki bit derinliği).
fn specs(bd: BitDepth) -> Vec<Spec> {
    let mut v = Vec::new();
    for fam in [
        FAM_PUT,
        FAM_PUT_SCALED,
        FAM_PREP,
        FAM_PREP_SCALED,
        FAM_AVG,
        FAM_W_AVG,
        FAM_MASK,
        FAM_W_MASK,
        FAM_BLEND,
        FAM_BLEND_V,
        FAM_BLEND_H,
        FAM_WARP,
        FAM_WARP_T,
        FAM_EMU_EDGE,
    ]
    .iter()
    {
        let sub = if *fam == FAM_W_MASK {
            2
        } else if matches!(*fam, FAM_PREP | FAM_PREP_SCALED | FAM_PUT | FAM_PUT_SCALED) {
            5 // SHARP (8-tap tablosunun orta birleşimi)
        } else {
            0
        };
        let (w, h) = (8, 8);
        v.push(Spec {
            fam: *fam,
            sub,
            w,
            h,
            mx: 7,
            my: 5,
            dx: 1024,
            dy: 1024,
            aux: if *fam == FAM_EMU_EDGE {
                // iw, ih, x, y — blok 8×8, referans 16×16, sol/üst taşma.
                [16, 16, -2, -1]
            } else {
                [0, 0, 0, 0]
            },
            bd,
        });
    }
    v
}

/// (b1) — **hedef dışına yazma yok**: her ailede hedef tampon sentinel'li.
#[test]
fn hedefe_tasma_yok() {
    for bd in BPP_BD {
        for spec in specs(bd) {
            let out_len = spec.out_len();
            let mut buf = SentinelBuffer::with_defaults(out_len);
            buf.fill_data(0);
            let src_len = spec.payload_len();
            let mut src = vec![0u8; src_len];
            // Kaynak deseni: deterministik, sıfırdan farklı.
            for (i, b) in src.iter_mut().enumerate() {
                *b = ((i * 37 + 11) & 0xFF) as u8;
            }
            sanitize(&spec, &mut src);
            let out = buf.data_mut();
            spec.run(&src, out);
            buf.verify().unwrap_or_else(|v| {
                panic!(
                    "{}-bit {} (w={},h={}): hedef dışına yazma: {:?} indeks {} \
                     (beklenen {:#04x}, bulunan {:#04x})",
                    bd.bits(),
                    name(spec.fam),
                    spec.w,
                    spec.h,
                    v.side,
                    v.index,
                    v.expected,
                    v.found
                )
            });
        }
    }
    println!("[sentinel] hedefe taşma yok: 14 aile × 3 bit derinliği");
}

/// (b2) — **belgelenen okuma penceresi dışından okuma yok**.
///
/// Kaynak tam olarak belgelenen pencere uzunluğundadır; iki farklı padding
/// deseniyle koşturulur ve çıktılar bit-aynı olmalıdır.
#[test]
fn kaynak_penceresi_disina_okuma_yok() {
    for bd in BPP_BD {
        for mut spec in specs(bd) {
            // emu_edge: referans penceresi iw×ih; taşan erişim için küçük
            // referans + büyük blok (kenar genişletme yolu).
            if spec.fam == FAM_EMU_EDGE {
                spec.aux = [16, 16, -3, -2];
                spec.w = 12;
                spec.h = 10;
            }
            let (src_len, out_len) = (spec.payload_len(), spec.out_len());
            let probe = OverReadProbe::new(src_len, 64, out_len);
            let sp = spec;
            let sonuc = probe.run(|src, out| {
                // `src` padding'li tam tampondur; çekirdek **yalnız veri
                // bölgesini** görür (belgelenen pencere).
                let mut s = src[probe.data_offset()..probe.data_offset() + src_len].to_vec();
                sanitize(&sp, &mut s);
                sp.run(&s, out);
            });
            match sonuc {
                Ok(_) => {}
                Err(e) => panic!(
                    "{}-bit {} (w={},h={}): kaynak penceresi dışından okuma — \
                     padding A/B çıktıları farklı, ilk fark baytı {} (A={:#04x}, B={:#04x})",
                    bd.bits(),
                    name(spec.fam),
                    spec.w,
                    spec.h,
                    e.first_diff,
                    e.padding_a,
                    e.padding_b
                ),
            }
        }
    }
    println!("[sentinel] pencere dışı okuma yok: 14 aile × 3 bit derinliği");
}

/// (b3) — **negatif kontrol**: kasıtlı bir bayt taşan sarmalayıcı yakalanır.
///
/// Bu, ölçüm düzeneğinin duyarlı olduğunu kanıtlar: aynı `OverReadProbe`
/// düzeneği, pencereyi bir bayt aşan bir okumayı **görmelidir**.
#[test]
fn negatif_kontrol_kasitli_tasma_yakalanir() {
    let probe = OverReadProbe::new(64, 64, 4);
    // Kasıtlı hata: veri bölgesinin SON baytından bir sonrasını (padding)
    // okuyan bir "çekirdek".
    let sonuc = probe.run(|src, out| {
        let base = src[64 + 63] as i32; // verinin son baytı (64..127)
        let tasan = src[64 + 64] as i32; // kasıtlı: padding'den bir bayt
        out[0] = (base ^ tasan) as u8;
        out[1] = base as u8;
        out[2] = 0;
        out[3] = 0;
    });
    assert!(
        sonuc.is_err(),
        "negatif kontrol yakalanmadı: kasıtlı padding okuması tespit edilmeliydi"
    );
    let e = sonuc.unwrap_err();
    assert_eq!(e.first_diff, 0, "fark ilk baytta olmalı");
    // Aynı düzenek, doğru bir çekirdekle geçer (kontrol).
    let temiz = probe.run(|src, out| {
        out[0] = src[64 + 63];
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
    });
    assert!(temiz.is_ok(), "doğru çekirdek yanlış pozitif verdi");
    println!("[sentinel] negatif kontrol: kasıtlı taşma yakalandı (ilk fark 0)");
}
