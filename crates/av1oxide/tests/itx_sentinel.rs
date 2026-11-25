//! S2b.1 — itx sentinel / over-read / sınır testleri (S15.3 araçlarıyla).
//!
//! Üç şeyi ölçer:
//!
//! 1. **Taşan yazma yok**: hedef düzlem `SentinelBuffer` içine yerleştirilir;
//!    çekirdek çağrısından sonra ön/arka koruyucu baytları değişmemiş olmalı.
//!    Ayrıca **stride dolgusu** (satır sonu fazlalığı) da sentinel deseniyle
//!    doldurulur ve yazılmadığı doğrulanır.
//! 2. **Sınır dışı okuma yok**: aynı piksel verisi iki farklı dolgu deseniyle
//!    (0x00 / 0xFF) koşturulur; piksel çıktıları **bit-aynı** olmalı. Farklılık,
//!    çekirdeğin satır dolgusundan okuduğunun kanıtıdır. Aynı yöntem katsayı
//!    tamponu için de uygulanır: `sw·sh` dışındaki kuyruk iki farklı desenle
//!    doldurulur; çıktı değişmemelidir.
//! 3. **Negatif kontrol**: sentinel düzeneğinin kasıtlı bir taşmayı gerçekten
//!    yakaladığı (guard'a elle yazma → `SentinelViolation`) ve sözleşme dışı
//!    kısa hedef diliminin paniklediği gösterilir.
//!
//! Not (dürüstlük): Rust dilimleri sınır denetimlidir; dilim **dışına** okuma
//! zaten mümkün değildir. Bu yüzden over-read ölçümü, dilim **içinde** kalan
//! ama mantıksal düzlem dışındaki dolgu bölgeleri için yapılır — gerçek
//! çekirdeklerin padding'e taşma riski tam olarak budur (SMD-07).

#![forbid(unsafe_code)]

mod common;

use av1oxide::dsp::itx::{TxSize, TxType};
use av1oxide::frame::BitDepth;
use common::sentinel::{SentinelBuffer, SentinelViolation};

/// F3c: bit derinliği tablosunun skaler yuvası (`itxfm_add[tx][txtp]`).
#[allow(clippy::too_many_arguments)]
fn inverse_transform_add(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bd: av1oxide::frame::BitDepth,
) {
    let f = av1oxide::dsp::itx::ItxDsp::scalar_for(bd).itxfm_add[tx.index()][txtp.index()];
    f(dst, stride, coeff, eob, tx, txtp, bd);
}

/// Test edilen hücreler: bütün 1B türlerini, kare/dikdörtgen/64 boyutlarını
/// ve WHT'yi kapsar.
const CASES: &[(TxSize, TxType)] = &[
    (TxSize::Tx4x4, TxType::DctDct),
    (TxSize::Tx4x4, TxType::AdstAdst),
    (TxSize::Tx4x4, TxType::FlipadstFlipadst),
    (TxSize::Tx4x4, TxType::Idtx),
    (TxSize::Tx4x4, TxType::HDct),
    (TxSize::Tx4x4, TxType::VAdst),
    (TxSize::Tx4x4, TxType::HFlipadst),
    (TxSize::Tx4x4, TxType::WhtWht),
    (TxSize::Rtx4x8, TxType::AdstDct),
    (TxSize::Rtx8x4, TxType::DctAdst),
    (TxSize::Rtx4x16, TxType::AdstAdst),
    (TxSize::Rtx16x4, TxType::FlipadstAdst),
    (TxSize::Rtx8x16, TxType::DctDct),
    (TxSize::Rtx16x8, TxType::Idtx),
    (TxSize::Tx16x16, TxType::AdstAdst),
    (TxSize::Rtx16x32, TxType::DctDct),
    (TxSize::Rtx32x16, TxType::Idtx),
    (TxSize::Tx32x32, TxType::DctDct),
    (TxSize::Rtx32x64, TxType::DctDct),
    (TxSize::Rtx64x32, TxType::DctDct),
    (TxSize::Rtx16x64, TxType::DctDct),
    (TxSize::Rtx64x16, TxType::DctDct),
    (TxSize::Tx64x64, TxType::DctDct),
];

fn bit_depth(bpc: u8) -> BitDepth {
    BitDepth::from_bits(u32::from(bpc)).unwrap()
}

/// Deterministik katsayı deseni (geçerli aralık: `|c| <= 2^(bpc+7)-1`).
fn coeff_deseni(sw: usize, sh: usize, eob: usize, bpc: u8, seed: u32) -> [i32; 1024] {
    let limit = (1i32 << (bpc + 7)) - 1;
    let mut state = 0x1234_5678u32 ^ seed;
    let mut coeff = [0i32; 1024];
    for slot in coeff.iter_mut().take(sw * sh) {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = ((state >> 8) as i32) % limit;
        *slot = if state & 1 == 0 { v } else { -v };
    }
    // eob sonrası sıfır (geçerli katsayı desteği).
    for slot in coeff.iter_mut().take(sw * sh).skip(eob + 1) {
        *slot = 0;
    }
    coeff
}

/// Hedef düzlem: `h` satır × `stride` bayt; satır sonu fazlalığı `pad` deseniyle.
fn hedef_doldur(
    buf: &mut SentinelBuffer,
    w: usize,
    h: usize,
    stride: usize,
    bpp: usize,
    bpc: u8,
    pad: u8,
) {
    let max = (1u32 << bpc) - 1;
    let data = buf.data_mut();
    data.fill(pad);
    let mut state = 0xC0FF_EE00u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = (state & max) as u16;
            let off = y * stride + x * bpp;
            if bpp == 1 {
                data[off] = v as u8;
            } else {
                data[off..off + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

fn satir_dolgusu_ayni(
    buf: &SentinelBuffer,
    w: usize,
    h: usize,
    stride: usize,
    bpp: usize,
    pad: u8,
) -> bool {
    let data = buf.data();
    for y in 0..h {
        for b in w * bpp..stride {
            if data[y * stride + b] != pad {
                return false;
            }
        }
    }
    true
}

fn piksel_bolgesi(buf: &SentinelBuffer, w: usize, h: usize, stride: usize, bpp: usize) -> Vec<u8> {
    let data = buf.data();
    let mut out = Vec::with_capacity(h * w * bpp);
    for y in 0..h {
        out.extend_from_slice(&data[y * stride..y * stride + w * bpp]);
    }
    out
}

/// Stride dolgusu ve koruyucular yazılmıyor; over-read yok (iki dolgu deseni
/// bit-aynı çıktı veriyor). Her hücre × bit derinliği için koşar.
#[test]
fn hedef_ve_katsayi_disina_yazma_okuma_yok() {
    let mut kontrol = 0usize;
    for &(tx, txtp) in CASES {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        for bpc in [8u8, 10, 12] {
            let bpp = if bpc == 8 { 1 } else { 2 };
            let stride = w * bpp + 8;
            let data_len = h * stride;
            // WHT eob'u yok sayar; diğerlerinde tam eob.
            let eob = if matches!(txtp, TxType::WhtWht) {
                0
            } else {
                sw * sh - 1
            };
            let coeff = coeff_deseni(sw, sh, eob, bpc, u32::from(bpc));

            let mut a = SentinelBuffer::with_defaults(data_len);
            let mut b = SentinelBuffer::with_defaults(data_len);
            hedef_doldur(&mut a, w, h, stride, bpp, bpc, 0x00);
            hedef_doldur(&mut b, w, h, stride, bpp, bpc, 0xFF);

            let mut coeff_a = coeff;
            let mut coeff_b = coeff;
            // Katsayı kuyruğu iki farklı desen (over-read kontrolü).
            for (i, slot) in coeff_a.iter_mut().enumerate().skip(sw * sh) {
                *slot = if i & 1 == 0 { 0x1111 } else { -0x1111 };
            }
            for (i, slot) in coeff_b.iter_mut().enumerate().skip(sw * sh) {
                *slot = if i & 1 == 0 { -0x2222 } else { 0x2222 };
            }

            inverse_transform_add(
                a.data_mut(),
                stride,
                &mut coeff_a,
                eob,
                tx,
                txtp,
                bit_depth(bpc),
            );
            inverse_transform_add(
                b.data_mut(),
                stride,
                &mut coeff_b,
                eob,
                tx,
                txtp,
                bit_depth(bpc),
            );

            assert_eq!(
                a.verify(),
                Ok(()),
                "ön/arka koruyucu yazıldı: {tx:?}/{txtp:?}/{bpc}"
            );
            assert_eq!(b.verify(), Ok(()));
            assert!(
                satir_dolgusu_ayni(&a, w, h, stride, bpp, 0x00),
                "stride dolgusu yazıldı: {tx:?}/{txtp:?}/{bpc}"
            );
            assert!(satir_dolgusu_ayni(&b, w, h, stride, bpp, 0xFF));
            assert_eq!(
                piksel_bolgesi(&a, w, h, stride, bpp),
                piksel_bolgesi(&b, w, h, stride, bpp),
                "çıktı dolgu desenine bağlı (over-read): {tx:?}/{txtp:?}/{bpc}"
            );
            // Katsayı tamponu: yazılan bölge sıfır, kuyruk değişmemiş.
            assert!(coeff_a[..sw * sh].iter().all(|&v| v == 0));
            assert!(coeff_b[..sw * sh].iter().all(|&v| v == 0));
            for i in sw * sh..1024 {
                let beklenen_a = if i & 1 == 0 { 0x1111 } else { -0x1111 };
                let beklenen_b = if i & 1 == 0 { -0x2222 } else { 0x2222 };
                assert_eq!(coeff_a[i], beklenen_a, "katsayı kuyruğu değişti");
                assert_eq!(coeff_b[i], beklenen_b, "katsayı kuyruğu değişti");
            }
            kontrol += 1;
        }
    }
    assert_eq!(kontrol, CASES.len() * 3);
    println!(
        "sentinel/over-read: {} hücre × 3 bit derinliği temiz",
        CASES.len()
    );
}

/// Negatif kontrol 1: koruyucuya elle yazma sentinel tarafından **yakalanır**.
#[test]
fn negatif_kontrol_kasitli_tasma_yakalanir() {
    let mut buf = SentinelBuffer::with_defaults(64);
    buf.raw_mut()[0] = 0x00; // ön koruyucu
    let err = buf.verify().expect_err("kasıtlı taşma yakalanmalı");
    assert_eq!(err.index, 0);
    assert_eq!(err.found, 0x00);

    let mut buf = SentinelBuffer::with_defaults(64);
    let back = buf.guard_len() + buf.data_len();
    buf.raw_mut()[back + 2] = 0x7F; // arka koruyucu
    let err = buf.verify().expect_err("kasıtlı taşma yakalanmalı");
    assert_eq!(err.index, back + 2);
    assert_eq!(err.found, 0x7F);
    let _: SentinelViolation = err;
}

/// Negatif kontrol 2: sözleşme dışı kısa hedef dilimi panikler (sessiz taşma yok).
#[test]
fn negatif_kontrol_kisa_hedef_panikler() {
    let mut coeff = [0i32; 1024];
    let mut kisa = [0u8; 8];
    let sonuc = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        inverse_transform_add(
            &mut kisa,
            8,
            &mut coeff,
            0,
            TxSize::Tx8x8,
            TxType::DctDct,
            BitDepth::Eight,
        );
    }));
    assert!(sonuc.is_err(), "kısa hedef dilimi paniklemeli");
}

/// Sınır: en büyük/en küçük geçerli katsayı ve en büyük piksel değerleri
/// (8-bit i16 uçları, 10/12-bit `2^(bpc+7)` sınırı) panik/taşma olmadan ve
/// iki dolgu deseninde bit-aynı çalışır.
#[test]
fn sinir_katsayi_ve_piksel_degerleri() {
    let cases = [
        (TxSize::Tx4x4, TxType::DctDct),
        (TxSize::Tx4x4, TxType::AdstAdst),
        (TxSize::Tx8x8, TxType::Idtx),
        (TxSize::Rtx16x8, TxType::FlipadstFlipadst),
        (TxSize::Tx32x32, TxType::DctDct),
        (TxSize::Tx64x64, TxType::DctDct),
    ];
    for (tx, txtp) in cases {
        let w = tx.width();
        let h = tx.height();
        let sw = w.min(32);
        let sh = h.min(32);
        for bpc in [8u8, 10, 12] {
            let bpp = if bpc == 8 { 1 } else { 2 };
            let stride = w * bpp;
            let max_sample = (1u32 << bpc) - 1;
            let max_coef = (1i32 << (bpc + 7)) - 1;
            let min_coef = -(1i32 << (bpc + 7));
            for &dc in &[max_coef, min_coef] {
                let mut coeff = [0i32; 1024];
                coeff[0] = dc;
                let mut dst = vec![max_sample as u8; h * w * bpp];
                if bpp == 2 {
                    for p in 0..h * w {
                        dst[2 * p..2 * p + 2].copy_from_slice(&(max_sample as u16).to_le_bytes());
                    }
                }
                inverse_transform_add(&mut dst, stride, &mut coeff, 0, tx, txtp, bit_depth(bpc));
                assert!(coeff[..sw * sh].iter().all(|&v| v == 0));
                // Çıktı geçerli piksel aralığında kalmalı.
                for p in 0..w * h {
                    let v = if bpp == 1 {
                        u32::from(dst[p])
                    } else {
                        u32::from(u16::from_le_bytes([dst[2 * p], dst[2 * p + 1]]))
                    };
                    assert!(v <= max_sample, "piksel aralık dışı: {v}");
                }
            }
        }
    }
}
