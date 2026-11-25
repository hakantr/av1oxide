//! Blok yeniden oynatma probu — blok düzeyi tanılama yolu.
//!
//! Piksel farkı olan karelerde **ilk farklı bloğun** girdilerini Rust'tan
//! dökmek için kullanılır. Amaç üretim yolunu değiştirmek değil, tek bir
//! bloğun **gerçek** ara değerlerini (hazırlanmış kenar, tahmin, dequant
//! sonrası katsayılar, nihai pikseller) kaydetmektir.
//!
//! Alıcı (`set_sink`) ve hedef (`set_target`) kurulmadıkça **hiçbir kayıt
//! yazılmaz**; `enabled()` yalnız iki `OnceLock` denetimidir. Alıcıyı yalnız
//! `av1oxide-harness replay` alt komutu kurar, yani üretim API'sinin gözlenen
//! davranışı değişmez.
//!
//! ## Kayıt biçimi (küçük sonlu, bayt sırası sabit)
//!
//! ```text
//! "RPL1"
//! u8 kind (0 = ipred, 1 = cfl, 2 = pal)
//! u8 bpc, u8 layout (0 = 4:0:0, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4), u8 pad
//! u32 frame, u8 plane, u8 pad3
//! u32 x, u32 y, u32 w, u32 h          (düzlem pikseli)
//! i32 mode, i32 angle, i32 flags, i32 max_width, i32 max_height
//! i32 alpha, i32 bitdepth_max
//! u32 tx, u32 txtp, i32 eob
//! u32 dst_x, u32 dst_y                (itx'in **gerçekten** yazdığı konum)
//! u32 edge_len + edge baytları        (257*bpp; kind 0/1)
//! u32 ac_len  + i16 LE               (w*h; kind 1)
//! u32 idx_len + bayt                 (w*h/2; kind 2)
//! u32 pal_len + u16 LE               (8; kind 2)
//! u32 pred_len + bayt                (w*h*bpp)
//! u32 before_len + bayt              (tx bloğu, w*h*bpp)
//! u32 coeff_len + i32 LE             (1024)
//! u32 after_len + bayt               (w*h*bpp)
//! ```
//!
//! `frame`, **çözülen** kare sırasıdır (dav1d `dav1d_decode_frame` sırası;
//! `show_existing` çözmez, sayılmaz).

#![forbid(unsafe_code)]

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

/// Hedef blok: bu pikseli içeren blok(lar) dökülür.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// Çözülen kare sırası; `u32::MAX` = **bütün kareler** (kare eşlemesi
    /// çıktı sırasından farklı olabileceği için sürücü içerikten eşler).
    pub frame: u32,
    /// Düzlem (0 = Y, 1 = U, 2 = V).
    pub plane: u8,
    /// Düzlem pikseli.
    pub x: u32,
    /// Düzlem pikseli.
    pub y: u32,
}

/// Blok parametreleri (tahmin kaydı).
#[derive(Clone, Copy, Debug, Default)]
pub struct Pred {
    /// 0 = ipred, 1 = cfl, 2 = pal.
    pub kind: u8,
    /// DSP mod indeksi (ipred) / CfL mod indeksi (0..3).
    pub mode: i32,
    /// Mutlak açı + yumuşak/kenar filtresi bayrakları.
    pub angle: i32,
    /// Yumuşak blok + kenar filtresi bayrakları.
    pub flags: i32,
    /// Z2 `max_width`.
    pub max_width: i32,
    /// Z2 `max_height`.
    pub max_height: i32,
    /// CfL alfa (q3).
    pub alpha: i32,
    /// `bitdepth_max`.
    pub bitdepth_max: i32,
    /// Tx boyutu indeksi (TX kaydı için).
    pub tx: u8,
    /// Tx türü indeksi.
    pub txtp: u8,
    /// Eob (−1 = atlandı).
    pub eob: i32,
    /// Düzlem pikseli konum.
    pub x: u32,
    /// Düzlem pikseli konum.
    pub y: u32,
    /// Blok genişliği (piksel).
    pub w: u32,
    /// Blok yüksekliği (piksel).
    pub h: u32,
}

type Sink = fn(&[u8]);

static SINK: OnceLock<Sink> = OnceLock::new();
static TARGET: OnceLock<Option<Target>> = OnceLock::new();

thread_local! {
    static FRAME_NO: Cell<u32> = const { Cell::new(0) };
    /// Blok kayıtlarının taşıyacağı **geçerli** kare indeksi (RPL0 ile aynı).
    static CURRENT: Cell<u32> = const { Cell::new(0) };
    static BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// Düzlem başına tahmin kaydı; tx kaydı gelene kadar bekletilir.
    /// (Kroma U ve V tahminleri tx döngülerinden **önce** yazılır, bu yüzden
    /// tek bir ortak yuva yanlış eşleşirdi.)
    static PENDING: RefCell<[Option<Pred>; 3]> = const { RefCell::new([None; 3]) };
}

/// Alıcıyı kurar (ikinci çağrı yok sayılır).
pub fn set_sink(sink: Sink) -> bool {
    SINK.set(sink).is_ok()
}

/// Hedefi kurar (ikinci çağrı yok sayılır).
pub fn set_target(target: Target) -> bool {
    TARGET.set(Some(target)).is_ok()
}

/// Hedef kurulu mu? (kayıt yolu açık mı)
#[must_use]
#[inline(always)]
pub fn enabled() -> bool {
    SINK.get().is_some() && TARGET.get().is_some_and(Option::is_some)
}

/// Yeni bir **çözülen** kare başladı; sıra numarasını döndürür.
///
/// `RPL0` kaydı yalnız alıcı kuruluysa yazılır ve kare boyutlarını taşır:
/// sürücü, Rust YUV akışını bu boyutlarla bölerek probun kare sırasını
/// **çıktı** kareleriyle içerikten eşler (çözülemeyen kareler çıktı üretmez,
/// bu yüzden iki sıra numarası aynı olmayabilir).
pub fn frame_start(width: u32, height: u32, bpc: u8, layout: u8) -> u32 {
    FRAME_NO.with(|c| {
        let v = c.get();
        c.set(v.wrapping_add(1));
        CURRENT.with(|cur| cur.set(v));
        if enabled() {
            BUF.with(|buf| {
                let mut b = buf.borrow_mut();
                b.clear();
                b.extend_from_slice(b"RPL0");
                push_u32(&mut b, v);
                push_u32(&mut b, width);
                push_u32(&mut b, height);
                b.push(bpc);
                b.push(layout);
                b.extend_from_slice(&[0, 0]);
                if let Some(sink) = SINK.get() {
                    sink(&b);
                }
            });
        }
        v
    })
}

/// Çözülen karenin **içerik özeti** (FNV-1a 64) — `RPL3`.
///
/// Sürücü, Rust YUV çıktısındaki kareyi bu özetle eşler; `show_existing` ve
/// çözülemeyen kareler yüzünden çözme sırası ile çıktı sırası farklı olabilir.
pub fn frame_end(hash: u64, bytes: u64) {
    if !enabled() {
        return;
    }
    let frame = CURRENT.with(Cell::get);
    BUF.with(|buf| {
        let mut b = buf.borrow_mut();
        b.clear();
        b.extend_from_slice(b"RPL3");
        push_u32(&mut b, frame);
        b.extend_from_slice(&hash.to_le_bytes());
        b.extend_from_slice(&bytes.to_le_bytes());
        if let Some(sink) = SINK.get() {
            sink(&b);
        }
    });
}

/// Kare süzgeci: `u32::MAX` bütün kareleri seçer.
fn frame_matches(target: Target, frame: u32) -> bool {
    target.frame == u32::MAX || target.frame == frame
}

/// Hedefle kesişim var mı?
fn hits(target: Target, plane: u8, x: u32, y: u32, w: u32, h: u32) -> bool {
    target.plane == plane && target.x >= x && target.x < x + w && target.y >= y && target.y < y + h
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_blob(buf: &mut Vec<u8>, data: &[u8]) {
    push_u32(buf, data.len() as u32);
    buf.extend_from_slice(data);
}

fn push_i32_blob(buf: &mut Vec<u8>, data: &[i32]) {
    push_u32(buf, (data.len() * 4) as u32);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_i16_blob(buf: &mut Vec<u8>, data: &[i16]) {
    push_u32(buf, (data.len() * 2) as u32);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Tahmin kaydını **hemen** yazar (kenar/tahmin hazır olduğunda).
///
/// `edge`: hazırlanmış kenar tamponu (`257*bpp`); `ac`: CfL AC (yalnız kind 1);
/// `idx`/`pal`: palet (yalnız kind 2); `pred`: tahmin pikselleri.
#[allow(clippy::too_many_arguments)]
pub fn pred_record(
    p: Pred,
    plane: u8,
    bpc: u8,
    layout: u8,
    edge: &[u8],
    ac: &[i16],
    idx: &[u8],
    pal: &[u16],
    pred: &[u8],
) {
    if !enabled() {
        return;
    }
    let Some(target) = TARGET.get().copied().flatten() else {
        return;
    };
    let frame = CURRENT.with(Cell::get);
    if !frame_matches(target, frame) || !hits(target, plane, p.x, p.y, p.w, p.h) {
        return;
    }
    BUF.with(|buf| {
        let mut b = buf.borrow_mut();
        b.clear();
        b.extend_from_slice(b"RPL1");
        b.push(p.kind);
        b.push(bpc);
        b.push(layout);
        b.push(0);
        push_u32(&mut b, frame);
        b.push(plane);
        b.extend_from_slice(&[0, 0, 0]);
        push_u32(&mut b, p.x);
        push_u32(&mut b, p.y);
        push_u32(&mut b, p.w);
        push_u32(&mut b, p.h);
        push_i32(&mut b, p.mode);
        push_i32(&mut b, p.angle);
        push_i32(&mut b, p.flags);
        push_i32(&mut b, p.max_width);
        push_i32(&mut b, p.max_height);
        push_i32(&mut b, p.alpha);
        push_i32(&mut b, p.bitdepth_max);
        push_u32(&mut b, u32::from(p.tx));
        push_u32(&mut b, u32::from(p.txtp));
        push_i32(&mut b, p.eob);
        push_blob(&mut b, edge);
        push_i16_blob(&mut b, ac);
        push_blob(&mut b, idx);
        push_u32(&mut b, (pal.len() * 2) as u32);
        for v in pal {
            b.extend_from_slice(&v.to_le_bytes());
        }
        push_blob(&mut b, pred);
        if let Some(sink) = SINK.get() {
            sink(&b);
        }
        PENDING.with(|slot| slot.borrow_mut()[usize::from(plane)] = Some(p));
    });
}

/// Tx bloğu kaydı: itx **öncesi** hedef, katsayılar ve itx **sonrası** hedef.
///
/// Yalnız hedef pikseli içeren tx bloğu için yazılır. `PENDING` doluysa tahmin
/// alanları da kayda eklenir (tx alanları güncellenir).
#[allow(clippy::too_many_arguments)]
pub fn tx_record(
    plane: u8,
    bpc: u8,
    layout: u8,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    tx: u8,
    txtp: u8,
    eob: i32,
    dst_x: u32,
    dst_y: u32,
    before: &[u8],
    coeffs: &[i32],
    after: &[u8],
) {
    if !enabled() {
        return;
    }
    let Some(target) = TARGET.get().copied().flatten() else {
        return;
    };
    let frame = CURRENT.with(Cell::get);
    if !frame_matches(target, frame) || !hits(target, plane, x, y, w, h) {
        return;
    }
    let pending = PENDING.with(|slot| slot.borrow()[usize::from(plane)]);
    BUF.with(|buf| {
        let mut b = buf.borrow_mut();
        b.clear();
        b.extend_from_slice(b"RPL2");
        b.push(plane);
        b.push(bpc);
        b.push(layout);
        b.push(tx);
        b.push(txtp);
        b.extend_from_slice(&[0, 0, 0]);
        push_u32(&mut b, frame);
        push_u32(&mut b, x);
        push_u32(&mut b, y);
        push_u32(&mut b, w);
        push_u32(&mut b, h);
        push_i32(&mut b, eob);
        push_u32(&mut b, dst_x);
        push_u32(&mut b, dst_y);
        match pending {
            Some(p) => {
                b.push(1);
                b.push(p.kind);
                b.extend_from_slice(&[0, 0]);
                push_i32(&mut b, p.mode);
                push_i32(&mut b, p.angle);
                push_i32(&mut b, p.flags);
                push_i32(&mut b, p.max_width);
                push_i32(&mut b, p.max_height);
                push_i32(&mut b, p.alpha);
                push_i32(&mut b, p.bitdepth_max);
                push_u32(&mut b, p.x);
                push_u32(&mut b, p.y);
                push_u32(&mut b, p.w);
                push_u32(&mut b, p.h);
            }
            None => {
                // 1 (has_pred) + 1 (kind) + 2 (pad) + 7 i32 + 4 u32 = 48 bayt.
                b.push(0);
                b.extend_from_slice(&[0; 47]);
            }
        }
        push_blob(&mut b, before);
        push_i32_blob(&mut b, coeffs);
        push_blob(&mut b, after);
        if let Some(sink) = SINK.get() {
            sink(&b);
        }
    });
}

/// Bekleyen tahmin kayıtlarını temizler (blok/parça sınırında).
pub fn clear_pending() {
    PENDING.with(|slot| *slot.borrow_mut() = [None; 3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static CAPTURED: Mutex<Vec<u8>> = Mutex::new(Vec::new());

    fn capture(data: &[u8]) {
        CAPTURED.lock().unwrap().extend_from_slice(data);
    }

    /// `RPL2` kaydı itx'in **gerçekten** yazdığı konumu (`dst_x`,`dst_y`)
    /// taşır; yeniden oynatma aracı hedef kaymasını bununla ayırır.
    #[test]
    fn tx_record_hedef_konumunu_yazar() {
        let _ = set_sink(capture);
        let _ = set_target(Target {
            frame: u32::MAX,
            plane: 0,
            x: 2,
            y: 3,
        });
        let before = vec![0u8; 4 * 4 * 2];
        let coeffs = vec![0i32; 1024];
        let after = vec![7u8; 4 * 4 * 2];
        tx_record(
            0, 10, 0, 2, 3, 4, 4, 0, 9, 12, 6, 7, &before, &coeffs, &after,
        );
        let data = CAPTURED.lock().unwrap().clone();
        assert!(data.starts_with(b"RPL2"), "kayıt büyüsü");
        assert_eq!(data[4], 0, "düzlem");
        assert_eq!(data[5], 10, "bpc");
        assert_eq!(&data[36..40], &6u32.to_le_bytes(), "dst_x");
        assert_eq!(&data[40..44], &7u32.to_le_bytes(), "dst_y");
    }
}
