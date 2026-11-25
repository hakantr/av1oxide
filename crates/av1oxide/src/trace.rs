//! MSAC sembol izi — yalnız `msac-trace` özelliği açıkken derlenen tanılama yolu.
//!
//! Üretim API'sinin anlamı değişmeden,
//! bağımlılık eklenmeden, harness'in açabildiği bir yolla her MSAC ilkeli için
//! kare/tile/sıra, işlev türü, dönen değer ve `rng`/`cnt` durumu yazılır.
//!
//! Özellik kapalıyken bu modül **hiç derlenmez**; üretim yolu aynı kalır.
//! Biçim `scripts/oracle/msac_trace.c` ile birebirdir:
//!
//! ```text
//! TILE <tile#> <boyut> <ilk4bayt-hex>
//! S <tile#> <sıra#> <tür> <n> <dönen> <rng> <cnt>
//! M <tür> <a> <b> <c> <d>
//! ```

#![forbid(unsafe_code)]

use std::cell::{Cell, RefCell};
use std::fmt;
use std::sync::OnceLock;

type Sink = fn(&str);

static SINK: OnceLock<Sink> = OnceLock::new();

thread_local! {
    static BUF: RefCell<String> = const { RefCell::new(String::new()) };
    static TILE_NO: Cell<u32> = const { Cell::new(0) };
    static SEQ: Cell<u32> = const { Cell::new(0) };
    static SUPPRESS: Cell<u32> = const { Cell::new(0) };
    static FRAME_NO: Cell<u32> = const { Cell::new(0) };
}

/// İç içe çağrı bastırma koruması (dav1d'de aynı derleme birimindeki iç
/// çağrılar `--wrap` ile görünmez; Rust tarafı da onları yazmamalıdır).
pub struct SuppressGuard(());

impl Drop for SuppressGuard {
    fn drop(&mut self) {
        SUPPRESS.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Bu blok boyunca `sym` olaylarını bastırır (`bool_adapt -> bool`,
/// `hi_tok -> adapt`, `subexp -> bool_equi` iç çağrıları).
pub fn suppress() -> SuppressGuard {
    SUPPRESS.with(|c| c.set(c.get().saturating_add(1)));
    SuppressGuard(())
}

/// İz alıcısını bir kez kurar (ikinci çağrı yok sayılır).
pub fn set_sink(sink: Sink) -> bool {
    SINK.set(sink).is_ok()
}

/// İz alıcısı kurulu mu?
#[inline]
#[must_use]
pub fn enabled() -> bool {
    SINK.get().is_some()
}

/// Biçimlendirilmiş tek iz satırı yazar (satır sonu eklenir).
pub fn line(args: fmt::Arguments<'_>) {
    let Some(sink) = SINK.get() else { return };
    BUF.with(|buf| {
        let mut s = buf.borrow_mut();
        s.clear();
        use std::fmt::Write as _;
        let _ = s.write_fmt(args);
        s.push('\n');
        sink(&s);
    });
}

/// Çözme sırası kare işareti: `FRAME <sıra#> <tür> <gen> <yük>`.
///
/// Sayaç **çözülen** kare başına artar (dav1d `dav1d_decode_frame` ile aynı
/// sıra); `show_existing` çıktıları kare çözmez, bu yüzden sayılmaz.
pub fn frame(kind: u32, width: u32, height: u32, sb128: u32) {
    let no = FRAME_NO.with(|c| {
        let v = c.get();
        c.set(v.wrapping_add(1));
        v
    });
    TILE_NO.with(|c| c.set(0));
    SEQ.with(|c| c.set(0));
    line(format_args!("FRAME {no} {kind} {width} {height} {sb128}"));
}

/// `MsacContext::init` başına tile işareti: `TILE <tile#> <boyut> <ilk4>`.
pub fn tile(data: &[u8]) {
    let no = TILE_NO.with(|c| {
        let v = c.get();
        c.set(v.wrapping_add(1));
        v
    });
    SEQ.with(|c| c.set(0));
    let b0 = data.first().copied().unwrap_or(0);
    let b1 = data.get(1).copied().unwrap_or(0);
    let b2 = data.get(2).copied().unwrap_or(0);
    let b3 = data.get(3).copied().unwrap_or(0);
    line(format_args!(
        "TILE {no} {} {b0:02x}{b1:02x}{b2:02x}{b3:02x}",
        data.len()
    ));
}

/// Tek MSAC ilkeli: `S <tile#> <sıra#> <tür> <n> <dönen> <rng> <cnt>
/// <cdf0> <cdfn> <cdf_off>`.
#[allow(clippy::too_many_arguments)]
pub fn sym(kind: &str, n: usize, ret: i64, rng: u32, cnt: i32, cdf0: u32, cdfn: u32, cdf_off: i64) {
    if SUPPRESS.with(|c| c.get() > 0) {
        return;
    }
    let seq = SEQ.with(|c| {
        let v = c.get();
        c.set(v.wrapping_add(1));
        v
    });
    let tile = TILE_NO.with(|c| c.get().saturating_sub(1));
    line(format_args!(
        "S {tile} {seq} {kind} {n} {ret} {rng} {cnt} {cdf0} {cdfn} {cdf_off}"
    ));
}

/// Sözdizimi işareti: `M <tür> <a> <b> <c> <d>` (blok konumu / öğe adı).
pub fn mark(kind: &str, a: i32, b: i32, c: i32, d: i32) {
    line(format_args!("M {kind} {a} {b} {c} {d}"));
}
