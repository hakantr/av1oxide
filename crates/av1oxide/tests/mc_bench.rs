//! S15.5 — MC pilot benchmark'ı (Rust skaler + Rust AVX2 + sarmalayıcı maliyeti).
//!
//! ## Yöntem
//!
//! - Saat: `std::time::Instant` (Linux'ta `CLOCK_MONOTONIC`) — dav1d tarafındaki
//!   C ölçümüyle **aynı** kaynak (`scripts/oracle/mc_bench.c`).
//! - Girdi: genişlik başına 1024 blok, sabit tohumlu (xorshift64*) ve
//!   **dav1d tarafıyla bayt-aynı** (aynı PRNG, aynı sıra: `mx`, `my`, sonra
//!   `(h+7)×(w+7)` kaynak baytı). Kanıt: iki taraf da aynı `girdi-fnv1a64`
//!   değerini basar.
//! - Ölçüm: aynı 1024 blok `tur` kez koşulur; sonuç **ns/blok** = geçen süre /
//!   (tur × 1024). Isınma turu ölçüme katılmaz.
//! - `tur` varsayılanı 200; `MC_BENCH_TUR` ortam değişkeniyle değiştirilir.
//!
//! ## Üç Rust ölçümü (S15.6 için)
//!
//! | Ölçüm | Neyi içerir | Neyi içermez |
//! |---|---|---|
//! | `rust-scalar` | skaler referans gövde | sarmalayıcı yok |
//! | `rust-avx2-sarmalayici` | ISA tespiti + tam sözleşme denetimi + yerel dolgu **kopyası** + sıcak döngü | — |
//! | `rust-avx2-hazir` | kısa denetim (assert) + sıcak döngü | **kopya yok** |
//!
//! `sarmalayici - hazir` farkı, güvenli sarmalayıcının (uzunluk denetimi +
//! ISA tespiti + `w+7` baytlık satır kopyası) maliyetidir; sıcak döngünün
//! kendisi `hazir` sütunudur. Performans **kabul ölçütü değildir**.
//!
//! ## Neden `#[ignore]`?
//!
//! Benchmark kabul ölçütü değildir (S15.5/10) ve `cargo test --workspace`
//! içinde koşmamalıdır. Elle:
//!
//! ```text
//! MC_BENCH_TUR=200 cargo test --release -p av1oxide --test mc_bench -- --ignored --nocapture
//! ```
//!
//! Dav1d tarafı ayrı süreçtir: `bash scripts/oracle/run_mc_bench.sh`.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::mc::{McBitDepth, McFn};
use av1oxide::dsp::scalar::mc as scalar;
use common::mc_fixture::Rng;
use common::on_kosul::{self, Neden};
use std::time::Instant;

/// Ölçümde kullanılan blok sayısı (dav1d tarafındaki `NBLOK` ile aynı).
const NBLOK: usize = 1024;
/// Genişlik listesi (dav1d blok boyutu tablosu).
const W_LIST: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];
/// Her genişlik için dav1d'nin desteklediği en küçük yükseklik.
const H_LIST: [usize; 7] = [2, 4, 4, 4, 8, 16, 64];

/// AVX2 yolu bu platformda kullanılabilir mi?
fn avx2_var() -> bool {
    // F3b: AVX2 yolu yalnız kurulumda üretilen belirteçle açılır.
    av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect())
        .avx2_token()
        .is_some()
}

fn tur_sayisi() -> usize {
    std::env::var("MC_BENCH_TUR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200)
}

/// Bir genişliğin 1024 bloğu (dav1d `bloklari_kur` ile bayt-aynı).
struct Bloklar {
    /// Blok başına dolgu tamponu: `(h+7)×(w+7)` bayt.
    src: Vec<u8>,
    /// Blok başına hedef: `w×h`.
    dst: Vec<u8>,
    mx: Vec<u8>,
    my: Vec<u8>,
    /// Blok başına kaynak uzunluğu (bayt).
    src_len: usize,
}

impl Bloklar {
    fn uret(rng: &mut Rng, w: usize, h: usize, fnv: &mut u64) -> Self {
        let src_len = (h + 7) * (w + 7);
        let mut src = vec![0u8; NBLOK * src_len];
        let dst = vec![0u8; NBLOK * w * h];
        let mut mx = vec![0u8; NBLOK];
        let mut my = vec![0u8; NBLOK];
        for i in 0..NBLOK {
            mx[i] = rng.below(16) as u8;
            my[i] = rng.below(16) as u8;
            fnv_ekle(fnv, mx[i]);
            fnv_ekle(fnv, my[i]);
            let row = &mut src[i * src_len..(i + 1) * src_len];
            for byte in row.iter_mut() {
                *byte = rng.below(256) as u8;
                fnv_ekle(fnv, *byte);
            }
        }
        Self {
            src,
            dst,
            mx,
            my,
            src_len,
        }
    }
}

fn fnv_ekle(fnv: &mut u64, byte: u8) {
    *fnv ^= u64::from(byte);
    *fnv = fnv.wrapping_mul(0x0000_0100_0000_01b3);
}

/// Tam güvenli sarmalayıcı (veya skaler) ile `tur` × 1024 blok, ns/blok.
fn olc(kernel: McFn, blocks: &mut Bloklar, w: usize, h: usize, tur: usize) -> f64 {
    // Isınma.
    for i in 0..NBLOK {
        let mut dst = vec![0u8; w * h];
        let s0 = i * blocks.src_len;
        kernel(
            &mut dst,
            w,
            &blocks.src[s0..s0 + blocks.src_len],
            w + 7,
            w,
            h,
            blocks.mx[i],
            blocks.my[i],
            McBitDepth::B8,
        );
        std::hint::black_box(dst[0]);
    }
    let t0 = Instant::now();
    for _ in 0..tur {
        for i in 0..NBLOK {
            let start = i * w * h;
            let s0 = i * blocks.src_len;
            kernel(
                &mut blocks.dst[start..start + w * h],
                w,
                &blocks.src[s0..s0 + blocks.src_len],
                w + 7,
                w,
                h,
                blocks.mx[i],
                blocks.my[i],
                McBitDepth::B8,
            );
        }
    }
    t0.elapsed().as_nanos() as f64 / (tur as f64 * NBLOK as f64)
}

/// Sıcak döngü (kopyasız, kısa denetimli) ile `tur` × 1024 blok, ns/blok.
/// Yalnız x86_64 (`dsp::x86` girişi); başka hedefte derlenmez (P29).
#[cfg(target_arch = "x86_64")]
fn olc_hazir(blocks: &mut Bloklar, w: usize, h: usize, tur: usize) -> f64 {
    let token = av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect())
        .avx2_token()
        .expect("AVX2 belirteci (çağıran avx2_var ile denetler)");
    let kernel = |dst: &mut [u8],
                  dst_stride: usize,
                  padded: &[u8],
                  padded_stride: usize,
                  w: usize,
                  h: usize,
                  mx: u8,
                  my: u8| {
        av1oxide::dsp::x86::mc::put_8tap_regular_prepared(
            token,
            dst,
            dst_stride,
            padded,
            padded_stride,
            w,
            h,
            mx,
            my,
        );
    };

    // Sıcak döngü, sarmalayıcının **kendi dolgu düzenini** görür: satır adımı
    // `align32(w+39)`, veri `w+7` bayt, kalan sıfır. Bu hazırlık ölçüme
    // katılmaz; girdi piksel baytları `blocks.src` ile aynıdır.
    let lstride = (w + 39).next_multiple_of(32);
    let rows = h + 7;
    let mut padded = vec![0u8; NBLOK * rows * lstride];
    for i in 0..NBLOK {
        let s0 = i * blocks.src_len;
        for r in 0..rows {
            let d = i * rows * lstride + r * lstride;
            padded[d..d + w + 7].copy_from_slice(&blocks.src[s0 + r * (w + 7)..][..w + 7]);
        }
    }
    for i in 0..NBLOK {
        let mut dst = vec![0u8; w * h];
        let p0 = i * rows * lstride;
        kernel(
            &mut dst,
            w,
            &padded[p0..p0 + rows * lstride],
            lstride,
            w,
            h,
            blocks.mx[i],
            blocks.my[i],
        );
        std::hint::black_box(dst[0]);
    }
    let t0 = Instant::now();
    for _ in 0..tur {
        for i in 0..NBLOK {
            let start = i * w * h;
            let p0 = i * rows * lstride;
            kernel(
                &mut blocks.dst[start..start + w * h],
                w,
                &padded[p0..p0 + rows * lstride],
                lstride,
                w,
                h,
                blocks.mx[i],
                blocks.my[i],
            );
        }
    }
    t0.elapsed().as_nanos() as f64 / (tur as f64 * NBLOK as f64)
}

/// Rust skaler + Rust AVX2 + sarmalayıcı/sıcak döngü ayrımı (ns/blok).
#[test]
#[ignore = "benchmark: kabul ölçütü değil; elle koşulur (--ignored)"]
fn mc_bench_rust() {
    if !avx2_var() {
        on_kosul::eksik("mc_bench::mc_bench_rust", Neden::avx2(), "Rust AVX2 ölçümü");
        return;
    }
    let tur = tur_sayisi();
    let mut rng = Rng::new(0x5A15_05C0_BEEF_0001);
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    println!("yol=rust tur={tur} blok/tur={NBLOK}");
    for wi in 0..7 {
        let w = W_LIST[wi];
        let h = H_LIST[wi];
        let mut blocks = Bloklar::uret(&mut rng, w, h, &mut fnv);
        let skaler = olc(scalar::put_8tap_regular, &mut blocks, w, h, tur);
        #[cfg(target_arch = "x86_64")]
        let sarmalayici = olc(
            av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect())
                .avx2_token()
                .map_or(
                    scalar::put_8tap_regular as McFn,
                    av1oxide::dsp::x86::mc::put_8tap_regular_fn,
                ),
            &mut blocks,
            w,
            h,
            tur,
        );
        #[cfg(not(target_arch = "x86_64"))]
        let sarmalayici = skaler;
        #[cfg(target_arch = "x86_64")]
        let hazir = olc_hazir(&mut blocks, w, h, tur);
        #[cfg(not(target_arch = "x86_64"))]
        let hazir = skaler;
        println!(
            "rust w={w:3} h={h:3} : skaler={skaler:.2} avx2-sarmalayici={sarmalayici:.2} \
             avx2-hazir={hazir:.2} sarmalayici-farki={:.2} ns/blok",
            sarmalayici - hazir
        );
    }
    println!("girdi-fnv1a64={fnv:016x}");
}
