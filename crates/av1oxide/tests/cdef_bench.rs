//! S15.4 — CDEF pilot benchmark'ı (Rust skaler + Rust AVX2).
//!
//! ## Yöntem
//!
//! - Saat: `std::time::Instant` (Linux'ta `CLOCK_MONOTONIC`) — dav1d tarafındaki
//!   C ölçümüyle **aynı** kaynak (`scripts/oracle/cdef_bench.c`).
//! - Girdi: 1024 blok, sabit tohumlu (xorshift64*) rastgele piksel + değişen
//!   `dir`/`pri`/`sec`/`damping`/`edges` (blokların %87,5'inde dört kenar da var).
//! - Ölçüm: aynı 1024 blok `tur` kez koşulur; sonuç **ns/blok** = geçen süre /
//!   (tur × 1024). Isınma turu ölçüme katılmaz.
//! - `tur` varsayılanı 200 → blok başına 200 örnek. Değer `CDEF_BENCH_TUR`
//!   ortam değişkeniyle değiştirilir.
//!
//! ## Neden `#[ignore]`?
//!
//! Benchmark **kabul ölçütü değildir** (S15.4/9) ve `cargo test --workspace`
//! içinde koşmamalıdır. Elle:
//!
//! ```text
//! cargo test --release -p av1oxide --test cdef_bench -- --ignored --nocapture
//! ```
//!
//! Dav1d tarafı ayrı süreçtir: `bash scripts/oracle/run_cdef_bench.sh`.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::cdef::{CdefDirFn, CdefFilterFn};
use av1oxide::dsp::scalar::cdef as scalar;
use common::cdef_fixture::{Case, Rng, random_case, run_dir, run_filter};
use common::on_kosul::{self, Neden};
use std::time::Instant;

/// Ölçümde kullanılan blok sayısı (dav1d tarafındaki `NBLOK` ile aynı).
const NBLOK: usize = 1024;

fn tur_sayisi() -> usize {
    std::env::var("CDEF_BENCH_TUR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200)
}

/// 1024 blokluk sabit küme üretir (blok boyutu verilir).
fn bloklar(size: u8) -> Vec<Case> {
    let mut rng = Rng::new(0x5A15_04C0_BEEF_0001);
    let mut out = Vec::with_capacity(NBLOK);
    for i in 0..NBLOK {
        let mut case = random_case(&mut rng, 1, 8, size, (i % 4) as u8);
        // dav1d tarafıyla aynı dağılım: blokların çoğunda dört kenar da var.
        if i % 8 != 0 {
            case.edges = 0b1111;
        }
        out.push(case);
    }
    out
}

/// 1024 blokluk `dir` kümesi.
fn dir_bloklari() -> Vec<Case> {
    let mut rng = Rng::new(0x5A15_04C0_BEEF_0002);
    (0..NBLOK)
        .map(|i| random_case(&mut rng, 0, 8, 0, (i % 4) as u8))
        .collect()
}

/// Filtre ölçümü: `tur` × 1024 blok, ns/blok.
fn olc_filter(kernel: CdefFilterFn, size: u8, tur: usize) -> f64 {
    let cases = bloklar(size);
    // Isınma.
    for case in cases.iter().take(NBLOK / 8) {
        std::hint::black_box(run_filter(case, kernel));
    }
    let t0 = Instant::now();
    for _ in 0..tur {
        for case in &cases {
            std::hint::black_box(run_filter(case, kernel));
        }
    }
    t0.elapsed().as_nanos() as f64 / (tur as f64 * NBLOK as f64)
}

/// `dir` ölçümü: `tur` × 1024 blok, ns/blok.
fn olc_dir(kernel: CdefDirFn, tur: usize) -> f64 {
    let cases = dir_bloklari();
    for case in cases.iter().take(NBLOK / 8) {
        std::hint::black_box(run_dir(case, kernel));
    }
    let t0 = Instant::now();
    for _ in 0..tur {
        for case in &cases {
            std::hint::black_box(run_dir(case, kernel));
        }
    }
    t0.elapsed().as_nanos() as f64 / (tur as f64 * NBLOK as f64)
}

fn avx2_var() -> bool {
    // F3b: AVX2 yolu yalnız kurulumda üretilen belirteçle açılır.
    av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect())
        .avx2_token()
        .is_some()
}

/// Belirteçli kurulumun `dir` yuvası (AVX2 yoksa skaler — ölçülmez).
fn avx2_dir() -> CdefDirFn {
    av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect())
        .for_bit_depth(av1oxide::frame::BitDepth::Eight)
        .cdef()
        .dir
}

/// Belirteçli kurulumun `filter` yuvası (AVX2 yoksa skaler — ölçülmez).
fn avx2_filter() -> CdefFilterFn {
    av1oxide::dsp::DspDispatch::select(av1oxide::cpu::CpuFlags::detect())
        .for_bit_depth(av1oxide::frame::BitDepth::Eight)
        .cdef()
        .fb[0]
}

/// Rust skaler + Rust AVX2 süreleri (ns/blok). Kabul ölçütü **değildir**.
#[test]
#[ignore = "benchmark: kabul ölçütü değil; elle koşulur (--ignored)"]
fn cdef_bench_rust() {
    let tur = tur_sayisi();
    println!("yol=rust tur={tur} blok/tur={NBLOK}");
    for (ad, kernel) in [("rust-scalar", scalar::filter::<8> as CdefFilterFn)] {
        for (size, etiket) in [(0u8, "8x8"), (1, "4x8"), (2, "4x4")] {
            let ns = olc_filter(kernel, size, tur);
            println!("{ad} filter {etiket} : {ns:.2} ns/blok");
        }
        let ns = olc_dir(scalar::find_dir::<8>, tur);
        println!("{ad} dir    8x8  : {ns:.2} ns/blok");
    }
    if avx2_var() {
        let kernel = avx2_filter();
        for (size, etiket) in [(0u8, "8x8"), (1, "4x8"), (2, "4x4")] {
            let ns = olc_filter(kernel, size, tur);
            println!("rust-avx2 filter {etiket} : {ns:.2} ns/blok");
        }
        let ns = olc_dir(avx2_dir(), tur);
        println!("rust-avx2 dir    8x8  : {ns:.2} ns/blok");
    } else {
        on_kosul::eksik(
            "cdef_bench::cdef_bench_rust[avx2]",
            Neden::avx2(),
            "Rust AVX2 ölçümü",
        );
    }
}
