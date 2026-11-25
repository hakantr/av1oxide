# av1oxide (`v0.4.0`)

**English** | [Türkçe](README.tr.md)

`av1oxide` is a **100% stable-Rust** (`rustc 1.100.0`, Edition 2024) reimplementation of the VideoLAN [`dav1d`](https://code.videolan.org/videolan/dav1d) (`1.5.4`) AV1 video decoder with **no external codec or media dependencies**, a **`libdav1d` (`7.0.0`) binary-compatible C-ABI layer**, and a **pure-Rust `.avif` image decoder layer**.

- **Profiles & Bit Depths:** Main (`0`), High (`1`), Professional (`2`); `8-bit`, `10-bit`, `12-bit`; `4:2:0`, `4:2:2`, `4:4:4`, and `Monochrome` (`4:0:0`).
- **AV1 Pipeline:** OBU parsing, Sequence/Frame Headers, MSAC arithmetic entropy decoder, inverse transforms (`ITX`), intra prediction (`IPRED`, CfL, Palette), motion compensation (`MC`, Warp, Superres/Resize, IntraBC, `REFMVS`), in-loop filters (`LF`, `CDEF`, `LR` Wiener/SGR), and Film Grain (`FG`) synthesis.
- **Bit-Exact Conformance:** Verified 100% MD5 bit-exact parity against `dav1d 1.5.4` across **`2,781 / 2,781` streams (`37,664 / 37,664` frames)** — covering `dav1d-test-data` (`19/19` streams, `964` frames) and Argon Streams `v2.1.1` (`2,762/2,762` streams, `36,700` frames) across all CPU capability masks and multithreaded configurations.
- **Capability-Token SIMD (`767` DSP Slots):**
  - `x86_64` (`SSE2`, `SSSE3`, `SSE4.1`, `AVX2`, `AVX-512ICL`): `756 / 767` SIMD slots
  - `aarch64` (`NEON`, `DOTPROD`): `701 / 767` SIMD slots
  - `wasm32` (`+simd128`): `31 / 767` SIMD slots under `100% #![forbid(unsafe_code)]`
  - Secondary architectures (`arm32`, `riscv64`, `loongarch64`, `powerpc64le`): `100%` safe bit-exact scalar fallback
- **Multithreading & Memory Budgets:** Intra-frame (`Tile`, superblock row, in-loop filter) and inter-frame (`FrameSlot`) parallelism (`Config::n_threads`, `Config::max_frame_delay`); pre-allocation pixel and memory limits (`Config::frame_size_limit`, `Config::allocation_limit`).
- **`libdav1d` (`7.0.0`) Compatible C-ABI (`capi`):** Implements all 19 official `DAV1D_API` functions (`dav1d_open`, `dav1d_send_data`, `dav1d_get_picture`, `dav1d_apply_grain`, `dav1d_parse_sequence_header`, etc.) and ships the official `include/dav1d/*.h` C headers.
- **Pure-Rust `.avif` Image Reader (`avif`):** `100% #![forbid(unsafe_code)]` ISOBMFF/HEIF/MIAF/AVIF container parser supporting single `av01` items, multithreaded `grid` tile stitching, `iden` derived chains, auxiliary alpha (`auxl`/`auxC`), `clap`/`irot`/`imir` transforms, `ICC`/`Exif`/`XMP` metadata (including `defl`/`zlib`/`gzip` compressed HEIF metadata items and `cmin` boxes with bounded decompression guards), and ITU-T H.273 conversions (`RGB8`/`RGBA8`/`RGB16`/`RGBA16`) — verified `100%` MD5 bit-exact parity (`156 / 156` valid `.avif` images across planar `Y`, `U`, `V`, `Alpha`, `RGBA8`, and `RGBA16`) against reference C `libavif` on the license-audited `AOMediaCodec/av1-avif` and `AOMediaCodec/libavif` corpora.

---

## Dependency Policy

`av1oxide` does not depend on any third-party codec, image, multimedia, or C media library crates; all AV1 bitstream/entropy/DSP decoding, ISOBMFF/AVIF container parsing, ITU-T H.273 color conversion, and `libdav1d` C-ABI layers are implemented in-tree. All external dependencies are strictly isolated as internal implementation details (`pub(crate)` / private); no types from any dependency crate are re-exported or exposed in `av1oxide`'s public (`pub`) API.

For low-level concurrency, memory layout, and decompression primitives, `av1oxide` uses only foundational crates maintained by the `rust-lang` organization and Rust Core Team members with `default-features = false`:

| Crate | Version | License | Source | Features (`default-features = false`) | Purpose in `av1oxide` | Unsafe / Native Code |
|---|---|---|---|---|---|---|
| `crossbeam-utils` | `0.8` (`0.8.23`) | `MIT OR Apache-2.0` | [crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) | `["std"]` | `CachePadded` cache-line isolation of shared atomic counters/slots (`thread::TaskBatch`, `thread::ScratchPool`, `frame::butce::Butce`) and `Backoff` in lock-free CAS loops | Pure Rust (internal atomic/alignment `unsafe`, `0` C code) |
| `parking_lot` | `0.12` (`0.12.5`) | `MIT OR Apache-2.0` | [Amanieu/parking_lot](https://github.com/Amanieu/parking_lot) | `[]` (`not(target_arch = "wasm32")`) | Non-poisoning compact userspace `Mutex` and `Condvar` (`sync`) across `thread::WorkerPool`, `thread::FrameSlot`, `api::Decoder`, and `decode::intra` | Pure Rust + OS futex / `libc` syscall binding on non-Wasm (`0` compiled C files) |
| `memchr` | `2` (`2.8.3`) | `Unlicense OR MIT` | [BurntSushi/memchr](https://github.com/BurntSushi/memchr) | `[]` | SIMD/SWAR byte scanning (`memchr`, `memrchr`) for ISOBMFF NUL-terminated string parsing (`avif::boxes`) and OBU metadata trailing-byte locating (`api::Decoder`) | Pure Rust (internal SIMD/pointer `unsafe`, `0` C code) |
| `rustc-hash` | `2` (`2.1.3`) | `Apache-2.0 OR MIT` | [rust-lang/rustc-hash](https://github.com/rust-lang/rustc-hash) | `[]` | Fast deterministic integer/tuple hashing (`FxBuildHasher`) for ISOBMFF `item_id` tables (`avif`) and DSP dispatch setup (`dsp`) | Safe Rust (`0` `unsafe` blocks, `0` C code) |
| `indexmap` | `2` (`2.14.2`) | `Apache-2.0 OR MIT` | [indexmap-rs/indexmap](https://github.com/indexmap-rs/indexmap) | `[]` | Order-preserving $O(1)$ hash tables and sets (`IndexMap`, `IndexSet` over Miri-verified `hashbrown`) as the single hash table engine (`avif::boxes`, `avif::decode`, `dsp`) | Pure Rust (`indexmap` `#![forbid(unsafe_code)]` over Miri-verified `hashbrown`, `0` C code) |
| `flate2` | `1.1` (`1.1.11`) | `MIT OR Apache-2.0` | [rust-lang/flate2-rs](https://github.com/rust-lang/flate2-rs) | `["rust_backend"]` | Pure-Rust bounded `deflate`/`zlib`/`gzip` decompression for HEIF compressed metadata items (`infe` `content_encoding`, `cmpC`) and `cmin` boxes (`avif::boxes`) | Safe Rust (`miniz_oxide` `#![forbid(unsafe_code)]` backend + `crc32fast`, `0` C/`libz-sys` code) |

---

## Installation

Add `av1oxide` to your `Cargo.toml`:

```toml
[dependencies]
av1oxide = "0.4"
```

To build and browse the local API documentation (`rustdoc`):

```bash
cargo doc -p av1oxide --no-deps --open
```

---

## Library Usage

### 1. Basic OBU Packet Decoding and Reading YUV Planes

```rust
use av1oxide::{BitDepth, Config, Decoder, Frame, Packet, PlaneType, Result};

fn decode_packets(packets: &[&[u8]]) -> Result<()> {
    let config = Config {
        n_threads: 4,
        max_frame_delay: 2,
        ..Config::default()
    };
    let mut decoder = Decoder::new(config)?;

    for (pts, payload) in packets.iter().enumerate() {
        decoder.send_packet(&Packet::with_timestamp(payload, pts as i64))?;
        while let Some(frame) = decoder.receive_frame()? {
            process_frame(&frame);
        }
    }

    // Drain any buffered frames at the end of the stream.
    decoder.flush();
    while let Some(frame) = decoder.receive_frame()? {
        process_frame(&frame);
    }

    Ok(())
}

fn process_frame(frame: &Frame) {
    let y_plane = frame.plane(PlaneType::Y).expect("Y plane");
    match frame.bit_depth() {
        BitDepth::Eight => {
            for y in 0..y_plane.height() {
                let _row: &[u8] = y_plane.row(y).expect("8-bit row slice");
            }
        }
        BitDepth::Ten | BitDepth::Twelve => {
            // 10/12-bit samples are stored as little-endian u16 pairs in each row.
            for y in 0..y_plane.height() {
                let row_bytes = y_plane.row(y).expect("16-bit row bytes");
                for pair in row_bytes.chunks_exact(2) {
                    let _sample = u16::from_le_bytes([pair[0], pair[1]]);
                }
            }
        }
    }
}
```

### 2. Resource Budgets and Error Recovery (`reset`) for Untrusted Input

```rust
use av1oxide::{Config, Decoder, Error, Packet, Result};

fn decode_untrusted(payload: &[u8]) -> Result<()> {
    let config = Config {
        n_threads: 2,
        max_frame_delay: 1,
        frame_size_limit: 3840 * 2160,              // Maximum 4K frame area (pixels)
        allocation_limit: Some(128 * 1024 * 1024),  // Maximum 128 MiB working memory
        ..Config::default()
    };
    let mut decoder = Decoder::new(config)?;

    match decoder.send_packet(&Packet::new(payload)) {
        Ok(()) => {
            while let Some(frame) = decoder.receive_frame()? {
                println!(
                    "Frame: {}x{}, peak memory: {} bytes",
                    frame.width(),
                    frame.height(),
                    decoder.allocation_peak()
                );
            }
        }
        Err(Error::BudgetExceeded { requested, limit }) => {
            eprintln!("Memory budget exceeded: requested={requested}, limit={limit}");
            decoder.reset();
        }
        Err(Error::Bitstream(msg)) => {
            eprintln!("Malformed bitstream: {msg}");
            decoder.reset();
        }
        Err(err) => return Err(err),
    }

    Ok(())
}
```

### 3. `libdav1d`-Compatible C-ABI Usage (`--features capi`)

Build shared (`libav1oxide.so` / `.dylib` / `.dll`) and static (`libav1oxide.a`) libraries exporting the 19 `dav1d_*` symbols:

```bash
cargo rustc --release -p av1oxide --features capi --crate-type cdylib,staticlib
```

Link any C/C++ program against `crates/av1oxide/include` (`#include <dav1d/dav1d.h>`) and `libav1oxide.a` / `libav1oxide.so`:

```c
#include <dav1d/dav1d.h>

int decode_obu_c(const uint8_t *obu_buf, size_t obu_len) {
    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = 4;

    Dav1dContext *c = NULL;
    if (dav1d_open(&c, &s) < 0) return -1;

    Dav1dData data = { 0 };
    uint8_t *dst = dav1d_data_create(&data, obu_len);
    memcpy(dst, obu_buf, obu_len);

    if (dav1d_send_data(c, &data) < 0) {
        dav1d_data_unref(&data);
        dav1d_close(&c);
        return -1;
    }

    Dav1dPicture pic = { 0 };
    while (dav1d_get_picture(c, &pic) == 0) {
        /* pic.data[0..3], pic.stride[0..2], pic.p.w, pic.p.h, pic.p.bpc */
        dav1d_picture_unref(&pic);
    }

    dav1d_close(&c);
    return 0;
}
```

### 4. `.avif` Image Decoding (`av1oxide::avif`)

```rust
use av1oxide::Result;
use av1oxide::avif::{AvifDecodeConfig, decode_rgba8, decode_with, parse_info, probe};

fn read_avif(avif_bytes: &[u8]) -> Result<()> {
    if !probe(avif_bytes) {
        return Ok(());
    }

    // Inspect image dimensions, color descriptors, alpha, and metadata without decoding pixels.
    let info = parse_info(avif_bytes)?;
    println!(
        "AVIF header: {}x{}, {:?}, {:?}, alpha={}",
        info.width, info.height, info.bit_depth, info.chroma_format, info.has_alpha
    );

    // Decode directly to tightly packed 8-bit RGBA ([R, G, B, A, ...]).
    let rgba = decode_rgba8(avif_bytes)?;
    assert_eq!(rgba.pixels.len(), (rgba.width as usize) * (rgba.height as usize) * 4);

    // Or decode with explicit thread and memory limits to planar 8/10/12-bit samples.
    let cfg = AvifDecodeConfig {
        n_threads: 4,
        max_pixels: 8192 * 8192,
        allocation_limit: Some(256 * 1024 * 1024),
        ..AvifDecodeConfig::default()
    };
    let planar = decode_with(avif_bytes, &cfg)?;
    let _rgb16 = planar.to_rgb16()?;
    Ok(())
}
```

### 5. Runnable Examples (`examples/`)

Five standalone examples are included in the crate:

- **[`examples/decode_obu.rs`](crates/av1oxide/examples/decode_obu.rs):** Decode raw AV1 OBU packets with timestamps (`Packet::with_timestamp`), inspect frame metadata, and read plane rows:
  ```bash
  cargo run --release -p av1oxide --example decode_obu
  cargo run --release -p av1oxide --example decode_obu -- <file.obu>
  ```
- **[`examples/decode_ivf.rs`](crates/av1oxide/examples/decode_ivf.rs):** Decode an `.ivf` stream frame-by-frame with a multithreaded (`n_threads: 4, max_frame_delay: 2`) decoder and optionally write planar YUV output:
  ```bash
  cargo run --release -p av1oxide --example decode_ivf
  cargo run --release -p av1oxide --example decode_ivf -- <input.ivf> [output.yuv]
  ```
- **[`examples/bounded_decoder.rs`](crates/av1oxide/examples/bounded_decoder.rs):** Enforce pre-allocation memory limits (`frame_size_limit` and `allocation_limit`), catch malformed packets without panicking, and recover the same decoder instance via `decoder.reset()`:
  ```bash
  cargo run --release -p av1oxide --example bounded_decoder
  ```
- **[`examples/c_abi_decode.rs`](crates/av1oxide/examples/c_abi_decode.rs):** End-to-end decoding through the `libdav1d` (`7.0.0`) C-ABI (`dav1d_default_settings` → `dav1d_open` → `dav1d_data_create` → `dav1d_send_data` → `dav1d_get_picture` → `dav1d_picture_unref` → `dav1d_close`):
  ```bash
  cargo run --release -p av1oxide --example c_abi_decode
  ```
- **[`examples/decode_avif.rs`](crates/av1oxide/examples/decode_avif.rs):** Inspect `.avif` headers via `avif::parse_info` and decode planar and 8-bit RGBA pixels via `avif::decode` / `avif::decode_rgba8`:
  ```bash
  cargo run --release -p av1oxide --example decode_avif
  cargo run --release -p av1oxide --example decode_avif -- <input.avif>
  ```

---

## Performance & Memory Comparison vs. `dav1d 1.5.4` and `libavif 1.4.2`

All benchmarks were measured under `100%` MD5 bit-exact output parity between `av1oxide` (`v0.4.0`, `rustc 1.100.0 --release`, `opt-level = 3`, `lto = "thin"`, `strip = "symbols"`), the reference C/ASM `dav1d 1.5.4` (`99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`), and reference C `libavif 1.4.2` (`b8551fe61aaf92733234efc5e11cdfac5a5f54b3`, built with `-DAVIF_LIBYUV=OFF -DAVIF_LIBSHARPYUV=OFF` over `dav1d 1.5.4`). File I/O, container demuxing, and MD5 verification execute outside the timed decode window; each workload and decoder runs in an isolated subprocess across `5–7` interleaved runs, reporting median (`p50`) wall time, OS Peak RSS (`MiB`), and `av1oxide` live heap peak (`MiB`).

### 1. `x86_64` — AMD Ryzen 9 9950X3D (Zen 5, `16C / 32T`, `AVX-512ICL`)

#### Single-Thread (`n_threads = 1`, `max_frame_delay = 1`) Suite Summary

| Suite | Streams / Frames | `av1oxide` Total (`ms`) | `dav1d` Total (`ms`) | Wall-Time Ratio (`av1oxide / dav1d`) | Geometric Mean | Median (`p50`) | Suite Max `av1oxide` Peak RSS (`MiB`) | Suite Max `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) | MD5 Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `19` streams / `950` decoded frames | `645.5 ms` | `430.0 ms` | **`1.501×`** | `1.552×` | `1.741×` | `19.01 MiB` (`10.90 MiB` live heap) | `9.16 MiB` | `2.08×` | `19 / 19` (`100%`) |
| **`temsilci`** | `8` streams / `485` decoded frames | `1,963.6 ms` | `1,063.1 ms` | **`1.847×`** | `1.593×` | `1.704×` | `1,231.36 MiB` (`1,212.62 MiB` live heap) | `608.98 MiB` | `2.02×` | `8 / 8` (`100%`) |

#### Single-Thread (`n_threads = 1`) Representative Streams Breakdown

| Stream | Resolution | Bit / Chroma | Frames | `av1oxide` (`ms/frame`) | `av1oxide` (`FPS`) | `dav1d` (`ms/frame`) | `dav1d` (`FPS`) | Time Ratio (`av1oxide / dav1d`) | Speedup vs. Scalar | `av1oxide` Peak RSS / Live Heap (`MiB`) | `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0.293` | `3,415.8` | `0.253` | `3,951.8` | **`1.157×`** | `2.60×` | `6.92` / `0.37` | `3.66` | `1.89×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0.296` | `3,373.9` | `0.255` | `3,919.0` | **`1.162×`** | `2.79×` | `7.48` / `0.56` | `3.91` | `1.91×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `2.845` | `351.5` | `2.169` | `461.0` | **`1.312×`** | `4.08×` | `11.20` / `3.40` | `5.41` | `2.07×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2.674` | `374.0` | `1.649` | `606.5` | **`1.622×`** | `4.17×` | `23.41` / `13.29` | `11.39` | `2.05×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0.835` | `1,197.8` | `0.467` | `2,139.8` | **`1.786×`** | `4.48×` | `14.66` / `6.79` | `6.40` | `2.29×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `246.217` | `4.1` | `125.826` | `8.0` | **`1.957×`** | `1.89×` | `1,231.36` / `1,212.62` | `608.98` | `2.02×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0.308` | `3,243.4` | `0.153` | `6,532.7` | **`2.014×`** | `4.12×` | `10.45` / `2.90` | `5.41` | `1.93×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1.354` | `738.5` | `0.659` | `1,517.9` | **`2.055×`** | `5.42×` | `19.04` / `10.90` | `9.16` | `2.08×` |

#### Multi-Thread (`AMD Ryzen 9 9950X3D`) AV1 Comparison

| Stream | Mode (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Multi-Thread) | `av1oxide` Self-Speedup | `dav1d` (Multi-Thread) | Time Ratio (`av1oxide / dav1d`) | `av1oxide` Peak RSS / Live Heap (`MiB`) | `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) | MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `10×10=100` frames) | `mfd=1, t=8` | `272.9 ms` | `257.7 ms` | `1.06×` | `242.5 ms` | **`1.06×`** | `11.36` / `3.40` | `6.72` | `1.69×` | `100%` |
| `fg10_film_grain_50` (`352×288`, `10b`, `10×15=150` frames) | `mfd=1, t=8` | `119.4 ms` | `103.4 ms` | `1.15×` | `71.5 ms` | **`1.45×`** | `14.81` / `7.14` | `7.46` | `1.98×` | `100%` |
| `ccvb_film_grain` (`640×480`, `8b`, `100×3=300` frames) | `mfd=1, t=8` | `384.2 ms` | `298.4 ms` | `1.29×` | `182.2 ms` | **`1.64×`** | `19.46` / `11.39` | `9.97` | `1.95×` | `100%` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles, `8×15=120` frames) | `mfd=0, t=16` | `1,210.3 ms` | `258.6 ms` | **`4.68×`** | `141.1 ms` | **`1.83×`** | `196.18` / `94.59` | `315.22` | **`0.62×`** | `100%` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles, `8×15=120` frames) | `mfd=4, t=16` | `1,210.3 ms` | `241.6 ms` | **`5.01×`** | `145.3 ms` | **`1.66×`** | `219.47` / `95.27` | `306.50` | **`0.72×`** | `100%` |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tiles + FG, `6` frames) | `mfd=0, t=8` | `1,500.1 ms` | `429.5 ms` | **`3.49×`** | `186.9 ms` | **`2.30×`** (`t=4`: `2.58×`) | `1,990.94` / `1,416.23` | `1,416.41` | `1.41×` | `100%` |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tiles, `10×2=20` frames) | `mfd=0, t=16` | `1,590.1 ms` | `570.0 ms` | **`2.79×`** | `165.5 ms` | **`3.44×`** | `1,321.86` / `811.50` | `1,020.45` | `1.30×` | `100%` |

#### `.avif` Image Decoding + `RGBA8` Conversion (`av1oxide::avif` vs. C `libavif 1.4.2 + dav1d 1.5.4`)

| Workload | Images / Mode | `av1oxide` Time (`ms`) | C `libavif` Time (`ms`) | Time Ratio (`av1oxide / libavif`) | `av1oxide` Peak RSS (`MiB`) | `av1oxide` Live Heap (`MiB`) | C `libavif` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / libavif`) | `RGBA8` MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** | `156` valid `.avif` images, `n_threads = 4` (`1` pass) | `3,617.2 ms` | `2,477.6 ms` | **`1.460×`** | `183.33 MiB` | `99.15 MiB` | `103.39 MiB` | **`1.77×`** | `156 / 156` (`100%`) |
| **`avif_bbb_4k_t8`** | `bbb_4k.avif` (`3840×2160`), `n_threads = 8` (`5` repeats) | `407.2 ms` | `273.7 ms` | **`1.488×`** | `75.30 MiB` | `67.06 MiB` | `51.23 MiB` | **`1.47×`** | `5 / 5` (`100%`) |

---

### 2. `aarch64` — Apple M4 Pro (`14C = 10P + 4E`, `NEON + DOTPROD`)

#### Single-Thread (`n_threads = 1`, `max_frame_delay = 1`) Suite Summary

| Suite | SIMD Mode | Streams / Frames | `av1oxide` Total (`ms`) | `dav1d` Total (`ms`) | Wall-Time Ratio (`av1oxide / dav1d`) | Geometric Mean | Median (`p50`) | Suite Max `av1oxide` Peak RSS (`MiB`) | Suite Max `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) | MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `NEON + DOTPROD` (same-ISA) | `19` streams / `950` decoded frames | `738.5 ms` | `474.1 ms` | **`1.558×`** | `1.623×` | `1.599×` | `16.52 MiB` (`10.89 MiB` live heap) | `8.91 MiB` | `1.85×` | `19 / 19` (`100%`) |
| **`temsilci`** | `NEON + DOTPROD` (same-ISA) | `8` streams / `485` decoded frames | `1,994.9 ms` | `1,123.8 ms` | **`1.775×`** | `1.560×` | `1.632×` | `1,234.05 MiB` (`1,212.62 MiB` live heap) | `610.30 MiB` | `2.02×` | `8 / 8` (`100%`) |
| **`korpus`** | `tespit` (`dav1d` + `I8MM`) | `19` streams / `950` decoded frames | `734.2 ms` | `474.5 ms` | **`1.547×`** | `1.647×` | `1.585×` | `16.50 MiB` (`10.89 MiB` live heap) | `8.89 MiB` | `1.86×` | `19 / 19` (`100%`) |
| **`temsilci`** | `tespit` (`dav1d` + `I8MM`) | `8` streams / `485` decoded frames | `1,982.5 ms` | `1,122.2 ms` | **`1.767×`** | `1.570×` | `1.622×` | `1,233.30 MiB` (`1,212.62 MiB` live heap) | `610.31 MiB` | `2.02×` | `8 / 8` (`100%`) |

#### Single-Thread (`n_threads = 1`) Representative Streams Breakdown (`NEON + DOTPROD`)

| Stream | Resolution | Bit / Chroma | Frames | `av1oxide` (`ms/frame`) | `av1oxide` (`FPS`) | `dav1d` (`ms/frame`) | `dav1d` (`FPS`) | Time Ratio (`av1oxide / dav1d`) | Speedup vs. Scalar | `av1oxide` Peak RSS / Live Heap (`MiB`) | `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0.292` | `3,423.5` | `0.274` | `3,645.0` | **`1.065×`** | `1.52×` | `4.20` / `0.36` | `2.91` | `1.45×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0.310` | `3,225.8` | `0.283` | `3,538.6` | **`1.097×`** | `1.74×` | `4.58` / `0.55` | `3.00` | `1.53×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `3.263` | `306.5` | `2.526` | `395.8` | **`1.291×`** | `2.34×` | `8.64` / `3.39` | `5.08` | `1.70×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2.760` | `362.4` | `1.947` | `513.5` | **`1.417×`** | `2.64×` | `20.70` / `13.28` | `11.56` | `1.79×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0.980` | `1,020.7` | `0.509` | `1,965.6` | **`1.926×`** | `2.98×` | `12.00` / `6.78` | `6.02` | `1.99×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `234.857` | `4.3` | `127.112` | `7.9` | **`1.848×`** | `1.68×` | `1,234.05` / `1,212.62` | `610.30` | `2.02×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0.366` | `2,735.2` | `0.182` | `5,501.9` | **`2.011×`** | `2.32×` | `8.34` / `2.90` | `4.95` | `1.68×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1.938` | `516.0` | `0.845` | `1,183.6` | **`2.294×`** | `2.98×` | `16.44` / `10.89` | `8.91` | `1.85×` |

#### Multi-Thread (`Apple M4 Pro`) AV1 Comparison

| Stream | Mode (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Multi-Thread) | `av1oxide` Self-Speedup | `dav1d` (Multi-Thread) | Time Ratio (`av1oxide / dav1d`) | `av1oxide` Peak RSS / Live Heap (`MiB`) | `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) | MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `10×10=100` frames) | `mfd=1, t=4` | `311.1 ms` | `277.9 ms` | `1.12×` | `231.1 ms` | **`1.20×`** | `9.50` / `3.39` | `5.97` | `1.59×` | `100%` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles, `8×15=120` frames) | `mfd=0, t=14` | `1,550.6 ms` | `238.2 ms` | **`6.51×`** | `137.5 ms` | **`1.73×`** | `140.00` / `94.42` | `102.77` | `1.36×` | `100%` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles, `8×15=120` frames) | `mfd=4, t=14` | `1,550.6 ms` | `239.4 ms` | **`6.48×`** | `136.9 ms` | **`1.75×`** | `139.89` / `94.69` | `103.03` | `1.36×` | `100%` |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tiles + FG, `6` frames) | `mfd=0, t=14` | `1,418.8 ms` | `147.7 ms` | **`9.60×`** | `66.2 ms` | **`2.23×`** | `2,055.83` / `1,705.67` | `1,398.83` | `1.47×` | `100%` |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tiles, `10×2=20` frames) | `mfd=0, t=14` | `1,694.6 ms` | `238.7 ms` | **`7.10×`** | `97.3 ms` | **`2.45×`** | `1,410.64` / `1,131.39` | `1,097.81` | `1.28×` | `100%` |

#### `.avif` Image Decoding + `RGBA8` Conversion (`av1oxide::avif` vs. C `libavif 1.4.2 + dav1d 1.5.4`)

| Workload | Images / Mode | `av1oxide` Time (`ms`) | C `libavif` Time (`ms`) | Time Ratio (`av1oxide / libavif`) | `av1oxide` Peak RSS (`MiB`) | `av1oxide` Live Heap (`MiB`) | C `libavif` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / libavif`) | `RGBA8` MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** | `156` valid `.avif` images, `n_threads = 4` (`1` pass) | `3,348.8 ms` | `2,600.8 ms` | **`1.288×`** | `335.95 MiB` | `106.74 MiB` | `205.16 MiB` | **`1.64×`** | `156 / 156` (`100%`) |
| **`avif_bbb_4k_t8`** | `bbb_4k.avif` (`3840×2160`), `n_threads = 8` (`5` repeats) | `366.1 ms` | `222.8 ms` | **`1.644×`** | `104.80 MiB` | `67.06 MiB` | `51.31 MiB` | **`2.04×`** | `5 / 5` (`100%`) |

---

### 3. Technical Analysis of Runtime and Memory Differences

#### Resource Checks with Negligible Runtime Cost (`~0%` Impact)

- Memory budget and frame dimension checks (`Config::allocation_limit`, `Config::frame_size_limit`) execute only during frame and buffer allocation rather than inside pixel or block loops; they have no measurable impact on frame decoding speed.

#### Measured Technical Factors Behind the Runtime Difference (`~1.50×–1.70×`)

- **Slice Bounds Checking (`23%–50%` of decode time):** High-resolution frames traverse millions of `4×4` / `8×8` blocks and neighbor tables. Whereas C `dav1d` accesses these structures through unchecked raw pointer arithmetic (`ptr += offset`), the entire non-DSP decoder core in `av1oxide` is compiled under `#![forbid(unsafe_code)]` and performs runtime bounds checks on every slice index (`slice[i]`). This accounts for the primary portion of the `~1.65×–1.71×` ratio measured between pure C and pure Rust when SIMD is disabled on both sides.
- **SIMD Entry Validation and Zero Over-Read (`~2–5 ns` per call):** C `dav1d` Assembly routines perform no entry length checks and read 16–64 bytes past the end of buffers into padding regions (over-read). Every SIMD function in `av1oxide` validates slice length preconditions on entry and uses masked or bounded loads at edges to avoid reading past slice bounds; on short (`10–20 ns`) small-block (`4×4`) calls, this validation adds `15%–25%` execution time.
- **Hand-Written Assembly vs. Compiler (`LLVM`) Code Generation:** `dav1d` uses approximately `265,000` lines of hand-written NASM/GAS Assembly that bypass standard function calling conventions (ABI) and manage CPU flag registers and 32 `zmm` registers directly, whereas `av1oxide` compiles stable Rust `core::arch` intrinsics through the standard LLVM ABI frame.

#### Measured Technical Factors Behind the Memory (`Peak RSS`) Behavior

- **Single-Thread Baseline and High-Resolution Planar Buffers (`~1.85×–2.08×` RSS):** On low- and medium-resolution streams (`16×16` to `640×480`), `av1oxide` live heap usage is `0.21–10.90 MiB`, while process resident set size (`6.9–19.0 MiB` on Linux `x86_64`, `4.2–16.5 MiB` on macOS `aarch64` vs. `2.9–9.2 MiB` for C `dav1d`) reflects the statically linked Rust runtime and multi-ISA SIMD dispatch code pages (`9.7 MiB` ELF on `x86_64`, `2.8 MiB` Mach-O on `aarch64`) compared to `dav1d`'s shared library mapping. On `8K` `4:4:4` streams with Film Grain (`test25_8k_444_fg`, `1,231–1,234 MiB` vs. `609–610 MiB`, `2.02×`), `av1oxide` allocates separate owned reference frame planes, loop-restoration stripe/CDEF staging buffers, and a separate Film Grain output copy so reference frame buffers remain immutable without aliasing raw pointers.
- **8K Multi-Thread Frame-Delay Scaling (`~1.28×–1.47×` RSS):** Under multi-threaded frame-delay decoding on `8K` `120`-tile streams (`test25_8k_120_tile` `1.28×–1.30×` RSS; `test25_8k_444_fg` `1.41×–1.47×` RSS), both decoders maintain `n_fc = 3..4` concurrent frame contexts while sharing reference frames across threads (`Arc` + `DecodeBufPool` recycling in `av1oxide`), reducing the relative RSS gap compared to single-thread execution.
- **16-Thread Multi-Tile Scratch Reuse (`0.62×–0.72×` RSS on `x86_64`):** On `1024×1024` `16`-tile `10-bit` streams at `n_threads = 16` (`test10103_coklu_tile`), `av1oxide` records `196.2–219.5 MiB` Peak RSS (`94.6–95.3 MiB` live heap) vs. `306.5–315.2 MiB` for `dav1d` (`0.62×–0.72×`) on Linux `x86_64`, because `av1oxide` reuses bounded worker scratch slots (`ScratchPool`) and recycled plane buffers (`DecodeBufPool`) on demand rather than pre-allocating full per-frame-thread tile and coefficient arenas upfront. On macOS `aarch64` (`t=14`), `16 KiB` virtual memory pages and system `libmalloc` thread-magazine caching result in `140.0 MiB` vs. `102.8 MiB` (`1.36×`, with `94.4 MiB` live heap).
- **`.avif` Container and `RGBA8` Conversion (`1.47×–2.04×` RSS):** During `.avif` decoding (`67.1 MiB` live heap on `4K` `bbb_4k.avif`, `99.2–106.7 MiB` across the `156`-image corpus), `av1oxide::avif` stores decoded planar YUVA channels as `u16` slices prior to allocating the `RGBA8` output buffer (`31.64 MiB` at `3840×2160`), whereas C `libavif` stores `8-bit` planar channels as `uint8_t` buffers before `avifImageYUVToRGB`.

---

## License

[BSD-2-Clause](LICENSE)
