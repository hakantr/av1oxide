# av1oxide (`v0.3.2`)

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
av1oxide = "0.3"
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

## Performance Comparison vs. `dav1d 1.5.4`

All benchmarks were measured under `100%` MD5 bit-exact output parity between `av1oxide` (`rustc 1.100.0 --release`) and the reference C/ASM `dav1d 1.5.4` (`99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`).

### 1. `x86_64` — AMD Ryzen 9 9950X3D (Zen 5, `16C / 32T`, `AVX-512ICL`)

#### Single-Thread (`n_threads = 1`) Suite Summary

| Suite | Streams / Frames | `av1oxide` Total (`ms`) | `dav1d` Total (`ms`) | Wall-Time Ratio (`av1oxide / dav1d`) | Geometric Mean | Median (`p50`) | MD5 Parity |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `19` streams / `964` frames | `647.7 ms` | `431.3 ms` | **`1.502×`** | `1.555×` | `1.767×` | `19 / 19` (`100%`) |
| **`temsilci`** | `8` streams / `485` frames | `1,980.4 ms` | `1,070.7 ms` | **`1.850×`** | `1.589×` | `1.699×` | `8 / 8` (`100%`) |

#### Single-Thread (`n_threads = 1`) Representative Streams Breakdown

| Stream | Resolution | Bit / Chroma | Frames | `av1oxide` (`ms/frame`) | `av1oxide` (`FPS`) | `dav1d` (`ms/frame`) | `dav1d` (`FPS`) | Ratio (`av1oxide / dav1d`) | Speedup vs. Scalar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0.292` | `3,418.8` | `0.253` | `3,957.3` | **`1.156×`** | `2.59×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0.297` | `3,370.4` | `0.255` | `3,915.4` | **`1.165×`** | `2.79×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `2.837` | `352.5` | `2.168` | `461.3` | **`1.309×`** | `4.06×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2.660` | `375.9` | `1.645` | `607.8` | **`1.617×`** | `4.19×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0.835` | `1,197.5` | `0.469` | `2,131.7` | **`1.780×`** | `4.47×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `248.828` | `4.0` | `126.890` | `7.9` | **`1.959×`** | `1.89×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0.307` | `3,254.1` | `0.154` | `6,502.0` | **`1.996×`** | `4.14×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1.372` | `728.8` | `0.670` | `1,493.2` | **`2.047×`** | `5.34×` |

#### Multi-Thread (`AMD Ryzen 9 9950X3D`) Comparison

| Stream | Mode (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Multi-Thread) | `av1oxide` Self-Speedup | `dav1d` (Multi-Thread) | Ratio (`av1oxide / dav1d`) |
|---|---|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `100×10` frames) | `mfd=1, t=8` | `272.6 ms` | `257.8 ms` | `1.06×` | `251.7 ms` | **`1.02×`** |
| `fg10_film_grain_50` (`352×288`, `10b`, `150×15` frames) | `mfd=1, t=8` | `122.4 ms` | `104.4 ms` | `1.17×` | `75.5 ms` | **`1.38×`** |
| `ccvb_film_grain` (`640×480`, `8b`, `300×3` frames) | `mfd=1, t=8` | `387.1 ms` | `310.5 ms` | `1.25×` | `203.0 ms` | **`1.53×`** |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles) | `mfd=0, t=16` | `1,217.5 ms` | `234.3 ms` | **`5.20×`** | `136.9 ms` | **`1.71×`** |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tiles + FG) | `mfd=0, t=8` | `1,476.5 ms` | `159.6 ms` | **`9.25×`** | `63.1 ms` | **`2.53×`** (`t=4`: `2.18×`) |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tiles) | `mfd=0, t=16` | `1,589.0 ms` | `401.0 ms` | **`3.96×`** | `99.9 ms` | **`4.01×`** |

---

### 2. `aarch64` — Apple M4 Pro (`14C = 10P + 4E`, `NEON + DOTPROD`)

#### Single-Thread (`n_threads = 1`) Suite Summary

| Suite | Streams / Frames | SIMD Level | `av1oxide` Total (`ms`) | `dav1d` Total (`ms`) | Wall-Time Ratio (`av1oxide / dav1d`) | Geometric Mean | Median (`p50`) | MD5 Parity |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `19` streams / `964` frames | `NEON + DOTPROD` | `796.9 ms` | `486.4 ms` | **`1.638×`** | `1.680×` | `1.786×` | `19 / 19` (`100%`) |
| **`temsilci`** | `8` streams / `485` frames | `NEON + DOTPROD` | `1,995.2 ms` | `1,117.0 ms` | **`1.786×`** | `1.675×` | `1.649×` | `8 / 8` (`100%`) |

#### Multi-Thread (`Apple M4 Pro`) Comparison

| Stream | Mode (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Multi-Thread) | `av1oxide` Self-Speedup | `dav1d` (Multi-Thread) | Ratio (`av1oxide / dav1d`) |
|---|---|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`) | `mfd=1, t=4` | `318.0 ms` | `290.7 ms` | `1.09×` | `236.9 ms` | **`1.23×`** |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles) | `mfd=0, t=14` | `1,558.9 ms` | `236.2 ms` | **`6.60×`** | `135.7 ms` | **`1.74×`** |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tiles + FG) | `mfd=0, t=14` | `1,410.1 ms` | `149.6 ms` | **`9.43×`** | `68.0 ms` | **`2.20×`** |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tiles) | `mfd=0, t=14` | `1,687.5 ms` | `233.0 ms` | **`7.24×`** | `95.0 ms` | **`2.45×`** |

---

### 3. Technical Analysis of Runtime Differences

#### Resource Checks with Negligible Runtime Cost (`~0%` Impact)

- Memory budget and frame dimension checks (`Config::allocation_limit`, `Config::frame_size_limit`) execute only during frame and buffer allocation rather than inside pixel or block loops; they have no measurable impact on frame decoding speed.

#### Measured Technical Factors Behind the Runtime Difference (`~1.50×–1.70×`)

- **Slice Bounds Checking (`23%–50%` of decode time):** High-resolution frames traverse millions of `4×4` / `8×8` blocks and neighbor tables. Whereas C `dav1d` accesses these structures through unchecked raw pointer arithmetic (`ptr += offset`), the entire non-DSP decoder core in `av1oxide` is compiled under `#![forbid(unsafe_code)]` and performs runtime bounds checks on every slice index (`slice[i]`). This accounts for the primary portion of the `~1.65×–1.71×` ratio measured between pure C and pure Rust when SIMD is disabled on both sides.
- **SIMD Entry Validation and Zero Over-Read (`~2–5 ns` per call):** C `dav1d` Assembly routines perform no entry length checks and read 16–64 bytes past the end of buffers into padding regions (over-read). Every SIMD function in `av1oxide` validates slice length preconditions on entry and uses masked or bounded loads at edges to avoid reading past slice bounds; on short (`10–20 ns`) small-block (`4×4`) calls, this validation adds `15%–25%` execution time.
- **Hand-Written Assembly vs. Compiler (`LLVM`) Code Generation:** `dav1d` uses approximately `265,000` lines of hand-written NASM/GAS Assembly that bypass standard function calling conventions (ABI) and manage CPU flag registers and 32 `zmm` registers directly, whereas `av1oxide` compiles stable Rust `core::arch` intrinsics through the standard LLVM ABI frame.

---

## License

[BSD-2-Clause](LICENSE)
