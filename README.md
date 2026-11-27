# av1oxide (`v0.5.1`)

**English** | [Türkçe](README.tr.md)

`av1oxide` is a **100% stable-Rust** (`rustc 1.100.0`, Edition 2024) reimplementation of the VideoLAN [`dav1d`](https://code.videolan.org/videolan/dav1d) (`1.5.4`) AV1 video decoder with **no external codec or media dependencies**, optional **`#![no_std] + alloc`** support, a **`libdav1d` (`1.5.4` / API `7.0.0`) drop-in binary-compatible C-ABI layer**, and a **pure-Rust `.avif` / `.avis` image and sequence decoder layer**.

- **Profiles & Bit Depths:** Main (`0`), High (`1`), Professional (`2`); `8-bit`, `10-bit`, `12-bit`; `4:2:0`, `4:2:2`, `4:4:4`, and `Monochrome` (`4:0:0`).
- **AV1 Pipeline:** OBU parsing, Sequence/Frame Headers, MSAC arithmetic entropy decoder, inverse transforms (`ITX`), intra prediction (`IPRED`, CfL, Palette), motion compensation (`MC`, Warp, Superres/Resize, IntraBC, `REFMVS`), in-loop filters (`LF`, `CDEF`, `LR` Wiener/SGR with `4`-row `LrCdefBackup`), and strip-based (`32`-row) Film Grain (`FG`) synthesis.
- **Bit-Exact Conformance:** Verified 100% MD5 bit-exact parity against `dav1d 1.5.4` across **`2,781 / 2,781` streams (`37,664 / 37,664` frames)** — covering `dav1d-test-data` (`19/19` streams, `964` frames) and Argon Streams `v2.1.1` (`2,762/2,762` streams, `36,700` frames) across all CPU capability masks and multithreaded configurations.
- **Capability-Token SIMD (`767` DSP Slots):**
  - `x86_64` (`SSE2`, `SSSE3`, `SSE4.1`, `AVX2`, `AVX-512ICL`): `767 / 767` (`100%`) SIMD slots
  - `aarch64` (`NEON`, `DOTPROD`): `767 / 767` (`100%`) SIMD slots
  - `wasm32` (`+simd128`): `682 / 767` (`88.9%`) SIMD slots under `100% #![forbid(unsafe_code)]` (`1.88×–2.40×` speedup vs. scalar WASM in Node.js and Deno; `0` `v128` instructions in scalar builds)
  - Secondary architectures (`arm32`, `riscv64`, `loongarch64`, `powerpc64le`): `100%` safe bit-exact scalar fallback
- **Multithreading & Memory Budgets:** Intra-frame (`Tile`, superblock row, in-loop filter) and inter-frame (`FrameSlot`) parallelism (`Config::n_threads`, `Config::max_frame_delay`), recycled plane pools (`DecodeBufPool`), and pre-allocation pixel and memory limits (`Config::frame_size_limit`, `Config::allocation_limit`).
- **Optional `#![no_std] + alloc` (`default = ["std"]`):** Compiling with `--no-default-features` builds the complete single-thread AV1 + AVIF decoder against `core + alloc` with zero `std` dependencies (verified on `wasm32-unknown-unknown`, `aarch64`, `armv7`, and `riscv64gc`).
- **Container Helpers (`container`):** `100% #![forbid(unsafe_code)]` zero-copy IVF (`container::ivf::{IvfParser, IvfReader}`) and AV1 Annex-B (`container::annexb::{AnnexBDemuxer, read_leb128}`) demuxers.
- **`libdav1d` (`1.5.4` / API `7.0.0`) Drop-In C-ABI (`capi`):** Implements all 19 official `DAV1D_API` functions (`dav1d_open`, `dav1d_send_data`, `dav1d_get_picture`, `dav1d_apply_grain`, `dav1d_parse_sequence_header`, etc.), ships the official `include/dav1d/*.h` headers, and declares `[package.metadata.capi]` / `dav1d.pc` metadata — verified by dynamically linking unmodified C `libavif 1.4.2` against `libdav1d.so.7` (Linux `x86_64`) and `libdav1d.7.dylib` (macOS `arm64`) (`162 / 162` corpus parity) and an independent multithreaded C IVF client.
- **Pure-Rust `.avif` / `.avis` Reader (`avif`):** `100% #![forbid(unsafe_code)]` ISOBMFF/HEIF/MIAF/AVIF parser with native `u8` 8-bit plane storage (`AvifPlane::U8`/`U16`) and strip-parallel YUVA→RGB conversion, single `av01` items, multithreaded `grid` tile stitching, `iden` chains, auxiliary alpha (`auxl`/`auxC`), `clap`/`irot`/`imir` transforms, ISO 21496-1 / Ultra HDR `tmap` Gain-Map (`9 / 9` parity), streaming animated `.avis` decoding (`AvifSequenceDecoder`, `6 / 6` animated streams + `nth_frame` seeking), `ICC`/`Exif`/`XMP` metadata (including bounded `defl`/`zlib`/`gzip` compressed HEIF metadata and `cmin` boxes under `std`), and linear + nonlinear ITU-T H.273 conversions (`ICtCp` `mc = 14` PQ/HLG and `BT.2020 CL` `mc = 10`, alongside `RGB8`/`RGBA8`/`RGB16`/`RGBA16`) — verified `100%` MD5 bit-exact parity against reference C `libavif 1.4.2` across `162 / 162` still files (`156` valid + `6` negative, `9 / 9` Gain-Map) and `6 / 6` animated `.avif` sequences.
- **Separate `image` Ecosystem Adapter (`av1oxide-image`):** Workspace crate [`crates/av1oxide-image`](crates/av1oxide-image/README.md) (`v0.1.1`, `100% #![forbid(unsafe_code)]`) provides `image::ImageDecoder` (`Rgb8`/`Rgba8`/`Rgb16`/`Rgba16`, `icc_profile`, `exif_metadata`, `xmp_metadata`, `Limits`) and `register_decoding_hook` without adding `image` to `av1oxide`'s dependency graph.

---

## Dependency Policy

`av1oxide` does not depend on any third-party codec, image, multimedia, or C media library crates; all AV1 bitstream/entropy/DSP decoding, ISOBMFF/AVIF/IVF/Annex-B container parsing, ITU-T H.273 color conversion, and `libdav1d` C-ABI layers are implemented in-tree. All external dependencies are strictly isolated as internal implementation details (`pub(crate)` / private); no types from any dependency crate are re-exported or exposed in `av1oxide`'s public (`pub`) API.

For low-level concurrency, memory layout, and decompression primitives, `av1oxide` uses only foundational crates maintained by the `rust-lang` organization and Rust Core Team members with `default-features = false`:

| Crate | Version | License | Source | Features (`default-features = false`) | Purpose in `av1oxide` | Unsafe / Native Code |
|---|---|---|---|---|---|---|
| `crossbeam-utils` | `0.8` (`0.8.23`) | `MIT OR Apache-2.0` | [crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) | `[]` (`std` enabled via `av1oxide/std`) | `CachePadded` cache-line isolation (`thread::TaskBatch`, `thread::ScratchPool`, `frame::butce::Butce`), `AtomicCell` (`no_std` sync), and `Backoff` in lock-free CAS loops | Pure Rust (internal atomic/alignment `unsafe`, `0` C code) |
| `parking_lot` | `0.12` (`0.12.5`) | `MIT OR Apache-2.0` | [Amanieu/parking_lot](https://github.com/Amanieu/parking_lot) | `[]` (optional: `std` & `not(target_arch = "wasm32")`) | Non-poisoning compact userspace `Mutex` and `Condvar` (`sync`) across `thread::WorkerPool`, `thread::FrameSlot`, `api::Decoder`, and `decode::intra` | Pure Rust + OS futex / `libc` syscall binding on non-Wasm (`0` compiled C files) |
| `memchr` | `2` (`2.8.3`) | `Unlicense OR MIT` | [BurntSushi/memchr](https://github.com/BurntSushi/memchr) | `[]` | SIMD/SWAR byte scanning (`memchr`, `memrchr`) for ISOBMFF NUL-terminated string parsing (`avif::boxes`) and OBU metadata trailing-byte locating (`api::Decoder`) | Pure Rust (internal SIMD/pointer `unsafe`, `0` C code) |
| `rustc-hash` | `2` (`2.1.3`) | `Apache-2.0 OR MIT` | [rust-lang/rustc-hash](https://github.com/rust-lang/rustc-hash) | `[]` | Fast deterministic integer/tuple hashing (`FxBuildHasher`) for ISOBMFF `item_id` tables (`avif`) and DSP dispatch setup (`dsp`) | Safe Rust (`0` `unsafe` blocks, `0` C code) |
| `indexmap` | `2` (`2.14.2`) | `Apache-2.0 OR MIT` | [indexmap-rs/indexmap](https://github.com/indexmap-rs/indexmap) | `[]` | Order-preserving $O(1)$ hash tables and sets (`IndexMap`, `IndexSet` over Miri-verified `hashbrown`) as the single hash table engine (`avif::boxes`, `avif::decode`, `dsp`) | Pure Rust (`indexmap` `#![forbid(unsafe_code)]` over Miri-verified `hashbrown`, `0` C code) |
| `flate2` | `1.1` (`1.1.11`) | `MIT OR Apache-2.0` | [rust-lang/flate2-rs](https://github.com/rust-lang/flate2-rs) | `["rust_backend"]` (optional: `std`) | Pure-Rust bounded `deflate`/`zlib`/`gzip` decompression for HEIF compressed metadata items (`infe` `content_encoding`, `cmpC`) and `cmin` boxes (`avif::boxes`) | Safe Rust (`miniz_oxide` `#![forbid(unsafe_code)]` backend + `crc32fast`, `0` C/`libz-sys` code) |

---

## Installation

Add `av1oxide` to your `Cargo.toml`:

```toml
[dependencies]
av1oxide = "0.5"
```

For `#![no_std] + alloc` environments (single-thread AV1 + AVIF without `std`, `parking_lot`, or `flate2`):

```toml
[dependencies]
av1oxide = { version = "0.5", default-features = false }
```

To integrate `av1oxide::avif` with the `image` (`0.25`) crate without altering `av1oxide`'s dependency tree, use the separate `av1oxide-image` crate ([`crates/av1oxide-image/README.md`](crates/av1oxide-image/README.md)).

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

Install a complete drop-in `libdav1d` (`1.5.4` / API `7.0.0`) prefix (`include/dav1d/*.h`, versioned `libdav1d.so.7` / `libdav1d.7.dylib` with proper `SONAME`/`install_name`, `libdav1d.a`, and `lib/pkgconfig/dav1d.pc`) using the official installer example:

```bash
cargo run --locked --release -p av1oxide --example install_capi_prefix -- /path/to/prefix
PKG_CONFIG_PATH=/path/to/prefix/lib/pkgconfig pkg-config --modversion --cflags --libs dav1d
```

Alternatively, build raw shared (`libav1oxide.so` / `.dylib` / `.dll`) and static (`libav1oxide.a`) libraries directly:

```bash
cargo rustc --release -p av1oxide --features capi --crate-type cdylib,staticlib
```

Link any C/C++ program against `<prefix>/include` (`#include <dav1d/dav1d.h>`) and `pkg-config --cflags --libs dav1d` (or `libdav1d.a` / `-ldav1d`):

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

### 4. `.avif` Still, Gain-Map, and Animated `.avis` Decoding (`av1oxide::avif`)

```rust
use av1oxide::Result;
use av1oxide::avif::{
    AvifDecodeConfig, AvifSequenceDecoder, decode_rgba8, decode_with, parse_info, probe,
};

fn read_avif(avif_bytes: &[u8]) -> Result<()> {
    if !probe(avif_bytes) {
        return Ok(());
    }

    // Inspect image dimensions, color descriptors, alpha, gain map, and metadata without decoding pixels.
    let info = parse_info(avif_bytes)?;
    println!(
        "AVIF header: {}x{}, {:?}, {:?}, alpha={}, gain_map={}",
        info.width, info.height, info.bit_depth, info.chroma_format, info.has_alpha, info.has_gain_map
    );

    // Decode directly to tightly packed 8-bit RGBA ([R, G, B, A, ...]).
    let rgba = decode_rgba8(avif_bytes)?;
    assert_eq!(rgba.pixels.len(), (rgba.width as usize) * (rgba.height as usize) * 4);

    // Decode with explicit thread/memory limits and optional ISO 21496-1 gain map.
    let cfg = AvifDecodeConfig {
        n_threads: 4,
        max_pixels: 8192 * 8192,
        allocation_limit: Some(256 * 1024 * 1024),
        decode_gain_map: true,
        ..AvifDecodeConfig::default()
    };
    let planar = decode_with(avif_bytes, &cfg)?;
    let _rgb16 = planar.to_rgb16()?;
    let _gain_map = planar.gain_map();

    // Stream animated .avis frames without buffering the full sequence in memory.
    let mut seq = AvifSequenceDecoder::with_config(avif_bytes, cfg)?;
    while let Some(frame) = seq.next_frame()? {
        let _frame_rgba = frame.image.to_rgba8()?;
    }
    Ok(())
}
```

### 5. IVF and Annex-B Demuxing (`av1oxide::container`)

```rust
use av1oxide::container::annexb::AnnexBDemuxer;
use av1oxide::container::ivf::IvfParser;
use av1oxide::{Config, Decoder, Packet, Result};

fn decode_ivf_slice(ivf_bytes: &[u8]) -> Result<()> {
    let mut parser = IvfParser::new(ivf_bytes)?;
    let mut decoder = Decoder::new(Config::default())?;
    for pkt in &mut parser {
        let pkt = pkt?;
        decoder.send_packet(&Packet::with_timestamp(pkt.data, pkt.timestamp as i64))?;
        while let Some(_frame) = decoder.receive_frame()? {}
    }
    decoder.flush();
    while let Some(_frame) = decoder.receive_frame()? {}
    Ok(())
}

fn demux_annexb(annexb_bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    AnnexBDemuxer::new(annexb_bytes).demux_all()
}
```

### 6. Runnable Examples (`examples/`)

Five standalone examples are included in the crate:

- **[`examples/decode_obu.rs`](crates/av1oxide/examples/decode_obu.rs):** Decode raw AV1 OBU packets with timestamps (`Packet::with_timestamp`), inspect frame metadata, and read plane rows:
  ```bash
  cargo run --release -p av1oxide --example decode_obu
  cargo run --release -p av1oxide --example decode_obu -- <file.obu>
  ```
- **[`examples/decode_ivf.rs`](crates/av1oxide/examples/decode_ivf.rs):** Decode an `.ivf` stream frame-by-frame via `container::ivf::IvfParser` with a multithreaded (`n_threads: 4, max_frame_delay: 2`) decoder and optionally write planar YUV output:
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
- **[`examples/install_capi_prefix.rs`](crates/av1oxide/examples/install_capi_prefix.rs):** Build and install a drop-in `libdav1d` (`1.5.4` / API `7.0.0`) prefix (`include/dav1d/*.h`, `libdav1d.so.7` / `libdav1d.7.dylib`, `libdav1d.a`, `lib/pkgconfig/dav1d.pc`):
  ```bash
  cargo run --locked --release -p av1oxide --example install_capi_prefix -- /path/to/prefix
  ```
- **[`examples/decode_avif.rs`](crates/av1oxide/examples/decode_avif.rs):** Inspect `.avif` headers via `avif::parse_info` and decode planar and 8-bit RGBA pixels via `avif::decode` / `avif::decode_rgba8`:
  ```bash
  cargo run --release -p av1oxide --example decode_avif
  cargo run --release -p av1oxide --example decode_avif -- <input.avif>
  ```

---

## Performance & Memory Comparison vs. `dav1d 1.5.4` and `libavif 1.4.2`

All benchmarks were measured under `100%` MD5 bit-exact output parity between `av1oxide` (`rustc 1.100.0 --release`, `opt-level = 3`, `lto = "thin"`, `strip = "symbols"`), reference C/ASM `dav1d 1.5.4` (`99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`), and reference C `libavif 1.4.2` (`b8551fe61aaf92733234efc5e11cdfac5a5f54b3` over `dav1d 1.5.4`). For `.avif`, both the bit-exact reference configuration (`-DAVIF_LIBYUV=OFF`, `avoidLibYUV = AVIF_TRUE`, `BILINEAR`, `100%` MD5 parity) and the default production configuration (`-DAVIF_LIBYUV=LOCAL`, `libyuv 1960` commit `26e56be0f`, `avoidLibYUV = AVIF_FALSE`, `AUTOMATIC`) are reported alongside a 3-stage pipeline breakdown (`parse`, `AV1 → YUV`, `YUVA → RGBA8`). File I/O, container demuxing, and MD5 verification execute outside the timed decode window; each workload and decoder runs in an isolated subprocess across `5–7` interleaved runs, reporting median (`p50`) wall time, OS Peak RSS (`MiB`), and `av1oxide` live heap peak (`MiB`). The `x86_64` (`AMD Ryzen 9 9950X3D`) tables below report **`v0.5.1`** measurements; the `aarch64` (`Apple M4 Pro`) section retains the **`v0.5.0`** baseline pending the `v0.5.1` Apple Silicon rerun.

### 1. `x86_64` — AMD Ryzen 9 9950X3D (Zen 5, `16C / 32T`, `AVX-512ICL`, `v0.5.1`)

#### Single-Thread (`n_threads = 1`, `max_frame_delay = 1`) Suite Summary

| Suite | Streams / Frames | `av1oxide` Total (`ms`) | `dav1d` Total (`ms`) | Wall-Time Ratio (`av1oxide / dav1d`) | Geometric Mean | Median (`p50`) | Suite Max `av1oxide` Peak RSS (`MiB`) | Suite Max `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) | MD5 Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `19` streams / `950` decoded frames | `639.5 ms` | `431.3 ms` | **`1.483×`** | `1.529×` | `1.746×` | `14.97 MiB` (`6.42 MiB` live heap) | `9.16 MiB` | `1.63×` | `19 / 19` (`100%`) |
| **`temsilci`** | `8` streams / `485` decoded frames | `1,842.8 ms` | `1,071.8 ms` | **`1.719×`** | `1.549×` | `1.669×` | `667.51 MiB` (`644.76 MiB` live heap) | `608.98 MiB` | `1.10×` | `8 / 8` (`100%`) |

#### Single-Thread (`n_threads = 1`) Representative Streams Breakdown

| Stream | Resolution | Bit / Chroma | Frames | `av1oxide` (`ms/frame`) | `av1oxide` (`FPS`) | `dav1d` (`ms/frame`) | `dav1d` (`FPS`) | Time Ratio (`av1oxide / dav1d`) | Speedup vs. Scalar | `av1oxide` Peak RSS / Live Heap (`MiB`) | `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0.286` | `3,497.9` | `0.254` | `3,945.3` | **`1.128×`** | `2.77×` | `7.47` / `0.29` | `3.66` | `2.04×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0.288` | `3,475.5` | `0.256` | `3,911.1` | **`1.125×`** | `2.99×` | `7.97` / `0.41` | `3.91` | `2.04×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `2.811` | `355.7` | `2.171` | `460.5` | **`1.295×`** | `4.14×` | `10.64` / `2.22` | `5.41` | `1.97×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2.608` | `383.4` | `1.651` | `605.7` | **`1.580×`** | `4.26×` | `16.93` / `8.25` | `11.39` | `1.49×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0.825` | `1,212.1` | `0.469` | `2,131.7` | **`1.759×`** | `4.52×` | `12.25` / `4.22` | `6.41` | `1.91×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `226.928` | `4.4` | `127.165` | `7.9` | **`1.784×`** | `1.99×` | `667.51` / `644.76` | `608.98` | `1.10×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0.302` | `3,309.8` | `0.153` | `6,538.8` | **`1.976×`** | `4.17×` | `9.72` / `1.69` | `5.41` | `1.80×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1.361` | `734.9` | `0.660` | `1,513.9` | **`2.060×`** | `5.38×` | `14.91` / `6.42` | `9.16` | `1.63×` |

#### Multi-Thread (`AMD Ryzen 9 9950X3D`) AV1 Comparison

| Stream | Mode (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Multi-Thread) | `av1oxide` Self-Speedup | `dav1d` (Multi-Thread) | Time Ratio (`av1oxide / dav1d`) | `av1oxide` Peak RSS / Live Heap (`MiB`) | `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) | MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `10×10=100` frames) | `mfd=1, t=8` | `270.1 ms` | `259.9 ms` | `1.04×` | `260.5 ms` | **`1.00×`** | `10.46` / `2.24` | `6.72` | `1.56×` | `100%` |
| `fg10_film_grain_50` (`352×288`, `10b`, `10×15=150` frames) | `mfd=1, t=8` | `120.3 ms` | `124.4 ms` | `0.97×` | `74.5 ms` | **`1.67×`** | `12.87` / `4.70` | `7.47` | `1.72×` | `100%` |
| `ccvb_film_grain` (`640×480`, `8b`, `100×3=300` frames) | `mfd=1, t=8` | `385.7 ms` | `333.4 ms` | `1.16×` | `188.1 ms` | **`1.77×`** | `15.21` / `6.86` | `9.72` | `1.56×` | `100%` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles, `8×15=120` frames) | `mfd=0, t=16` | `1,184.4 ms` | `222.6 ms` | **`5.32×`** | `144.7 ms` | **`1.54×`** | `163.25` / `75.83` | `296.49` | **`0.55×`** | `100%` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles, `8×15=120` frames) | `mfd=4, t=16` | `1,184.4 ms` | `220.9 ms` | **`5.36×`** | `142.4 ms` | **`1.55×`** | `155.28` / `76.02` | `287.26` | **`0.54×`** | `100%` |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tiles + FG, `6` frames) | `mfd=0, t=8` | `1,343.2 ms` | `356.6 ms` | **`3.77×`** | `186.7 ms` | **`1.91×`** (`t=4`: `2.21×`) | `1,079.75` / `921.64` | `1,410.41` | **`0.77×`** (`t=4`: `0.90×`) | `100%` |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tiles, `10×2=20` frames) | `mfd=0, t=16` | `1,418.0 ms` | `395.7 ms` | **`3.58×`** | `166.5 ms` | **`2.38×`** | `907.98` / `575.31` | `1,020.44` | **`0.89×`** | `100%` |

#### `.avif` Image Decoding + `RGBA8` Conversion (`av1oxide::avif` vs. C `libavif 1.4.2 + dav1d 1.5.4`)

Bit-exact reference comparison (`-DAVIF_LIBYUV=OFF`, `avoidLibYUV = AVIF_TRUE`, `BILINEAR`, `100%` `RGBA8` MD5 parity):

| Workload | Images / Mode | `av1oxide` Time (`ms`) | C `libavif` (`libyuv=OFF`) Time (`ms`) | Time Ratio (`av1oxide / libavif`) | `av1oxide` Peak RSS (`MiB`) | `av1oxide` Live Heap (`MiB`) | C `libavif` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / libavif`) | `RGBA8` MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** | `156` valid `.avif` images, `n_threads = 4` (`1` pass) | `2,207.3 ms` | `2,488.0 ms` | **`0.887×`** | `126.01 MiB` | `51.69 MiB` | `103.24 MiB` | **`1.22×`** | `156 / 156` (`100%`) |
| **`avif_bbb_4k_t8`** | `bbb_4k.avif` (`3840×2160`), `n_threads = 8` (`5` repeats) | `87.9 ms` | `273.4 ms` | **`0.322×`** | `62.66 MiB` | `43.53 MiB` | `51.22 MiB` | **`1.22×`** | `5 / 5` (`100%`) |

Production (`libyuv = ON`, `libyuv 1960`, `avoidLibYUV = AVIF_FALSE`, `AUTOMATIC`) comparison and 3-stage pipeline breakdown (`5` interleaved rounds):

| Workload | Implementation / Configuration | `parse` (`ms`) | `AV1 → YUV` (`ms`) | `YUVA → RGBA8` (`ms`) | `cleanup` (`ms`) | Total (`ms`) | Ratio vs. `libyuv=OFF` / `libyuv=ON` | Peak RSS (`MiB`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** (`t=4`) | `av1oxide v0.5.1` (`AvifPrepared` + `to_rgba8`) | `0.68` | `2,055.01` | `176.20` | `0.10` | **`2,233.53`** | **`0.907×`** / **`1.169×`** | `132.85` (`26.31` heap) |
| **`avif_corpus_156_t4`** (`t=4`) | `av1oxide v0.5.1` (`AvifPrepared` + `write_rgba8_into`, reused buffer) | `0.68` | `2,065.12` | `154.61` | `3.44` | **`2,222.80`** | **`0.902×`** / **`1.164×`** | `165.10` (`53.83` heap) |
| **`avif_corpus_156_t4`** (`t=4`) | C `libavif 1.4.2` (`libyuv = OFF`, `BILINEAR`, bit-exact) | `0.65` | `1,847.97` | `603.41` | `12.43` | `2,463.36` | `1.000×` / `1.290×` | `102.89` |
| **`avif_corpus_156_t4`** (`t=4`) | C `libavif 1.4.2` (`libyuv = ON`, `1960`, `AUTOMATIC`, production) | `0.69` | `1,842.45` | `59.91` | `11.33` | `1,910.03` | `0.775×` / `1.000×` | `113.99` |
| **`avif_bbb_4k_t8`** (`t=8, 5×`) | `av1oxide v0.5.1` (`AvifPrepared` + `to_rgba8`) | `0.10` | `61.14` | `29.19` | `0.01` | **`90.03`** | **`0.331×`** / **`1.652×`** | `63.57` (`43.50` heap) |
| **`avif_bbb_4k_t8`** (`t=8, 5×`) | `av1oxide v0.5.1` (`AvifPrepared` + `write_rgba8_into`, reused buffer) | `0.09` | `69.58` | `23.30` | `3.48` | `96.82` | `0.356×` / `1.776×` | `109.54` (`63.08` heap) |
| **`avif_bbb_4k_t8`** (`t=8, 5×`) | C `libavif 1.4.2` (`libyuv = OFF`, `BILINEAR`, bit-exact) | `0.06` | `37.79` | `235.28` | `0.84` | `272.36` | `1.000×` / `4.996×` | `51.13` |
| **`avif_bbb_4k_t8`** (`t=8, 5×`) | C `libavif 1.4.2` (`libyuv = ON`, `1960`, `AUTOMATIC`, production) | `0.05` | `35.68` | `17.28` | `0.70` | `54.51` | `0.200×` / `1.000×` | `51.67` |

---

### 2. `aarch64` — Apple M4 Pro (`14C = 10P + 4E`, `NEON + DOTPROD`, `v0.5.0` Baseline)

#### Single-Thread (`n_threads = 1`, `max_frame_delay = 1`) Suite Summary

| Suite | SIMD Mode | Streams / Frames | `av1oxide` Total (`ms`) | `dav1d` Total (`ms`) | Wall-Time Ratio (`av1oxide / dav1d`) | Geometric Mean | Median (`p50`) | Suite Max `av1oxide` Peak RSS (`MiB`) | Suite Max `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) | MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `NEON + DOTPROD` (same-ISA) | `19` streams / `950` decoded frames | `760.3 ms` | `503.7 ms` | **`1.509×`** | `1.540×` | `1.503×` | `12.19 MiB` (`6.41 MiB` live heap) | `8.88 MiB` | `1.37×` | `19 / 19` (`100%`) |
| **`temsilci`** | `NEON + DOTPROD` (same-ISA) | `8` streams / `485` decoded frames | `1,964.3 ms` | `1,163.4 ms` | **`1.688×`** | `1.504×` | `1.555×` | `666.14 MiB` (`644.75 MiB` live heap) | `610.30 MiB` | `1.09×` | `8 / 8` (`100%`) |
| **`korpus`** | `tespit` (`dav1d` + `I8MM`) | `19` streams / `950` decoded frames | `759.9 ms` | `500.4 ms` | **`1.519×`** | `1.580×` | `1.522×` | `12.17 MiB` (`6.41 MiB` live heap) | `8.92 MiB` | `1.36×` | `19 / 19` (`100%`) |
| **`temsilci`** | `tespit` (`dav1d` + `I8MM`) | `8` streams / `485` decoded frames | `1,961.2 ms` | `1,157.9 ms` | **`1.694×`** | `1.512×` | `1.553×` | `666.16 MiB` (`644.75 MiB` live heap) | `610.31 MiB` | `1.09×` | `8 / 8` (`100%`) |

#### Single-Thread (`n_threads = 1`) Representative Streams Breakdown (`NEON + DOTPROD`)

| Stream | Resolution | Bit / Chroma | Frames | `av1oxide` (`ms/frame`) | `av1oxide` (`FPS`) | `dav1d` (`ms/frame`) | `dav1d` (`FPS`) | Time Ratio (`av1oxide / dav1d`) | Speedup vs. Scalar | `av1oxide` Peak RSS / Live Heap (`MiB`) | `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0.297` | `3,367.3` | `0.272` | `3,676.3` | **`1.092×`** | `1.54×` | `4.25` / `0.28` | `2.91` | `1.46×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0.335` | `2,981.4` | `0.300` | `3,327.8` | **`1.116×`** | `1.58×` | `4.52` / `0.41` | `3.00` | `1.51×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `3.204` | `312.2` | `2.561` | `390.5` | **`1.251×`** | `2.50×` | `7.44` / `2.21` | `5.08` | `1.46×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2.839` | `352.2` | `2.038` | `490.8` | **`1.393×`** | `2.55×` | `15.36` / `8.25` | `11.56` | `1.33×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0.971` | `1,030.4` | `0.527` | `1,897.5` | **`1.842×`** | `3.03×` | `9.30` / `4.22` | `6.02` | `1.55×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `229.213` | `4.4` | `131.818` | `7.6` | **`1.739×`** | `1.73×` | `666.14` / `644.75` | `610.30` | `1.09×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0.363` | `2,753.6` | `0.212` | `4,725.9` | **`1.716×`** | `2.42×` | `6.91` / `1.78` | `4.98` | `1.39×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1.968` | `508.2` | `0.876` | `1,142.2` | **`2.247×`** | `2.96×` | `12.16` / `6.41` | `8.88` | `1.37×` |

#### Multi-Thread (`Apple M4 Pro`) AV1 Comparison

| Stream | Mode (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Multi-Thread) | `av1oxide` Self-Speedup | `dav1d` (Multi-Thread) | Time Ratio (`av1oxide / dav1d`) | `av1oxide` Peak RSS / Live Heap (`MiB`) | `dav1d` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / dav1d`) | MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `10×10=100` frames) | `mfd=1, t=4` | `317.4 ms` | `289.4 ms` | `1.10×` | `240.4 ms` | **`1.20×`** | `8.44` / `2.25` | `5.97` | `1.41×` | `100%` |
| `fg10_film_grain_50` (`352×288`, `10b`, `10×15=150` frames) | `mfd=1, t=8` | `144.1 ms` | `130.7 ms` | `1.10×` | `75.0 ms` | **`1.74×`** | `13.44` / `4.71` | `7.97` | `1.69×` | `100%` |
| `ccvb_film_grain` (`640×480`, `8b`, `100×3=300` frames) | `mfd=1, t=8` | `584.6 ms` | `440.9 ms` | `1.33×` | `205.7 ms` | **`2.14×`** | `16.83` / `7.00` | `11.02` | `1.53×` | `100%` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles, `8×15=120` frames) | `mfd=0, t=14` | `1,344.8 ms` | `205.6 ms` | **`6.54×`** | `136.9 ms` | **`1.50×`** | `103.72` / `67.50` | `104.89` | **`0.99×`** | `100%` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tiles, `8×15=120` frames) | `mfd=4, t=14` | `1,344.8 ms` | `205.5 ms` | **`6.54×`** | `137.8 ms` | **`1.49×`** | `103.16` / `67.19` | `104.73` | **`0.98×`** | `100%` |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tiles + FG, `6` frames) | `mfd=0, t=14` | `1,383.0 ms` | `140.2 ms` | **`9.87×`** | `69.3 ms` | **`2.02×`** | `1,413.62` / `1,140.47` | `1,398.67` | **`1.01×`** (`t=8`: `0.90×`) | `100%` |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tiles, `10×2=20` frames) | `mfd=0, t=14` | `1,347.1 ms` | `213.5 ms` | **`6.31×`** | `98.2 ms` | **`2.17×`** | `1,031.45` / `810.51` | `1,097.81` | **`0.94×`** | `100%` |

#### `.avif` Image Decoding + `RGBA8` Conversion (`Apple M4 Pro`)

| Workload | Images / Mode | `av1oxide` Time (`ms`) | C `libavif` Time (`ms`) | Time Ratio (`av1oxide / libavif`) | `av1oxide` Peak RSS (`MiB`) | `av1oxide` Live Heap (`MiB`) | C `libavif` Peak RSS (`MiB`) | RSS Ratio (`av1oxide / libavif`) | `RGBA8` MD5 Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** | `156` valid `.avif` images, `n_threads = 4` (`1` pass) | `2,353.9 ms` | `2,636.1 ms` | **`0.893×`** | `266.45 MiB` | `82.33 MiB` | `205.11 MiB` | **`1.30×`** | `156 / 156` (`100%`) |
| **`avif_bbb_4k_t8`** | `bbb_4k.avif` (`3840×2160`), `n_threads = 8` (`5` repeats) | `80.2 ms` | `225.4 ms` | **`0.356×`** | `82.14 MiB` | `43.52 MiB` | `51.28 MiB` | **`1.60×`** | `5 / 5` (`100%`) |

---

### 3. Technical Analysis of Runtime and Memory Differences

#### Resource Checks with Negligible Runtime Cost (`~0%` Impact)

- Memory budget and frame dimension checks (`Config::allocation_limit`, `Config::frame_size_limit`) execute only during frame and buffer allocation rather than inside pixel or block loops; they have no measurable impact on frame decoding speed.

#### Measured Technical Factors Behind the AV1 Runtime Difference (`~1.48×–1.72×`)

- **Slice Bounds Checking (`23%–50%` of decode time):** High-resolution frames traverse millions of `4×4` / `8×8` blocks and neighbor tables. Whereas C `dav1d` accesses these structures through unchecked raw pointer arithmetic (`ptr += offset`), the entire non-DSP decoder core in `av1oxide` is compiled under `#![forbid(unsafe_code)]` and performs runtime bounds checks on every slice index (`slice[i]`). This accounts for the primary portion of the `~1.65×–1.71×` ratio measured between pure C and pure Rust when SIMD is disabled on both sides.
- **SIMD Entry Validation and Zero Over-Read (`~2–5 ns` per call):** C `dav1d` Assembly routines perform no entry length checks and read 16–64 bytes past the end of buffers into padding regions (over-read). Every SIMD function in `av1oxide` validates slice length preconditions on entry and uses masked or bounded loads at edges to avoid reading past slice bounds; on short (`10–20 ns`) small-block (`4×4`) calls, this validation adds `15%–25%` execution time.
- **Hand-Written Assembly vs. Compiler (`LLVM`) Code Generation:** `dav1d` uses approximately `265,000` lines of hand-written NASM/GAS Assembly that bypass standard function calling conventions (ABI) and manage CPU flag registers and 32 `zmm` registers directly, whereas `av1oxide` compiles stable Rust `core::arch` intrinsics through the standard LLVM ABI frame.

#### Measured Technical Factors Behind the Memory (`Peak RSS`) and `.avif` Behavior

- **Single-Thread Baseline and Binary Code Pages (`1.33×–2.04×` RSS on small streams):** On low- and medium-resolution streams (`16×16` to `640×480`), `av1oxide` live heap usage is `0.28–8.25 MiB` across both architectures, while process resident set size (`7.5–16.9 MiB` on Linux `x86_64`, `4.2–15.4 MiB` on macOS `aarch64`) reflects the statically linked Rust runtime and SIMD dispatch code pages (`10.2 MiB` ELF on `x86_64`, `3.4 MiB` Mach-O on `aarch64`) compared to `dav1d`'s smaller C binary mapping.
- **8K Strip-Based Film Grain, Wavefront Tiles, and In-Place Loop Filters (`1.09×–1.10×` single-thread, `0.77×–1.01×` multi-thread RSS):** Replacing full-frame `pre_cdef` copies with a `4`-row `LrCdefBackup` boundary buffer, executing `LoopFilter`/`CDEF`/`LR` in-place across parallel strips, and scheduling tile decoding in wavefronts reduced `test25_8k_444_fg` (`7680×4320`, `4:4:4` + FG) multi-thread Peak RSS to `1,079.8 MiB` (`0.77×` at `t=8` on `x86_64`; `t=4`: `0.90×`) and `test25_8k_120_tile` (`t=16` on `x86_64`) decode time from `477.7 ms` (`2.83×`) in `v0.5.0` to `395.7 ms` (`2.38×`, live heap `575.3 MiB`, Peak RSS `908.0 MiB` / `0.89×` vs. `dav1d`).
- **Multi-Tile Scratch Reuse (`0.54×–0.55×` RSS on `x86_64`, `0.98×–0.99×` on Apple M4 Pro):** On `1024×1024` `16`-tile `10-bit` streams (`test10103_coklu_tile`), `av1oxide` records `155.3–163.3 MiB` Peak RSS (`75.8–76.0 MiB` live heap) vs. `287.3–296.5 MiB` for `dav1d` (`0.54×–0.55×`) at `t=16` on Linux `x86_64`, and `103.2–103.7 MiB` Peak RSS (`67.2–67.5 MiB` live heap) vs. `104.7–104.9 MiB` for `dav1d` (`0.98×–0.99×`) at `t=14` on Apple M4 Pro, by reusing bounded worker scratch slots (`ScratchPool`) and recycled plane buffers (`DecodeBufPool`) on demand.
- **`.avif` Native `u8` Planes, `libyuv=OFF` vs. `libyuv=ON`, and Stage Breakdown:** Storing `8-bit` AVIF channels natively as `u8` (`AvifPlane::U8`) and executing specialized `(9, 3, 3, 1)/16` row-pair YUVA→RGB conversion across `std::thread::scope` strips reduced `x86_64` `avif_corpus_156_t4` time to `2,207.3 ms` (`0.887×` vs. C `libavif` `libyuv=OFF`, Peak RSS `126.0 MiB` / `1.22×`, live heap `51.7 MiB`) and `avif_bbb_4k_t8` time to `87.9 ms` (`0.322×` vs. `libyuv=OFF`, Peak RSS `62.7 MiB` / `1.22×`). When C `libavif` enables hand-written `libyuv` (`1960`, `AUTOMATIC`) SIMD assembly, its `YUVA → RGBA8` stage drops from `603.41 ms` to `59.91 ms` (corpus) and from `235.28 ms` to `17.28 ms` (4K); against that production reference, `100% #![forbid(unsafe_code)]` `av1oxide::avif` measures `1.169×` total on `avif_corpus_156_t4` (`1.164×` with reused `write_rgba8_into`) and `1.652×` total on `avif_bbb_4k_t8` (`YUVA → RGBA8`: `29.19 ms` fresh / `23.30 ms` reused vs. `17.28 ms` `libyuv`).

---

## License

[BSD-2-Clause](LICENSE)
