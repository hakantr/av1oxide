# av1oxide (`v0.1.0`)

**English** | [Türkçe](README.tr.md)

`av1oxide` is a **100% stable-Rust** (`rustc 1.100.0`, Edition 2024) reimplementation of the VideoLAN [`dav1d`](https://code.videolan.org/videolan/dav1d) (`1.5.4`) AV1 video decoder with **zero runtime external dependencies**.

- **Profiles & Bit Depths:** Main (`0`), High (`1`), Professional (`2`); `8-bit`, `10-bit`, `12-bit`; `4:2:0`, `4:2:2`, `4:4:4`, and `Monochrome` (`4:0:0`).
- **Complete AV1 Pipeline:** OBU parsing, Sequence/Frame Headers, MSAC arithmetic entropy decoder, inverse transforms (`ITX`), intra prediction (`IPRED`, CfL, Palette), motion compensation (`MC`, Warp, Superres/Resize, IntraBC, `REFMVS`), in-loop filters (`LF`, `CDEF`, `LR` Wiener/SGR), and Film Grain (`FG`) synthesis.
- **Bit-Exact Conformance:** Verified 100% MD5 bit-exact parity against `dav1d 1.5.4` across **`2,781 / 2,781` streams (`37,664 / 37,664` frames)** — covering `dav1d-test-data` (`19/19` streams, `964` frames) and Argon Streams `v2.1.1` (`2,762/2,762` streams, `36,700` frames) across all CPU capability masks and multithreaded configurations.
- **Capability-Token SIMD (`767` DSP Slots):**
  - `x86_64` (`SSE2`, `SSSE3`, `SSE4.1`, `AVX2`, `AVX-512ICL`): `756 / 767` SIMD slots
  - `aarch64` (`NEON`, `DOTPROD`): `701 / 767` SIMD slots
  - `wasm32` (`+simd128`): `31 / 767` SIMD slots under `100% #![forbid(unsafe_code)]`
  - Secondary architectures (`arm32`, `riscv64`, `loongarch64`, `powerpc64le`): `100%` safe bit-exact scalar fallback
- **Multithreading & Memory Budgets:** Intra-frame (`Tile`, superblock row, in-loop filter) and inter-frame (`FrameSlot`) parallelism (`Config::n_threads`, `Config::max_frame_delay`); pre-allocation pixel and memory limits (`Config::frame_size_limit`, `Config::allocation_limit`).

---

## Installation

Add `av1oxide` to your `Cargo.toml`:

```toml
[dependencies]
av1oxide = "0.1"
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

### 3. Runnable Examples (`examples/`)

Three standalone examples are included in the crate:

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

---

## Performance Comparison vs. `dav1d 1.5.4`

All benchmarks were measured under `100%` MD5 bit-exact output parity between `av1oxide v0.1.0` (`rustc 1.100.0 --release`) and the reference C/ASM `dav1d 1.5.4` (`99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`).

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

## License

[BSD-2-Clause](LICENSE)
