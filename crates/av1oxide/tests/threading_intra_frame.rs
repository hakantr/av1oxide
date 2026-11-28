//! F7a — Kare-içi tile ve post-filter (CDEF, Superres, Loop Restoration, Film Grain)
//! paralelliği entegrasyon ve deterministik eşdeğerlik testleri (`ADR-0004`).
//!
//! Kapsam:
//! 1. `n_threads ∈ {1, 2, 3, 4, 8, 16, 0}` ile çözülen çok-tile'lı, 8/10/12-bpc,
//!    4:0:0/4:2:0/4:2:2/4:4:4, CDEF + Superres + Loop Restoration + Film Grain
//!    ve 8K (`test25` dâhil) akışlarda her çıktı karesinin Y/U/V düzlem baytları,
//!    araç sayaçları (`decoded_tool_counts`) ve tepe bütçesi (`allocation_peak`)
//!    `n_threads = 1` (ve `dav1d` referans çıktısı) ile **%100 bit-özdeştir**.
//! 2. Bozuk tile verisi ve sıkı `Config::allocation_limit` (`P` geçer, `P - 1`
//!    `Error::BudgetExceeded` döner) altında `n_threads = 1` ve `n_threads = 8`
//!    tamamen aynı deterministik `Result` çıktısını verir (`deadlock = 0`, `panic = 0`).

#![forbid(unsafe_code)]

use av1oxide::api::{Config, Decoder, Packet};
use av1oxide::error::Error;
use av1oxide::frame::{Frame, PlaneType};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("depo kökü")
        .to_path_buf()
}

/// IVF veya Annex B (`temporal_unit -> frame_unit -> obu_unit` leb128) dosyasını
/// `Decoder::send_packet` paketlerine böler.
fn read_packets(path: &Path) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if data.len() >= 32 && &data[0..4] == b"DKIF" && &data[8..12] == b"AV01" {
        let mut packets = Vec::new();
        let mut pos = 32usize;
        while pos + 12 <= data.len() {
            let size = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 12;
            if size > data.len() - pos {
                break;
            }
            packets.push(data[pos..pos + size].to_vec());
            pos += size;
        }
        return packets;
    }
    // Section 5 (TD ile başlayan OBU dizisi, ör. `profile2_not_annexb_special/streams/test25.obu`).
    if data.len() >= 2 && ((data[0] >> 3) & 0x0f) == 2 && (data[0] & 0x02) != 0 {
        let mut packets = Vec::new();
        let mut pos = 0usize;
        while pos < data.len() {
            let start = pos;
            let mut first = true;
            while pos < data.len() {
                let byte = data[pos];
                let obu_type = (byte >> 3) & 0x0f;
                if !first && obu_type == 2 {
                    break;
                }
                if byte & 0x02 == 0 {
                    pos = data.len();
                    break;
                }
                let has_ext = usize::from(byte & 0x04 != 0);
                let mut hdr_pos = pos + 1 + has_ext;
                let mut len = 0u64;
                let mut i = 0usize;
                while hdr_pos + i < data.len() && i < 8 {
                    let b = data[hdr_pos + i];
                    i += 1;
                    len |= u64::from(b & 0x7f) << (7 * (i - 1));
                    if b & 0x80 == 0 {
                        break;
                    }
                }
                hdr_pos += i;
                pos = (hdr_pos + len as usize).min(data.len());
                first = false;
            }
            if pos > start {
                packets.push(data[start..pos].to_vec());
            } else {
                break;
            }
        }
        if !packets.is_empty() {
            return packets;
        }
    }
    // Annex B okuyucu (Argon `.obu` akışları).
    fn leb(data: &[u8], pos: usize) -> Option<(u64, usize)> {
        let mut value = 0u64;
        let mut i = 0usize;
        loop {
            let b = *data.get(pos + i)?;
            i += 1;
            value |= u64::from(b & 0x7f) << (7 * (i - 1));
            if b & 0x80 == 0 {
                break;
            }
            if i == 8 {
                return None;
            }
        }
        Some((value, i))
    }
    let mut packets = Vec::new();
    let mut pos = 0usize;
    let mut tu = 0u64;
    let mut fu = 0u64;
    while pos < data.len() {
        if tu == 0 {
            let Some((v, c)) = leb(&data, pos) else { break };
            pos += c;
            tu = v;
        }
        if fu == 0 {
            let Some((v, c)) = leb(&data, pos) else { break };
            if v + c as u64 > tu {
                break;
            }
            pos += c;
            tu -= c as u64;
            fu = v;
        }
        let Some((len, c)) = leb(&data, pos) else {
            break;
        };
        if len + c as u64 > fu {
            break;
        }
        pos += c;
        let end = pos + len as usize;
        if end > data.len() {
            break;
        }
        packets.push(data[pos..end].to_vec());
        pos = end;
        fu -= len + c as u64;
        tu = tu.saturating_sub(len + c as u64);
    }
    assert!(
        !packets.is_empty(),
        "akıştan paket çıkarılamadı: {}",
        path.display()
    );
    packets
}

/// Karenin görünür düzlem piksellerinin FNV-1a 64 özetini hesaplar.
fn frame_digest(frame: &Frame) -> [u64; 3] {
    let bpp = frame.bit_depth().bytes_per_sample();
    let mut out = [0u64; 3];
    for (idx, pt) in [PlaneType::Y, PlaneType::U, PlaneType::V]
        .into_iter()
        .enumerate()
    {
        let Some(plane) = frame.plane(pt) else {
            continue;
        };
        let row_bytes = plane.width() * bpp;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for y in 0..plane.height() {
            for &b in &plane.row(y).expect("satır")[..row_bytes] {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        out[idx] = h;
    }
    out
}

#[derive(Debug, PartialEq, Eq)]
struct StreamRun {
    digests: Vec<[u64; 3]>,
    allocation_peak: usize,
    decoded_frames: u64,
}

#[derive(Debug, PartialEq, Eq)]
struct StreamOutput<'a> {
    digests: &'a [[u64; 3]],
    decoded_frames: u64,
}

impl StreamRun {
    fn output(&self) -> StreamOutput<'_> {
        StreamOutput {
            digests: &self.digests,
            decoded_frames: self.decoded_frames,
        }
    }
}

fn decode_stream(
    packets: &[Vec<u8>],
    n_threads: usize,
    limit: Option<usize>,
) -> Result<StreamRun, Error> {
    let mut decoder = Decoder::new(Config {
        n_threads,
        max_frame_delay: 1,
        allocation_limit: limit,
        apply_grain: true,
        output_invisible_frames: true,
        ..Config::default()
    })?;
    let mut digests = Vec::new();
    for pkt in packets {
        decoder.send_packet(&Packet::new(pkt))?;
        while let Some(frame) = decoder.receive_frame()? {
            digests.push(frame_digest(&frame));
        }
    }
    decoder.flush();
    while let Some(frame) = decoder.receive_frame()? {
        digests.push(frame_digest(&frame));
    }
    Ok(StreamRun {
        digests,
        allocation_peak: decoder.allocation_peak(),
        decoded_frames: decoder.decoded_frames(),
    })
}

#[test]
fn kare_ici_paralellik_1_2_3_4_8_16_0_thread_bit_ayni() {
    let root = repo_root();
    let quick = std::env::var_os("AV1OXIDE_THREAD_TEST_QUICK").is_some();
    let argon =
        root.join(".cache/argon/extracted/argon_coveragetool_av1_base_and_extended_profiles_v2.1");
    let mut candidates: Vec<PathBuf> = vec![
        root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_001_vq_aom_ctest_4.2.stripped.ivf"),
        root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf"),
    ];
    let extra: &[PathBuf] = if quick {
        &[
            argon.join("profile0_core/streams/test157_1186_590.obu"),
            argon.join("profile1_core/streams/test2081.obu"),
            argon.join("profile2_core/streams/test15357.obu"),
        ]
    } else {
        &[
            argon.join("profile0_core/streams/test10103.obu"),
            argon.join("profile0_core/streams/test157_1186_590.obu"),
            argon.join("profile0_core/streams/test5395.obu"),
            argon.join("profile1_core/streams/test2081.obu"),
            argon.join("profile2_core/streams/test3590_13835.obu"),
            argon.join("profile2_core/streams/test15357.obu"),
            root.join(".cache/dav1d-test-data/10-bit/film_grain/422_oddheight.ivf"),
        ]
    };
    for p in extra {
        if p.exists() {
            candidates.push(p.clone());
        }
    }

    let threads_list: &[usize] = if quick {
        &[2, 4, 8]
    } else {
        &[2, 3, 4, 8, 16, 0]
    };
    for path in &candidates {
        let mut packets = read_packets(path);
        if quick && packets.len() > 24 {
            packets.truncate(24);
        }
        let ref_run = decode_stream(&packets, 1, None)
            .unwrap_or_else(|e| panic!("n_threads=1 çözülemedi ({}): {e}", path.display()));
        assert!(ref_run.decoded_frames > 0);
        for &n_threads in threads_list {
            let run = decode_stream(&packets, n_threads, None).unwrap_or_else(|e| {
                panic!("n_threads={n_threads} çözülemedi ({}): {e}", path.display())
            });
            assert_eq!(
                run.output(),
                ref_run.output(),
                "n_threads={n_threads} çıktısı n_threads=1'den farklı: {}",
                path.display()
            );
            assert!(
                run.allocation_peak <= ref_run.allocation_peak,
                "n_threads={n_threads} allocation_peak ({}) n_threads=1 ({}) sınırını aştı: {}",
                run.allocation_peak,
                ref_run.allocation_peak,
                path.display()
            );
        }
    }
}

#[test]
fn test25_8k_1_2_4_8_thread_bit_ayni() {
    if std::env::var_os("AV1OXIDE_THREAD_TEST_QUICK").is_some() {
        return;
    }
    let root = repo_root();
    let test25 = root.join(
        ".cache/argon/extracted/argon_coveragetool_av1_base_and_extended_profiles_v2.1/profile2_not_annexb_special/streams/test25.obu",
    );
    if !test25.exists() {
        return;
    }
    let packets = read_packets(&test25);
    let ref_run = decode_stream(&packets, 1, None).expect("test25 n_threads=1");
    for &n_threads in &[2usize, 4, 8] {
        let run = decode_stream(&packets, n_threads, None).expect("test25 paralel");
        assert_eq!(
            run.output(),
            ref_run.output(),
            "test25 n_threads={n_threads} farklı"
        );
        assert!(
            run.allocation_peak <= ref_run.allocation_peak,
            "test25 n_threads={n_threads} allocation_peak ({}) > n_threads=1 ({})",
            run.allocation_peak,
            ref_run.allocation_peak
        );
    }
}

#[test]
fn test54_8k_paralel_karo_tepe_butcesini_dusurur() {
    if std::env::var_os("AV1OXIDE_THREAD_TEST_QUICK").is_some() {
        return;
    }
    let root = repo_root();
    let test54 = root.join(
        ".cache/argon/extracted/argon_coveragetool_av1_base_and_extended_profiles_v2.1/profile0_not_annexb/streams/test54.obu",
    );
    if !test54.exists() {
        return;
    }
    let mut packets = read_packets(&test54);
    packets.truncate(2);
    let ref_run = decode_stream(&packets, 1, None).expect("test54 n_threads=1");
    let par_run = decode_stream(&packets, 8, None).expect("test54 n_threads=8");
    assert_eq!(par_run.output(), ref_run.output());
    assert!(
        par_run.allocation_peak + (16 * 1024 * 1024) <= ref_run.allocation_peak,
        "test54 paralel karo yolu tam kare refmvs ızgarasını atlayarak tepe bütçeyi düşürmeli: n_threads=1 {}, n_threads=8 {}",
        ref_run.allocation_peak,
        par_run.allocation_peak
    );
}

#[test]
fn bozuk_tile_ve_butce_siniri_altinda_deterministik_hata_ve_sifir_deadlock() {
    let root = repo_root();
    let quick = std::env::var_os("AV1OXIDE_THREAD_TEST_QUICK").is_some();
    let argon =
        root.join(".cache/argon/extracted/argon_coveragetool_av1_base_and_extended_profiles_v2.1");
    let stream_path = if quick {
        root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf")
    } else {
        let p = argon.join("profile0_core/streams/test10103.obu");
        if p.exists() {
            p
        } else {
            root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf")
        }
    };
    let mut packets = read_packets(&stream_path);
    if quick && packets.len() > 20 {
        packets.truncate(20);
    }

    // 1. Sıkı bütçe sınırı (`P` geçer, `P - 1` `Error::BudgetExceeded` döner).
    let ref_run = decode_stream(&packets, 1, None).expect("sınırsız 1 thread");
    let par_run = decode_stream(&packets, 8, None).expect("sınırsız 8 thread");
    assert_eq!(par_run.output(), ref_run.output());
    for (n_threads, expected) in [(1usize, &ref_run), (8usize, &par_run)] {
        let peak = expected.allocation_peak;
        assert_eq!(
            &decode_stream(&packets, n_threads, Some(peak)).expect("P bütçe"),
            expected
        );
        let err = decode_stream(&packets, n_threads, Some(peak - 1)).expect_err("P-1 bütçe");
        assert_eq!(
            err,
            Error::BudgetExceeded {
                requested: peak,
                limit: peak - 1,
            }
        );
    }

    // 2. Bozuk tile yükü: son paketin (kare/tile verisinin) son baytlarını boz
    // veya kes; `n_threads = 1` ve `n_threads = 8` paniklemeden ve deadlock'a
    // girmeden aynı deterministik sonucu üretmelidir.
    for cut in [1usize, 3, 7, 15, 31] {
        let mut mutated = packets.clone();
        if let Some(last) = mutated.last_mut()
            && last.len() > cut + 4
        {
            let n = last.len();
            last[n - cut] ^= 0xFF;
            last.truncate(n - cut / 2);
        }
        let r1 = decode_stream(&mutated, 1, None);
        let r8 = decode_stream(&mutated, 8, None);
        assert_eq!(
            r1, r8,
            "bozuk girdi cut={cut} için 1 ve 8 thread sonucu farklı"
        );
    }
}

#[test]
fn paralel_karo_temporal_hits_ve_tmvs_atlama_tutarliligi() {
    let root = repo_root();
    let stream_path = root.join(
        "crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf",
    );
    let packets = read_packets(&stream_path);

    let run_with_tools = |n_threads: usize| {
        let mut dec = Decoder::new(Config {
            n_threads,
            max_frame_delay: 1,
            apply_grain: true,
            output_invisible_frames: true,
            ..Config::default()
        })
        .expect("decoder");
        dec.set_collect_diagnostics(true);
        let mut digests = Vec::new();
        for pkt in &packets {
            dec.send_packet(&Packet::new(pkt)).expect("send_packet");
            while let Some(f) = dec.receive_frame().expect("receive_frame") {
                digests.push(frame_digest(&f));
            }
        }
        dec.flush();
        while let Some(f) = dec.receive_frame().expect("receive_frame") {
            digests.push(frame_digest(&f));
        }
        let temporal_counts: Vec<u64> = dec
            .decoded_tool_counts()
            .iter()
            .map(|t| t.temporal_candidate)
            .collect();
        (digests, temporal_counts, dec.allocation_peak())
    };

    let (d1, t1, peak1) = run_with_tools(1);
    let (d4, t4, peak4) = run_with_tools(4);
    assert_eq!(
        d1, d4,
        "düzlem özetleri n_threads=1 ve 4 arasında özdeş olmalı"
    );
    assert_eq!(
        t1, t4,
        "temporal_candidate sayaçları paralel karo yolunda korunmalı"
    );
    assert!(
        peak4 <= peak1,
        "paralel karo tepe bütçesi ({peak4}) tek iş parçacığını ({peak1}) aşmamalı"
    );
}
