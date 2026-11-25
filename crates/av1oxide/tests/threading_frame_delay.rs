//! Faz `F7b` (`P48` — Adım 2 / 3): Kareler arası (`max_frame_delay` / `FramePipeline`)
//! paralellik, gecikmeli Film Grain örtüşmesi ve `receive_frame` / `flush` / `reset` /
//! `drop` / `Clone` yaşam döngüsü bütünleşme testleri (`ADR-0004`, `BLK-01`, `BLK-07`, `BLK-08`).

#![forbid(unsafe_code)]

use av1oxide::api::{Config, Decoder, DecoderState, Error, Packet};
use av1oxide::frame::{Frame, PlaneType};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace kök dizini")
        .to_path_buf()
}

fn read_leb128(slice: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for (i, &byte) in slice.iter().take(8).enumerate() {
        value |= u64::from(byte & 0x7f) << (i * 7);
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// IVF veya Section 5 / Annex-B `.obu` dosyasını paket listesine ayırır.
fn read_packets(path: &Path) -> Vec<Vec<u8>> {
    let data =
        std::fs::read(path).unwrap_or_else(|e| panic!("okunamadı ({}): {e}", path.display()));
    if data.len() >= 32 && &data[0..4] == b"DKIF" {
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

    if data.len() >= 2 && ((data[0] >> 3) & 0x0f) == 2 && (data[0] & 0x02) != 0 {
        let mut packets = Vec::new();
        let mut cur = Vec::new();
        let mut pos = 0usize;
        let mut ok = true;
        while pos < data.len() {
            let hdr = data[pos];
            let obu_type = (hdr >> 3) & 0x0f;
            let has_ext = (hdr & 0x04) != 0;
            let has_size = (hdr & 0x02) != 0;
            if !has_size {
                ok = false;
                break;
            }
            let hdr_len = 1 + usize::from(has_ext);
            if pos + hdr_len > data.len() {
                ok = false;
                break;
            }
            let Some((sz, leb_len)) = read_leb128(&data[pos + hdr_len..]) else {
                ok = false;
                break;
            };
            let total = hdr_len + leb_len + (sz as usize);
            if pos + total > data.len() {
                ok = false;
                break;
            }
            if obu_type == 2 && !cur.is_empty() {
                packets.push(core::mem::take(&mut cur));
            }
            cur.extend_from_slice(&data[pos..pos + total]);
            pos += total;
        }
        if ok && !cur.is_empty() {
            packets.push(cur);
            return packets;
        }
    }

    // Annex-B `.obu`
    let mut packets = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let Some((tu_size, n1)) = read_leb128(&data[pos..]) else {
            break;
        };
        pos += n1;
        let tu_end = (pos + tu_size as usize).min(data.len());
        let mut pkt = Vec::new();
        while pos < tu_end {
            let Some((frame_size, n2)) = read_leb128(&data[pos..tu_end]) else {
                break;
            };
            pos += n2;
            let frame_end = (pos + frame_size as usize).min(tu_end);
            while pos < frame_end {
                let Some((obu_len, n3)) = read_leb128(&data[pos..frame_end]) else {
                    break;
                };
                pos += n3;
                let obu_len = obu_len as usize;
                if obu_len == 0 || pos + obu_len > frame_end {
                    break;
                }
                let hdr = data[pos];
                let has_ext = (hdr & 0x04) != 0;
                let has_size = (hdr & 0x02) != 0;
                if has_size {
                    pkt.extend_from_slice(&data[pos..pos + obu_len]);
                } else {
                    let hdr_len = (1 + usize::from(has_ext)).min(obu_len);
                    let payload_len = obu_len - hdr_len;
                    pkt.push(hdr | 0x02);
                    if has_ext && hdr_len == 2 {
                        pkt.push(data[pos + 1]);
                    }
                    let mut rem = payload_len;
                    loop {
                        let mut b = (rem & 0x7f) as u8;
                        rem >>= 7;
                        if rem != 0 {
                            b |= 0x80;
                        }
                        pkt.push(b);
                        if rem == 0 {
                            break;
                        }
                    }
                    pkt.extend_from_slice(&data[pos + hdr_len..pos + obu_len]);
                }
                pos += obu_len;
            }
        }
        if !pkt.is_empty() {
            packets.push(pkt);
        }
        pos = tu_end;
    }
    packets
}

/// FNV-1a 64-bit düzlem özeti (`[Y, U, V]`).
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
            let row = plane.row(y).expect("satır");
            for &b in &row[..row_bytes] {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
        out[idx] = h;
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeedMode {
    Each,
    SendAll,
}

#[derive(Debug, PartialEq, Eq)]
struct StreamOutput {
    digests: Vec<[u64; 3]>,
    decoded_frames: u64,
}

fn run_stream(
    packets: &[Vec<u8>],
    n_threads: usize,
    max_frame_delay: usize,
    mode: FeedMode,
    output_invisible: bool,
    limit: Option<usize>,
) -> Result<StreamOutput, Error> {
    let mut decoder = Decoder::new(Config {
        n_threads,
        max_frame_delay,
        allocation_limit: limit,
        apply_grain: true,
        output_invisible_frames: output_invisible,
        ..Config::default()
    })?;
    let mut digests = Vec::new();
    match mode {
        FeedMode::Each => {
            for pkt in packets {
                decoder.send_packet(&Packet::new(pkt))?;
                while let Some(frame) = decoder.receive_frame()? {
                    digests.push(frame_digest(&frame));
                }
            }
        }
        FeedMode::SendAll => {
            for pkt in packets {
                decoder.send_packet(&Packet::new(pkt))?;
            }
        }
    }
    decoder.flush();
    assert_eq!(decoder.state(), DecoderState::Draining);
    while let Some(frame) = decoder.receive_frame()? {
        digests.push(frame_digest(&frame));
    }
    Ok(StreamOutput {
        digests,
        decoded_frames: decoder.decoded_frames(),
    })
}

#[test]
fn kareler_arasi_paralellik_each_ve_sendall_tum_matris_bit_ayni() {
    let root = repo_root();
    let quick = std::env::var_os("AV1OXIDE_THREAD_TEST_QUICK").is_some();
    let argon =
        root.join(".cache/argon/extracted/argon_coveragetool_av1_base_and_extended_profiles_v2.1");

    let mut candidates: Vec<PathBuf> = if quick {
        vec![
            root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_001_vq_aom_ctest_4.2.stripped.ivf"),
            root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf"),
            root.join(".cache/dav1d-test-data/10-bit/film_grain/av1-1-b10-23-film_grain-50.ivf"),
        ]
    } else {
        vec![
            root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_001_vq_aom_ctest_4.2.stripped.ivf"),
            root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf"),
            root.join(".cache/dav1d-test-data/8-bit/features/ccvb_film_grain.ivf"),
            root.join(".cache/dav1d-test-data/10-bit/film_grain/av1-1-b10-23-film_grain-50.ivf"),
        ]
    };
    let rels: &[&str] = if quick {
        &["profile0_core/streams/test157_1186_590.obu"]
    } else {
        &[
            "profile0_core/streams/test10103.obu",
            "profile0_core/streams/test157_1186_590.obu",
            "profile1_core/streams/test2081.obu",
            "profile1_core/streams/test4335.obu",
            "profile2_core/streams/test15357.obu",
        ]
    };
    for rel in rels {
        let p = argon.join(rel);
        if p.exists() {
            candidates.push(p);
        }
    }

    let mut tested = 0usize;
    let mut all_packet_sets: Vec<(String, Vec<Vec<u8>>)> = Vec::new();
    for path in &candidates {
        if !path.exists() {
            continue;
        }
        let mut packets = read_packets(path);
        if quick && packets.len() > 8 {
            packets.truncate(8);
        }
        if !packets.is_empty() {
            all_packet_sets.push((path.display().to_string(), packets));
        }
    }
    // Çoklu sequence header geçişini sınamak için iki farklı çözünürlüklü akışı uç uca ekle.
    if all_packet_sets.len() >= 3 {
        let mut combined = all_packet_sets[0].1.clone();
        combined.extend(all_packet_sets[2].1.iter().cloned());
        all_packet_sets.push(("birlesik_coklu_seqhdr".to_owned(), combined));
    }

    let matrix: &[(usize, usize)] = if quick {
        &[(2, 2), (4, 1), (4, 2), (8, 0)]
    } else {
        &[
            (1, 0),
            (1, 1),
            (2, 1),
            (2, 2),
            (2, 0),
            (4, 1),
            (4, 2),
            (4, 4),
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 8),
            (8, 0),
            (16, 4),
            (0, 0),
        ]
    };
    let inv_flags: &[bool] = if quick { &[true] } else { &[false, true] };

    for (name, packets) in &all_packet_sets {
        for &output_invisible in inv_flags {
            let ref_out = run_stream(packets, 1, 1, FeedMode::Each, output_invisible, None)
                .unwrap_or_else(|e| panic!("referans çözülemedi ({name}): {e}"));
            assert!(ref_out.decoded_frames > 0);
            for &(n_threads, max_frame_delay) in matrix {
                for &mode in &[FeedMode::Each, FeedMode::SendAll] {
                    let out = run_stream(
                        packets,
                        n_threads,
                        max_frame_delay,
                        mode,
                        output_invisible,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "çözülemedi ({name}, t={n_threads}, f={max_frame_delay}, {mode:?}): {e}"
                        )
                    });
                    assert_eq!(
                        out, ref_out,
                        "fark bulundu: {name}, t={n_threads}, f={max_frame_delay}, {mode:?}, inv={output_invisible}"
                    );
                }
            }
        }
        tested += 1;
    }
    assert!(tested >= 3, "en az 3 akış test edilmeli");
}

#[test]
fn yasam_dongusu_reset_drop_klon_ve_bozuk_paket_kilitlenmez() {
    let root = repo_root();
    let p1 = root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_001_vq_aom_ctest_4.2.stripped.ivf");
    let p2 = root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf");
    let pkts1 = read_packets(&p1);
    let pkts2 = read_packets(&p2);
    assert!(!pkts1.is_empty() && !pkts2.is_empty());

    // 1. `max_frame_delay = 1` iken her paketten sonra çıkan kare dizisi
    //    `n_threads ∈ {1, 2, 4, 8}` için paket-etiketi bazında birebir aynı olmalıdır.
    let per_packet_trace = |n_threads: usize| -> Vec<(usize, [u64; 3])> {
        let mut dec = Decoder::new(Config {
            n_threads,
            max_frame_delay: 1,
            ..Config::default()
        })
        .expect("decoder");
        let mut trace = Vec::new();
        for (p_idx, pkt) in pkts1.iter().enumerate() {
            dec.send_packet(&Packet::new(pkt)).expect("send_packet");
            while let Some(f) = dec.receive_frame().expect("receive_frame") {
                trace.push((p_idx, frame_digest(&f)));
            }
        }
        trace
    };
    let trace_1 = per_packet_trace(1);
    for nth in [2usize, 4, 8] {
        assert_eq!(per_packet_trace(nth), trace_1);
    }

    // 2. `max_frame_delay = 0` (`n_fc >= 2`) iken akış ortasında `reset()`
    //    (hiç `flush()` veya `receive_frame()` çağırmadan!) ve ardından 2. akışın çözümü.
    let ref_p2 = run_stream(&pkts2, 1, 1, FeedMode::Each, false, None).expect("ref p2");
    for nth in [2usize, 4, 8] {
        let mut dec = Decoder::new(Config {
            n_threads: nth,
            max_frame_delay: 0,
            ..Config::default()
        })
        .expect("decoder");
        for pkt in pkts1.iter().take(pkts1.len() / 2 + 1) {
            dec.send_packet(&Packet::new(pkt)).expect("send_packet p1");
        }
        // Uçuşta kareler varken doğrudan reset()!
        dec.reset();
        assert_eq!(dec.state(), DecoderState::Ready);
        assert_eq!(dec.decoded_frames(), 0);

        let mut digests = Vec::new();
        for pkt in &pkts2 {
            dec.send_packet(&Packet::new(pkt)).expect("send_packet p2");
            while let Some(f) = dec.receive_frame().expect("receive_frame p2") {
                digests.push(frame_digest(&f));
            }
        }
        dec.flush();
        while let Some(f) = dec.receive_frame().expect("drain p2") {
            digests.push(frame_digest(&f));
        }
        assert_eq!(digests, ref_p2.digests);
    }

    // 3. `max_frame_delay = 0` (`n_fc >= 2`) iken uçuştaki kareler çalışırken
    //    doğrudan `drop(decoder)` (kilitlenme veya panik olmamalı).
    for _ in 0..8 {
        let mut dec = Decoder::new(Config {
            n_threads: 8,
            max_frame_delay: 0,
            ..Config::default()
        })
        .expect("decoder");
        for pkt in pkts1.iter().take(4) {
            dec.send_packet(&Packet::new(pkt)).expect("send");
        }
        drop(dec);
    }

    // 4. `max_frame_delay = 0` iken akış ortasında `decoder.clone()`: hem orijinal
    //    hem klon kalan paketleri bağımsız çözüp referansla bit-aynı çıktıyı üretmelidir.
    let ref_p1 = run_stream(&pkts1, 1, 1, FeedMode::SendAll, false, None).expect("ref p1");
    let split_at = pkts1.len() / 2;
    let mut dec_orig = Decoder::new(Config {
        n_threads: 4,
        max_frame_delay: 0,
        ..Config::default()
    })
    .expect("decoder");
    for pkt in &pkts1[..split_at] {
        dec_orig
            .send_packet(&Packet::new(pkt))
            .expect("send prefix");
    }
    let mut dec_cloned = dec_orig.clone();
    for dec in [&mut dec_orig, &mut dec_cloned] {
        for pkt in &pkts1[split_at..] {
            dec.send_packet(&Packet::new(pkt)).expect("send suffix");
        }
        dec.flush();
        let mut digests = Vec::new();
        while let Some(f) = dec.receive_frame().expect("drain") {
            digests.push(frame_digest(&f));
        }
        assert_eq!(digests, ref_p1.digests);
    }

    // 5. Sıkı `allocation_limit` ve bozuk paketler altında `max_frame_delay = 0, n_threads = 8`:
    //    kilitlenme yok, deterministik `Err` ve `reset()` sonrası temiz kurtarma.
    for lim in [1024usize, 64 * 1024] {
        let res = run_stream(&pkts1, 8, 0, FeedMode::Each, true, Some(lim));
        assert!(
            matches!(res, Err(Error::BudgetExceeded { .. })),
            "bütçe aşımı bekleniyordu (lim={lim}): {res:?}"
        );
    }
    for cut in [2usize, 7, 15] {
        let mut corrupt = pkts1.clone();
        let last = corrupt.last_mut().expect("paket");
        if last.len() > cut {
            last.truncate(cut);
        }
        let r1 = run_stream(&corrupt, 1, 1, FeedMode::Each, true, None);
        let r8 = run_stream(&corrupt, 8, 0, FeedMode::Each, true, None);
        assert_eq!(r1.is_err(), r8.is_err());
    }
}

#[test]
fn f85_decode_buf_pool_cok_thread_geri_donusum_flush_reset_ve_klon() {
    let root = repo_root();
    let p1 = root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_001_vq_aom_ctest_4.2.stripped.ivf");
    let p2 = root.join("crates/av1oxide/tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf");
    let pkts1 = read_packets(&p1);
    let pkts2 = read_packets(&p2);
    assert!(!pkts1.is_empty() && !pkts2.is_empty());

    let ref1 = run_stream(&pkts1, 1, 1, FeedMode::Each, true, None).expect("ref1");
    let ref2 = run_stream(&pkts2, 1, 1, FeedMode::Each, true, None).expect("ref2");

    let mut dec = Decoder::new(Config {
        n_threads: 4,
        max_frame_delay: 0,
        output_invisible_frames: true,
        allocation_limit: None,
        ..Config::default()
    })
    .expect("decoder");

    // Aynı çözücü ve aynı DecodeBufPool üzerinde ardışık flush + reset döngüleriyle
    // tamponların geri dönüştürülmesini ve eşzamanlı arka plan iş parçacığında
    // Frame düşürülmesini (Arc::try_unwrap -> havuza iade) doğrula.
    for tur in 0..4usize {
        let pkts = if tur % 2 == 0 { &pkts1 } else { &pkts2 };
        let beklenen = if tur % 2 == 0 {
            &ref1.digests
        } else {
            &ref2.digests
        };
        let mut digests = Vec::new();
        let mut tutulan = Vec::new();
        let mut bg_threads = Vec::new();
        for pkt in pkts {
            dec.send_packet(&Packet::new(pkt)).expect("send_packet");
            while let Some(f) = dec.receive_frame().expect("receive_frame") {
                digests.push(frame_digest(&f));
                tutulan.push(f);
                if tutulan.len() >= 3 {
                    let parti = std::mem::take(&mut tutulan);
                    bg_threads.push(std::thread::spawn(move || {
                        for fr in parti {
                            assert!(fr.width() > 0);
                            drop(fr);
                        }
                    }));
                }
            }
        }
        bg_threads.push(std::thread::spawn(move || {
            for f in tutulan {
                assert!(f.width() > 0);
                drop(f);
            }
        }));
        dec.flush();
        while let Some(f) = dec.receive_frame().expect("drain") {
            digests.push(frame_digest(&f));
        }
        for bg in bg_threads {
            bg.join().expect("bg thread");
        }
        assert_eq!(&digests, beklenen, "tur={tur} DecodeBufPool çıktı farkı");
        dec.reset();
    }
}
