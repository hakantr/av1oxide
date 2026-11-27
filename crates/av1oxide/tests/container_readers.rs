//! `av1oxide::container::{ivf, annexb}` bütünleştirme ve gerçek akış MD5 doğrulama testleri.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use av1oxide::container::annexb::{AnnexBDemuxer, read_leb128};
use av1oxide::container::ivf::IvfParser;
#[cfg(feature = "std")]
use av1oxide::container::ivf::IvfReader;
use av1oxide::{Config, Decoder, Error, Frame, Packet, PlaneType};

/// RFC 1321 MD5 özeti (haricî bağımlılık gerektirmez).
struct Md5 {
    state: [u32; 4],
    count: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Md5 {
    fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            count: 0,
            buf: [0u8; 64],
            buf_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.count += data.len() as u64;
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finish_hex(mut self) -> String {
        let bit_len = self.count.wrapping_mul(8);
        self.update(&[0x80]);
        let zeros = [0u8; 64];
        if self.buf_len > 56 {
            let pad = 64 - self.buf_len;
            self.update(&zeros[..pad]);
        }
        let pad = 56 - self.buf_len;
        self.update(&zeros[..pad]);
        self.update(&bit_len.to_le_bytes());

        let mut out = String::with_capacity(32);
        for word in self.state {
            for b in word.to_le_bytes() {
                use core::fmt::Write as _;
                let _ = write!(&mut out, "{b:02x}");
            }
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
            5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        const K: [u32; 64] = [
            0xd76a_a478,
            0xe8c7_b756,
            0x2420_70db,
            0xc1bd_ceee,
            0xf57c_0faf,
            0x4787_c62a,
            0xa830_4613,
            0xfd46_9501,
            0x6980_98d8,
            0x8b44_f7af,
            0xffff_5bb1,
            0x895c_d7be,
            0x6b90_1122,
            0xfd98_7193,
            0xa679_438e,
            0x49b4_0821,
            0xf61e_2562,
            0xc040_b340,
            0x265e_5a51,
            0xe9b6_c7aa,
            0xd62f_105d,
            0x0244_1453,
            0xd8a1_e681,
            0xe7d3_fbc8,
            0x21e1_cde6,
            0xc337_07d6,
            0xf4d5_0d87,
            0x455a_14ed,
            0xa9e3_e905,
            0xfcef_a3f8,
            0x676f_02d9,
            0x8d2a_4c8a,
            0xfffa_3942,
            0x8771_f681,
            0x6d9d_6122,
            0xfde5_380c,
            0xa4be_ea44,
            0x4bde_cfa9,
            0xf6bb_4b60,
            0xbebf_bc70,
            0x289b_7ec6,
            0xeaa1_27fa,
            0xd4ef_3085,
            0x0488_1d05,
            0xd9d4_d039,
            0xe6db_99e5,
            0x1fa2_7cf8,
            0xc4ac_5665,
            0xf429_2244,
            0x432a_ff97,
            0xab94_23a7,
            0xfc93_a039,
            0x655b_59c3,
            0x8f0c_cc92,
            0xffef_f47d,
            0x8584_5dd1,
            0x6fa8_7e4f,
            0xfe2c_e6e0,
            0xa301_4314,
            0x4e08_11a1,
            0xf753_7e82,
            0xbd3a_f235,
            0x2ad7_d2bb,
            0xeb86_d391,
        ];
        let mut m = [0u32; 16];
        for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
            m[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | ((!b) & d), i),
                16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = temp;
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }
}

fn hash_frame(md5: &mut Md5, frame: &Frame) {
    for pt in [PlaneType::Y, PlaneType::U, PlaneType::V] {
        let Some(plane) = frame.plane(pt) else {
            continue;
        };
        for y in 0..plane.height() {
            if let Some(row) = plane.row(y) {
                md5.update(row);
            }
        }
    }
}

fn decode_packets_md5<'a>(packets: impl IntoIterator<Item = (i64, &'a [u8])>) -> (usize, String) {
    let mut decoder = Decoder::new(Config {
        n_threads: 1,
        max_frame_delay: 1,
        ..Config::default()
    })
    .expect("çözücü kurulumu");
    let mut md5 = Md5::new();
    let mut count = 0usize;
    for (pts, data) in packets {
        decoder
            .send_packet(&Packet::with_timestamp(data, pts))
            .expect("paket gönderimi");
        while let Some(frame) = decoder.receive_frame().expect("kare alımı") {
            hash_frame(&mut md5, &frame);
            count += 1;
        }
    }
    decoder.flush();
    while let Some(frame) = decoder.receive_frame().expect("boşaltma") {
        hash_frame(&mut md5, &frame);
        count += 1;
    }
    (count, md5.finish_hex())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("depo kökü")
        .to_path_buf()
}

fn write_leb128(out: &mut Vec<u8>, mut val: u32) {
    loop {
        let b = (val & 0x7f) as u8;
        val >>= 7;
        if val != 0 {
            out.push(b | 0x80);
        } else {
            out.push(b);
            break;
        }
    }
}

/// Standart Section 5 OBU paketini Annex B `temporal_unit` biçimine (`obu_has_size_field = 0`
/// dönüştürmesiyle) paketler.
fn pack_section5_as_annexb(section5: &[u8]) -> Vec<u8> {
    let mut fu = Vec::new();
    let mut pos = 0usize;
    while pos < section5.len() {
        let hdr = section5[pos];
        let has_ext = usize::from((hdr & 0x04) != 0);
        let has_size = (hdr & 0x02) != 0;
        assert!(has_size, "girdi Section 5 OBU boyut alanı taşımalı");
        let hdr_len = 1 + has_ext;
        let (payload_len, leb_len) = read_leb128(&section5[pos + hdr_len..]).expect("leb128");
        let payload_start = pos + hdr_len + leb_len;
        let payload_end = payload_start + payload_len as usize;
        let payload = &section5[payload_start..payload_end];

        // Annex B'de `obu_has_size_field = 0` yaz.
        let obu_len = hdr_len + payload.len();
        write_leb128(&mut fu, obu_len as u32);
        fu.push(hdr & !0x02);
        if has_ext == 1 {
            fu.push(section5[pos + 1]);
        }
        fu.extend_from_slice(payload);
        pos = payload_end;
    }
    let mut tu = Vec::new();
    write_leb128(&mut tu, fu.len() as u32);
    tu.extend_from_slice(&fu);

    let mut out = Vec::new();
    write_leb128(&mut out, tu.len() as u32);
    out.extend_from_slice(&tu);
    out
}

#[test]
fn ivf_parser_ve_reader_gercek_fixture_akislarinda_bit_birebir_md5_verir() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/streams");
    let streams = [
        "Syntax_AV1_mainb8ss420_432x240_001_vq_aom_ctest_4.2.stripped.ivf",
        "Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf",
    ];
    for name in streams {
        let path = fixture_dir.join(name);
        let bytes = std::fs::read(&path).expect("fixture IVF okunmalı");

        // 1. IvfParser (sıfır kopyalı dilim)
        let parser = IvfParser::new(&bytes).expect("IVF başlığı geçerli");
        assert_eq!(parser.header().fourcc, *b"AV01");
        assert_eq!(parser.header().width, 432);
        assert_eq!(parser.header().height, 240);
        let parsed: Vec<_> = parser.map(|r| r.expect("geçerli IVF paketi")).collect();
        assert!(!parsed.is_empty());

        let (count_parser, md5_parser) = decode_packets_md5(parsed.iter().map(|p| (p.pts, p.data)));
        assert!(count_parser > 0);

        // 2. IvfReader (akış tabanlı yeniden kullanılan tampon; `std` özelliği gerektirir)
        #[cfg(feature = "std")]
        {
            let mut reader = IvfReader::new(bytes.as_slice()).expect("IvfReader başlığı");
            assert_eq!(reader.header(), IvfParser::new(&bytes).unwrap().header());
            let mut decoder = Decoder::new(Config {
                n_threads: 1,
                max_frame_delay: 1,
                ..Config::default()
            })
            .expect("çözücü");
            let mut buf = Vec::with_capacity(65_536);
            let mut md5_reader = Md5::new();
            let mut count_reader = 0usize;
            let mut pkt_idx = 0usize;
            while let Some(pts) = reader.read_packet_into(&mut buf).expect("paket okunmalı") {
                assert_eq!(pts, parsed[pkt_idx].pts);
                assert_eq!(buf.as_slice(), parsed[pkt_idx].data);
                pkt_idx += 1;
                decoder
                    .send_packet(&Packet::with_timestamp(&buf, pts))
                    .expect("paket gönderimi");
                while let Some(frame) = decoder.receive_frame().expect("kare") {
                    hash_frame(&mut md5_reader, &frame);
                    count_reader += 1;
                }
            }
            decoder.flush();
            while let Some(frame) = decoder.receive_frame().expect("boşaltma") {
                hash_frame(&mut md5_reader, &frame);
                count_reader += 1;
            }
            assert_eq!(pkt_idx, parsed.len());
            assert_eq!(count_reader, count_parser);
            assert_eq!(md5_reader.finish_hex(), md5_parser);
        }

        // 3. Aynı paketleri Annex B'ye sarıp AnnexBDemuxer ile geri açarak MD5 eşitliği doğrula.
        let mut annexb_stream = Vec::new();
        for pkt in &parsed {
            annexb_stream.extend_from_slice(&pack_section5_as_annexb(pkt.data));
        }
        let demuxed = AnnexBDemuxer::demux_all(&annexb_stream).expect("Annex B ayrıştırma");
        assert_eq!(demuxed.len(), parsed.len());
        let (count_annexb, md5_annexb) = decode_packets_md5(
            demuxed
                .iter()
                .enumerate()
                .map(|(i, u)| (i as i64, u.as_slice())),
        );
        assert_eq!(count_annexb, count_parser);
        assert_eq!(md5_annexb, md5_parser);
    }
}

#[test]
fn annexb_demuxer_gercek_argon_akislari_referans_md5_ile_birebir_eslesir() {
    let argon_root = workspace_root()
        .join(".cache/argon/extracted/argon_coveragetool_av1_base_and_extended_profiles_v2.1");
    if !argon_root.exists() {
        return;
    }

    let cases = [
        ("profile0_core", "test10001"),
        ("profile0_core", "test10041"),
        ("profile0_core", "test10046"),
        ("profile0_core", "test10051"),
        ("profile1_core", "test10002"),
        ("profile2_core", "test10003"),
    ];

    let mut verified = 0usize;
    for (suite, stem) in cases {
        let obu_path = argon_root
            .join(suite)
            .join("streams")
            .join(format!("{stem}.obu"));
        let md5_path = argon_root
            .join(suite)
            .join("md5_ref")
            .join(format!("{stem}.md5"));
        if !obu_path.exists() || !md5_path.exists() {
            continue;
        }
        let raw = std::fs::read(&obu_path).expect("Argon .obu okunmalı");
        let expected_md5 = std::fs::read_to_string(&md5_path)
            .expect("Argon .md5 okunmalı")
            .split_whitespace()
            .next()
            .expect("MD5 dizesi")
            .to_ascii_lowercase();

        let units = AnnexBDemuxer::demux_all(&raw).expect("Argon Annex B ayrıştırılmalı");
        assert!(
            !units.is_empty(),
            "{stem}: en az bir temporal_unit beklenir"
        );

        let (count, actual_md5) = decode_packets_md5(
            units
                .iter()
                .enumerate()
                .map(|(i, u)| (i as i64, u.as_slice())),
        );
        assert!(count > 0, "{stem}: en az bir görünür kare çözülmeli");
        assert_eq!(
            actual_md5, expected_md5,
            "Argon Annex B MD5 uyuşmazlığı: {suite}/{stem}"
        );
        verified += 1;
    }
    assert!(verified >= 4, "en az 4 Argon Annex-B akışı doğrulanmalı");
}

#[test]
fn container_bozuk_ve_sinir_disi_girdilerde_temiz_hata_dondurur() {
    assert!(matches!(IvfParser::new(&[]), Err(Error::InvalidData(_))));
    #[cfg(feature = "std")]
    assert!(matches!(
        IvfReader::new(&[][..]),
        Err(Error::InvalidData(_))
    ));
    assert!(matches!(
        AnnexBDemuxer::demux_all(&[0x00]),
        Err(Error::InvalidData(_))
    ));
    assert!(matches!(
        AnnexBDemuxer::demux_all(&[0x80; 9]),
        Err(Error::InvalidData(_))
    ));
    // Bildirilen temporal_unit boyutu girdi uzunluğunu aşıyor.
    assert!(matches!(
        AnnexBDemuxer::demux_all(&[0x7f, 0x01, 0x01, 0x12]),
        Err(Error::InvalidData(_))
    ));
}

#[cfg(feature = "std")]
#[test]
fn ivf_reader_budget_exceeded_preserves_packet_alignment() {
    let pkt1: Vec<u8> = (0..15_106usize).map(|i| (i & 0xFF) as u8).collect();
    let pkt2: Vec<u8> = (0..7_581usize)
        .map(|i| ((i * 7 + 13) & 0xFF) as u8)
        .collect();
    let pkt3: Vec<u8> = (0..17_174usize)
        .map(|i| ((i * 3 + 5) & 0xFF) as u8)
        .collect();

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"AV01");
    ivf.extend_from_slice(&320u16.to_le_bytes());
    ivf.extend_from_slice(&240u16.to_le_bytes());
    ivf.extend_from_slice(&30u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&3u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (pts, payload) in [(0i64, &pkt1), (33i64, &pkt2), (66i64, &pkt3)] {
        ivf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&pts.to_le_bytes());
        ivf.extend_from_slice(payload);
    }

    let mut reader = IvfReader::with_max_frame_bytes(ivf.as_slice(), 7_581).expect("başlık");
    let mut buf = Vec::new();

    // 1. Paket (15 106 B > 7 581 B): BudgetExceeded dönmeli ve yükü akıştan tüketmeli.
    assert_eq!(
        reader.read_packet_into(&mut buf),
        Err(Error::BudgetExceeded {
            requested: 15_106,
            limit: 7_581,
        })
    );
    assert!(buf.is_empty());

    // 2. Paket (7 581 B == limit): paket sınırı bozulmadan tam okunmalı.
    assert_eq!(reader.read_packet_into(&mut buf), Ok(Some(33)));
    assert_eq!(buf, pkt2);

    // 3. Paket (17 174 B > 7 581 B): BudgetExceeded dönmeli ve yükü tüketmeli.
    assert_eq!(
        reader.read_packet_into(&mut buf),
        Err(Error::BudgetExceeded {
            requested: 17_174,
            limit: 7_581,
        })
    );
    assert!(buf.is_empty());

    // 4. Temiz EOF.
    assert_eq!(reader.read_packet_into(&mut buf), Ok(None));

    // 5. `6.I.2` (`P3-1`): Atlama sırasında (`4 096` B sonra) ve başlık ortasında geçici `WouldBlock`
    //    hatası geldiğinde kalan `pending_skip` / `hdr_filled` korunmalı ve sonraki çağrı ikinci
    //    paketi (`pts = 33`, `7 581 B`) hiza kaybı olmadan döndürmelidir.
    struct FlakyIvfStream<'a> {
        data: &'a [u8],
        pos: usize,
        fail_offsets: Vec<usize>,
        chunk_cap: usize,
    }

    impl std::io::Read for FlakyIvfStream<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            if let Some(idx) = self.fail_offsets.iter().position(|&off| off == self.pos) {
                self.fail_offsets.remove(idx);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "yapay geçici G/Ç kesintisi",
                ));
            }
            let next_fail = self
                .fail_offsets
                .iter()
                .copied()
                .filter(|&off| off > self.pos)
                .min()
                .unwrap_or(self.data.len());
            let n = out
                .len()
                .min(self.data.len() - self.pos)
                .min(self.chunk_cap)
                .min(next_fail - self.pos);
            out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    // 5a. İlk paket (15 106 B) atlanırken tam 4 096 bayt sonra (ofset 32 + 12 + 4096 = 4140)
    //     tekil `WouldBlock`: ilk çağrı `BudgetExceeded` döner, ikinci çağrı kalan 11 010 baytı
    //     tüketip 2. paketi (`pts = 33`, `7 581 B`) eksiksiz okur.
    let stream_single = FlakyIvfStream {
        data: &ivf,
        pos: 0,
        fail_offsets: vec![32 + 12 + 4096],
        chunk_cap: 4096,
    };
    let mut flaky_single =
        IvfReader::with_max_frame_bytes(stream_single, 7_581).expect("IVF başlığı");
    assert_eq!(
        flaky_single.read_packet_into(&mut buf),
        Err(Error::BudgetExceeded {
            requested: 15_106,
            limit: 7_581,
        })
    );
    assert_eq!(flaky_single.read_packet_into(&mut buf), Ok(Some(33)));
    assert_eq!(buf, pkt2);
    assert_eq!(
        flaky_single.read_packet_into(&mut buf),
        Err(Error::BudgetExceeded {
            requested: 17_174,
            limit: 7_581,
        })
    );
    assert_eq!(flaky_single.read_packet_into(&mut buf), Ok(None));

    // 5b. Ardışık çoklu `WouldBlock` (hem ilk atlama sırasında 4096. baytta, hem kalan `pending_skip`
    //     boşaltılırken +2048. baytta, hem de 2. paketin 12-bayt başlığının 5. baytında):
    let pkt2_hdr_off = 32 + 12 + 15_106;
    let stream_multi = FlakyIvfStream {
        data: &ivf,
        pos: 0,
        fail_offsets: vec![32 + 12 + 4096, 32 + 12 + 4096 + 2048, pkt2_hdr_off + 5],
        chunk_cap: 4096,
    };
    let mut flaky_multi =
        IvfReader::with_max_frame_bytes(stream_multi, 7_581).expect("IVF başlığı");
    // Çağrı 1: 4096 B atladıktan sonra WouldBlock -> BudgetExceeded (kalan skip = 11 010 B).
    assert_eq!(
        flaky_multi.read_packet_into(&mut buf),
        Err(Error::BudgetExceeded {
            requested: 15_106,
            limit: 7_581,
        })
    );
    // Çağrı 2: 2048 B daha atladıktan sonra WouldBlock -> InvalidData (kalan skip = 8 962 B).
    assert!(matches!(
        flaky_multi.read_packet_into(&mut buf),
        Err(Error::InvalidData(_))
    ));
    // Çağrı 3: Kalan 8 962 B atlanır, 2. paket başlığının ilk 5 baytı okunur ve WouldBlock -> InvalidData.
    assert!(matches!(
        flaky_multi.read_packet_into(&mut buf),
        Err(Error::InvalidData(_))
    ));
    // Çağrı 4: Başlığın kalan 7 baytı ve 7 581 B yük tamamlanır -> Ok(Some(33)).
    assert_eq!(flaky_multi.read_packet_into(&mut buf), Ok(Some(33)));
    assert_eq!(buf, pkt2);
}
