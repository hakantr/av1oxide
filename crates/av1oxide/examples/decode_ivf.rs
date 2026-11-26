//! `av1oxide` çok-thread IVF akışı çözme ve YUV düzlem okuma örneği (`v0.5.0`).
//!
//! Çalıştırma:
//!   cargo run --release -p av1oxide --example decode_ivf
//!   cargo run --release -p av1oxide --example decode_ivf -- [--md5] [--no-grain] [girdi.ivf] [cikti.yuv]

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use av1oxide::container::ivf::IvfParser;
use av1oxide::{Config, Decoder, Frame, Packet, PlaneType, Result};

/// Sıfır dış bağımlılıklı, `%100` güvenli RFC 1321 MD5 özetleyici (`dav1d --muxer md5` uyumlu).
struct Md5Hasher {
    state: [u32; 4],
    total_bits: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Md5Hasher {
    const fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            total_bits: 0,
            buf: [0; 64],
            buf_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_bits = self.total_bits.wrapping_add((data.len() as u64) << 3);
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = data.len().min(need);
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
        let (chunks, _) = block.as_chunks::<4>();
        for (slot, chunk) in m.iter_mut().zip(chunks.iter()) {
            *slot = u32::from_le_bytes(*chunk);
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
            let sum = a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]);
            b = b.wrapping_add(sum.rotate_left(S[i]));
            a = temp;
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }

    fn finish_hex(mut self) -> String {
        let bit_len = self.total_bits;
        self.update(&[0x80]);
        let pad_len = if self.buf_len <= 56 {
            56 - self.buf_len
        } else {
            120 - self.buf_len
        };
        let zeros = [0u8; 64];
        let mut rem = pad_len;
        while rem > 0 {
            let take = rem.min(zeros.len());
            self.update(&zeros[..take]);
            rem -= take;
        }
        self.update(&bit_len.to_le_bytes());
        let mut out = String::with_capacity(32);
        for word in self.state {
            for byte in word.to_le_bytes() {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{byte:02x}");
            }
        }
        out
    }
}

/// Çözülen bir karenin Y, U, V düzlemlerini satır satır (stride dolgusunu atlayarak) MD5'e besler
/// ve isteğe bağlı YUV dosyasına yazar.
fn process_planar_yuv(
    hasher: &mut Md5Hasher,
    mut writer: Option<&mut BufWriter<File>>,
    frame: &Frame,
) -> std::io::Result<()> {
    for plane_type in [PlaneType::Y, PlaneType::U, PlaneType::V] {
        let Some(plane) = frame.plane(plane_type) else {
            continue;
        };
        for y in 0..plane.height() {
            if let Some(row) = plane.row(y) {
                hasher.update(row);
                if let Some(w) = writer.as_deref_mut() {
                    w.write_all(row)?;
                }
            }
        }
    }
    Ok(())
}

fn default_ivf_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf",
    )
}

/// `.crate` arşivinden bağımsız çalıştırıldığında (`tests/fixtures/` paket dışında olduğunda)
/// kullanılan gömülü 2 karelik (1×1, 8-bit) geçerli AV1 IVF akışı.
fn embedded_fallback_ivf() -> Vec<u8> {
    const FRAME_0: &[u8] = &[
        0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
        0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
        0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ];
    const FRAME_1: &[u8] = &[
        0x12, 0x00, 0x32, 0x1B, 0x30, 0x02, 0x01, 0x00, 0xA7, 0x2E, 0x47, 0xB0, 0x80, 0x00, 0x02,
        0x40, 0x00, 0x00, 0x80, 0x00, 0x00, 0x03, 0x5A, 0xF0, 0x1B, 0x0B, 0xA1, 0xEF, 0xD4, 0x62,
        0xC0,
    ];
    let mut ivf = Vec::with_capacity(32 + (12 + FRAME_0.len()) + (12 + FRAME_1.len()));
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes()); // version
    ivf.extend_from_slice(&32u16.to_le_bytes()); // header length
    ivf.extend_from_slice(b"AV01");
    ivf.extend_from_slice(&1u16.to_le_bytes()); // width
    ivf.extend_from_slice(&1u16.to_le_bytes()); // height
    ivf.extend_from_slice(&30u32.to_le_bytes()); // timebase den
    ivf.extend_from_slice(&1u32.to_le_bytes()); // timebase num
    ivf.extend_from_slice(&2u32.to_le_bytes()); // num frames
    ivf.extend_from_slice(&0u32.to_le_bytes()); // unused
    for (pts, frame) in [(0u64, FRAME_0), (1u64, FRAME_1)] {
        ivf.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&pts.to_le_bytes());
        ivf.extend_from_slice(frame);
    }
    ivf
}

fn main() -> Result<()> {
    let mut md5_only = false;
    let mut apply_grain = true;
    let mut positional: Vec<PathBuf> = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--md5" => md5_only = true,
            "--no-grain" => apply_grain = false,
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let cli_ivf_path = positional.first().cloned();
    let yuv_out_path = positional.get(1).cloned();

    let (source_label, ivf_bytes) = if let Some(path) = cli_ivf_path {
        let label = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("akis.ivf")
            .to_string();
        let bytes = std::fs::read(&path).expect("IVF dosyası okunamadı");
        (label, bytes)
    } else {
        let default_path = default_ivf_path();
        if let Ok(bytes) = std::fs::read(&default_path) {
            let label = default_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("akis.ivf")
                .to_string();
            (label, bytes)
        } else {
            ("embedded_1x1.ivf".to_string(), embedded_fallback_ivf())
        }
    };

    let parser = IvfParser::new(&ivf_bytes)?;

    // Çok-thread çözücü yapılandırması:
    // - n_threads: 4 işçi thread'i (wasm32 üzerinde otomatik olarak tek iş parçacığına düşer)
    // - max_frame_delay: 2 eşzamanlı kare yuvası (kareler arası + kare içi paralellik)
    let config = Config {
        n_threads: 4,
        max_frame_delay: 2,
        apply_grain,
        ..Config::default()
    };
    let mut decoder = Decoder::new(config)?;
    let mut yuv_writer = yuv_out_path
        .as_ref()
        .map(|p| BufWriter::new(File::create(p).expect("YUV çıktı dosyası oluşturulamadı")));

    let mut frame_count = 0usize;
    let mut last_dims = (0usize, 0usize);
    let mut hasher = Md5Hasher::new();

    for pkt in parser {
        let pkt = pkt?;
        decoder.send_packet(&Packet::with_timestamp(pkt.data, pkt.pts))?;
        while let Some(frame) = decoder.receive_frame()? {
            last_dims = (frame.width(), frame.height());
            process_planar_yuv(&mut hasher, yuv_writer.as_mut(), &frame).expect("YUV yazma hatası");
            frame_count += 1;
        }
    }

    // Boru hattında bekleyen son kareleri boşalt.
    decoder.flush();
    while let Some(frame) = decoder.receive_frame()? {
        last_dims = (frame.width(), frame.height());
        process_planar_yuv(&mut hasher, yuv_writer.as_mut(), &frame).expect("YUV yazma hatası");
        frame_count += 1;
    }

    if let Some(mut w) = yuv_writer {
        w.flush().expect("YUV tampon boşaltma hatası");
    }

    let md5_hex = hasher.finish_hex();
    if md5_only {
        println!(
            "frames={} dims={}x{} md5={}",
            frame_count, last_dims.0, last_dims.1, md5_hex
        );
    } else {
        println!(
            "av1oxide v{}: '{}' içinden {} kare çözüldü ({}x{}, md5: {}, tepe tahsis: {} bayt).",
            av1oxide::VERSION,
            source_label,
            frame_count,
            last_dims.0,
            last_dims.1,
            md5_hex,
            decoder.allocation_peak()
        );
    }
    assert!(frame_count > 0, "en az bir kare çözülmeli");
    Ok(())
}
