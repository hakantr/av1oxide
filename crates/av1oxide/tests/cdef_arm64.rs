//! P51 (`F5a` — `F5-CDEF`) `aarch64` NEON CDEF (`8 / 8` matris kaydı = `12` yuva:
//! `dir` + `fb[0..2]` @ `8 / 10 / 12` bpc) bit-birebir doğruluk, `SentinelBuffer` ve
//! `OverReadProbe` testleri.

#![forbid(unsafe_code)]

#[cfg(target_arch = "aarch64")]
mod common;

use av1oxide::cpu::{ArmCpuId, CpuFlags, CpuLevel};
use av1oxide::dsp::{ARM_ISA_CANDIDATES, DspDispatch};

#[cfg(target_arch = "aarch64")]
use av1oxide::dsp::cdef::{CdefBitDepth, CdefBlockSize, CdefEdgeFlags, CdefFilterParams};
#[cfg(target_arch = "aarch64")]
use av1oxide::frame::BitDepth;
#[cfg(target_arch = "aarch64")]
use common::sentinel::{OverReadProbe, SentinelBuffer};

#[test]
fn f5a_arm64_cdef_aday_ve_yetki_sozlesmesi() {
    let cdef_adaylar: Vec<_> = ARM_ISA_CANDIDATES
        .iter()
        .filter(|c| c.family == "cdef")
        .collect();
    assert_eq!(cdef_adaylar.len(), 12);
    for bd in [8u8, 10, 12] {
        for field in ["dir", "fb[0]", "fb[1]", "fb[2]"] {
            assert!(
                cdef_adaylar.iter().any(|c| c.bit_depth == bd
                    && c.field == field
                    && c.implementation == "neon"
                    && c.requires == CpuFlags::NEON
                    && c.default),
                "eksik ARM64 CDEF adayı: cdef/{field}@{bd}"
            );
        }
    }

    let arm_id = ArmCpuId {
        neon: true,
        dotprod: true,
        ..ArmCpuId::default()
    };
    assert_eq!(arm_id.flags().dav1d_arm_bits(), 0x03);
    assert_eq!(CpuLevel::from_name("neon"), Some(CpuLevel::Neon));
    assert_eq!(CpuLevel::from_name("dotprod"), Some(CpuLevel::Dotprod));

    let skaler = DspDispatch::select(CpuLevel::None.allowed());
    assert_eq!(skaler.assigned_count(), 0);
    assert!(skaler.neon_token().is_none());

    #[cfg(target_arch = "aarch64")]
    {
        let neon = DspDispatch::select(CpuFlags::detect());
        assert!(neon.neon_token().is_some());
        // P52: atanan yuva sayısı, `ARM_ISA_CANDIDATES`'teki farklı (aile, alan,
        // bit derinliği) yuvalarının sayısıdır (F5 aileleri eklendikçe büyür).
        let mut yuvalar: Vec<_> = ARM_ISA_CANDIDATES
            .iter()
            .map(|c| (c.family, c.field, c.bit_depth))
            .collect();
        yuvalar.sort_unstable();
        yuvalar.dedup();
        assert_eq!(neon.assigned_count(), yuvalar.len());
        for bd in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            for field in ["dir", "fb[0]", "fb[1]", "fb[2]"] {
                assert_eq!(skaler.slot_implementation(bd, "cdef", field), "skaler");
                assert_eq!(neon.slot_implementation(bd, "cdef", field), "neon");
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5a_cdef_dir_8_10_12bpc_neon_skaler_bit_exact_ve_overread() {
    let d_neon = DspDispatch::select(CpuFlags::detect());
    let d_skaler = DspDispatch::select(CpuFlags::NONE);

    // 8-bpc find_dir + OverReadProbe
    {
        let neon_fn = d_neon.for_bit_depth(BitDepth::Eight).cdef().dir;
        let skaler_fn = d_skaler.for_bit_depth(BitDepth::Eight).cdef().dir;
        for stride in [8usize, 16, 31] {
            let len = 7 * stride + 8;
            let probe = OverReadProbe::new(len, 64, 8);
            let off = probe.data_offset();
            let res = probe
                .run(|inp, out| {
                    let slice = &inp[off..off + len];
                    let r = neon_fn(slice, stride, CdefBitDepth::B8);
                    out[0..4].copy_from_slice(&u32::from(r.dir).to_le_bytes());
                    out[4..8].copy_from_slice(&r.var.to_le_bytes());
                })
                .expect("8-bpc cdef/dir OverReadProbe taşması");
            let payload = probe.payload();
            let ref_r = skaler_fn(&payload, stride, CdefBitDepth::B8);
            assert_eq!(
                u32::from_le_bytes([res[0], res[1], res[2], res[3]]) as u8,
                ref_r.dir
            );
            assert_eq!(
                u32::from_le_bytes([res[4], res[5], res[6], res[7]]),
                ref_r.var
            );
        }
    }

    // 10/12-bpc find_dir + OverReadProbe
    for (bd, cbd, max_val) in [
        (BitDepth::Ten, CdefBitDepth::B10, 1023u16),
        (BitDepth::Twelve, CdefBitDepth::B12, 4095u16),
    ] {
        let neon_fn = d_neon.for_bit_depth(bd).cdef().dir;
        let skaler_fn = d_skaler.for_bit_depth(bd).cdef().dir;
        for stride in [16usize, 32, 48] {
            let len = 7 * stride + 16;
            let mut buf = vec![0u8; len];
            for (i, chunk2) in buf.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let v = ((i as u32 * 251 + 73) % u32::from(max_val + 1)) as u16;
                *chunk2 = v.to_le_bytes();
            }
            let r_neon = neon_fn(&buf, stride, cbd);
            let r_skaler = skaler_fn(&buf, stride, cbd);
            assert_eq!(
                r_neon, r_skaler,
                "16-bpc cdef/dir farkı ({bd:?}, stride={stride})"
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5a_cdef_filter_8_10_12bpc_neon_skaler_bit_exact_ve_sentinel() {
    let d_neon = DspDispatch::select(CpuFlags::detect());
    let d_skaler = DspDispatch::select(CpuFlags::NONE);

    let sizes = [
        (0usize, CdefBlockSize::W8H8),
        (1usize, CdefBlockSize::W4H8),
        (2usize, CdefBlockSize::W4H4),
    ];

    for (bd, cbd, bpp, max_val) in [
        (BitDepth::Eight, CdefBitDepth::B8, 1usize, 255u16),
        (BitDepth::Ten, CdefBitDepth::B10, 2usize, 1023u16),
        (BitDepth::Twelve, CdefBitDepth::B12, 2usize, 4095u16),
    ] {
        for &(slot_idx, size) in &sizes {
            let neon_fn = d_neon.for_bit_depth(bd).cdef().fb[slot_idx];
            let skaler_fn = d_skaler.for_bit_depth(bd).cdef().fb[slot_idx];
            let w = size.w();
            let h = size.h();
            let stride = (w + 8) * bpp;
            let dst_len = (h - 1) * stride + (w + 2) * bpp;
            let left_len = 2 * h * bpp;
            let tb_len = stride + (w + 4) * bpp;

            let mut left = vec![0u8; left_len];
            let mut top = vec![0u8; tb_len];
            let mut bottom = vec![0u8; tb_len];
            let mut init_dst = vec![0u8; dst_len];

            for (i, b) in init_dst
                .iter_mut()
                .chain(left.iter_mut())
                .chain(top.iter_mut())
                .chain(bottom.iter_mut())
                .enumerate()
            {
                *b = ((i * 79 + 19) & 0xFF) as u8;
            }
            if bpp == 2 {
                for buf in [&mut init_dst, &mut left, &mut top, &mut bottom] {
                    for c2 in buf.as_chunks_mut::<2>().0 {
                        let v = u16::from_le_bytes(*c2) & max_val;
                        *c2 = v.to_le_bytes();
                    }
                }
            }

            for edge_bits in 0u8..16 {
                let edges = CdefEdgeFlags::from_bits(edge_bits).expect("geçerli kenar bayrağı");
                for dir in 0u8..8 {
                    let shift = cbd.min_8();
                    let params = CdefFilterParams {
                        pri_strength: 5 << shift,
                        sec_strength: 2 << shift,
                        dir,
                        damping: 4 + shift as u8,
                        size,
                        edges,
                        bit_depth: cbd,
                    };
                    let mut sb_neon = SentinelBuffer::with_defaults(dst_len);
                    let mut sb_skaler = SentinelBuffer::with_defaults(dst_len);
                    sb_neon.write_data(&init_dst);
                    sb_skaler.write_data(&init_dst);

                    neon_fn(sb_neon.data_mut(), stride, &left, &top, &bottom, params);
                    skaler_fn(sb_skaler.data_mut(), stride, &left, &top, &bottom, params);

                    sb_neon.verify().expect("NEON cdef filter sentinel ihlali");
                    sb_skaler
                        .verify()
                        .expect("skaler cdef filter sentinel ihlali");
                    assert_eq!(
                        sb_neon.data(),
                        sb_skaler.data(),
                        "cdef filter farkı: bd={bd:?} size={size:?} edges={edge_bits:#x} dir={dir}"
                    );
                }
            }
        }
    }
}
