//! `libdav1d` (`1.5.4`, API `7.0.0`) C-ABI uyumluluk, yaşam döngüsü ve C derleyicisi (`cc`) ikili düzen testleri.

use core::ffi::{CStr, c_int, c_void};
use core::mem::{MaybeUninit, offset_of, size_of};
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::path::PathBuf;
use std::process::Command;

use av1oxide::capi::*;
use av1oxide::{Config, Decoder, Packet, PlaneType};

/// Gömülü 2 paketlik (1×1, 8-bit, 4:2:0) örnek AV1 OBU akışı.
const SAMPLE_OBU_PACKETS: &[&[u8]] = &[
    &[
        0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
        0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
        0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ],
    &[
        0x12, 0x00, 0x32, 0x1B, 0x30, 0x02, 0x01, 0x00, 0xA7, 0x2E, 0x47, 0xB0, 0x80, 0x00, 0x02,
        0x40, 0x00, 0x00, 0x80, 0x00, 0x00, 0x03, 0x5A, 0xF0, 0x1B, 0x0B, 0xA1, 0xEF, 0xD4, 0x62,
        0xC0,
    ],
];

fn parse_ivf(data: &[u8]) -> Vec<(i64, &[u8])> {
    assert!(data.len() >= 32 && &data[0..4] == b"DKIF");
    let mut offset = 32usize;
    let mut out = Vec::new();
    while offset + 12 <= data.len() {
        let sz = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        let pts = i64::from_le_bytes(data[offset + 4..offset + 12].try_into().unwrap());
        offset += 12;
        if offset + sz > data.len() {
            break;
        }
        out.push((pts, &data[offset..offset + sz]));
        offset += sz;
    }
    out
}

fn copy_c_picture_planes(pic: &Dav1dPicture) -> Vec<Vec<u8>> {
    let mut planes = Vec::new();
    let w = pic.p.w as usize;
    let h = pic.p.h as usize;
    let bps = if pic.p.bpc > 8 { 2 } else { 1 };
    let (cw, ch) = match pic.p.layout {
        DAV1D_PIXEL_LAYOUT_I400 => (0, 0),
        DAV1D_PIXEL_LAYOUT_I420 => (w.div_ceil(2), h.div_ceil(2)),
        DAV1D_PIXEL_LAYOUT_I422 => (w.div_ceil(2), h),
        DAV1D_PIXEL_LAYOUT_I444 => (w, h),
        _ => unreachable!(),
    };
    let dims = [(w, h), (cw, ch), (cw, ch)];
    for (idx, &(pw, ph)) in dims.iter().enumerate() {
        if pw == 0 || ph == 0 {
            continue;
        }
        let base = pic.data[idx] as *const u8;
        assert!(!base.is_null(), "düzlem {idx} null olmamalı");
        let stride = pic.stride[usize::from(idx > 0)];
        let row_bytes = pw * bps;
        let mut buf = Vec::with_capacity(row_bytes * ph);
        for y in 0..ph {
            let row = unsafe {
                core::slice::from_raw_parts(base.offset((y as isize) * stride), row_bytes)
            };
            buf.extend_from_slice(row);
        }
        planes.push(buf);
    }
    planes
}

#[test]
fn capi_version_settings_and_sequence_header() {
    unsafe {
        let v = CStr::from_ptr(dav1d_version()).to_str().unwrap();
        assert_eq!(v, "0.3.2");
        assert_eq!(dav1d_version_api(), 0x0007_0000);

        dav1d_default_settings(ptr::null_mut());
        assert_eq!(dav1d_get_frame_delay(ptr::null()), dav1d_err(EINVAL));

        let mut s = Dav1dSettings {
            n_threads: 999,
            ..Dav1dSettings::default()
        };
        dav1d_default_settings(&mut s);
        assert_eq!(s.n_threads, 0);
        assert_eq!(s.max_frame_delay, 0);
        assert_eq!(s.apply_grain, 1);
        assert_eq!(s.all_layers, 1);
        assert_eq!(s.inloop_filters, DAV1D_INLOOPFILTER_ALL);
        assert_eq!(s.decode_frame_type, DAV1D_DECODEFRAMETYPE_ALL);

        s.n_threads = 4;
        s.max_frame_delay = 0;
        assert_eq!(dav1d_get_frame_delay(&s), 2);
        s.n_threads = -1;
        assert_eq!(dav1d_get_frame_delay(&s), dav1d_err(EINVAL));

        // Sequence header ayrıştırma kontrolleri
        let mut seq = MaybeUninit::<Dav1dSequenceHeader>::zeroed();
        assert_eq!(
            dav1d_parse_sequence_header(ptr::null_mut(), SAMPLE_OBU_PACKETS[0].as_ptr(), 10),
            dav1d_err(EINVAL)
        );
        assert_eq!(
            dav1d_parse_sequence_header(seq.as_mut_ptr(), ptr::null(), 10),
            dav1d_err(EINVAL)
        );
        assert_eq!(
            dav1d_parse_sequence_header(seq.as_mut_ptr(), SAMPLE_OBU_PACKETS[0].as_ptr(), 0),
            dav1d_err(EINVAL)
        );
        // Yalnız Temporal Delimiter içeren tampon -> -ENOENT
        let td_only = [0x12u8, 0x00];
        assert_eq!(
            dav1d_parse_sequence_header(seq.as_mut_ptr(), td_only.as_ptr(), td_only.len()),
            dav1d_err(ENOENT)
        );
        // Geçerli paket -> 0
        assert_eq!(
            dav1d_parse_sequence_header(
                seq.as_mut_ptr(),
                SAMPLE_OBU_PACKETS[0].as_ptr(),
                SAMPLE_OBU_PACKETS[0].len()
            ),
            0
        );
        let parsed = seq.assume_init();
        assert_eq!(parsed.profile, 0);
        assert_eq!((parsed.max_width, parsed.max_height), (1, 1));
        assert_eq!(parsed.layout, DAV1D_PIXEL_LAYOUT_I420);
        assert_eq!(parsed.hbd, 0);
    }
}

extern "C" fn count_free_cb(_ptr: *const u8, cookie: *mut c_void) {
    let counter = unsafe { &*(cookie as *const AtomicUsize) };
    counter.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn capi_data_wrap_user_data_error_props_and_flush() {
    unsafe {
        let mut s = Dav1dSettings::default();
        dav1d_default_settings(&mut s);
        s.n_threads = 1;
        s.max_frame_delay = 1;

        // Tek taraflı özel allocator reddedilmeli
        let mut bad_s = s;
        bad_s.allocator.cookie = ptr::dangling_mut::<c_void>();
        let mut c: *mut Dav1dContext = ptr::null_mut();
        assert_eq!(dav1d_open(&mut c, &bad_s), dav1d_err(EINVAL));
        assert!(c.is_null());

        assert_eq!(dav1d_open(&mut c, &s), 0);
        assert!(!c.is_null());

        let buf_frees = AtomicUsize::new(0);
        let user_frees = AtomicUsize::new(0);
        let user_tag = [0xAAu8, 0xBB, 0xCC, 0xDD];

        // 1. Bozuk paket gönderildiğinde hata dönmeli, `data` çağıranın sahipliğinde kalmalı
        //    ve `dav1d_get_decode_error_data_props` bozuk paketin metaverisini döndürmeli.
        let bad_obu = [0x0Au8, 0x7F, 0x00];
        let mut bad_data = Dav1dData::default();
        assert_eq!(
            dav1d_data_wrap(
                &mut bad_data,
                bad_obu.as_ptr(),
                bad_obu.len(),
                Some(count_free_cb),
                (&buf_frees as *const AtomicUsize) as *mut c_void,
            ),
            0
        );
        assert_eq!(
            dav1d_data_wrap_user_data(
                &mut bad_data,
                user_tag.as_ptr(),
                Some(count_free_cb),
                (&user_frees as *const AtomicUsize) as *mut c_void,
            ),
            0
        );
        bad_data.m.timestamp = 9876;
        bad_data.m.offset = 54321;

        assert_eq!(dav1d_send_data(c, &mut bad_data), dav1d_err(EINVAL));
        assert!(
            !bad_data.data.is_null(),
            "hata durumunda Dav1dData çağıranın sahipliğinde kalmalı"
        );
        dav1d_data_unref(&mut bad_data);
        assert_eq!(buf_frees.load(Ordering::SeqCst), 1);
        // `cached_error_props` hâlâ `user_data` referansını tutuyor olmalı:
        assert_eq!(user_frees.load(Ordering::SeqCst), 0);

        let mut err_props = Dav1dDataProps::default();
        assert_eq!(dav1d_get_decode_error_data_props(c, &mut err_props), 0);
        assert_eq!(err_props.timestamp, 9876);
        assert_eq!(err_props.offset, 54321);
        assert_eq!(err_props.user_data.data, user_tag.as_ptr());
        dav1d_data_props_unref(&mut err_props);
        assert_eq!(user_frees.load(Ordering::SeqCst), 1);

        // 2. `dav1d_flush` sonrası geçerli paketler çözülmeli ve `user_data` çıktı resmine taşınmalı.
        dav1d_flush(c);

        let mut good_data = Dav1dData::default();
        assert_eq!(
            dav1d_data_wrap(
                &mut good_data,
                SAMPLE_OBU_PACKETS[0].as_ptr(),
                SAMPLE_OBU_PACKETS[0].len(),
                Some(count_free_cb),
                (&buf_frees as *const AtomicUsize) as *mut c_void,
            ),
            0
        );
        assert_eq!(
            dav1d_data_wrap_user_data(
                &mut good_data,
                user_tag.as_ptr(),
                Some(count_free_cb),
                (&user_frees as *const AtomicUsize) as *mut c_void,
            ),
            0
        );
        good_data.m.timestamp = 111;
        good_data.m.duration = 33;
        good_data.m.offset = 222;

        assert_eq!(dav1d_send_data(c, &mut good_data), 0);
        assert_eq!(buf_frees.load(Ordering::SeqCst), 2);

        let mut pic = Dav1dPicture::default();
        assert_eq!(dav1d_get_picture(c, &mut pic), 0);
        assert_eq!(pic.m.timestamp, 111);
        assert_eq!(pic.m.duration, 33);
        assert_eq!(pic.m.offset, 222);
        assert_eq!(pic.m.user_data.data, user_tag.as_ptr());

        // `apply_grain` grain olmayan resimde sığ referans kopyası döndürmeli
        let mut pic2 = Dav1dPicture::default();
        assert_eq!(dav1d_apply_grain(c, &mut pic2, &pic), 0);
        assert_eq!(pic2.data[0], pic.data[0]);
        dav1d_picture_unref(&mut pic2);

        // `apply_grain` etkin grain verisi (`num_y_points > 0`) taşıyan resimde yeni düzlem ayırıp sentezlemeli
        (*pic.frame_hdr).film_grain.present = 1;
        (*pic.frame_hdr).film_grain.update = 1;
        (*pic.frame_hdr).film_grain.data.seed = 7391;
        (*pic.frame_hdr).film_grain.data.num_y_points = 2;
        (*pic.frame_hdr).film_grain.data.y_points[0] = [0, 64];
        (*pic.frame_hdr).film_grain.data.y_points[1] = [255, 64];
        (*pic.frame_hdr).film_grain.data.scaling_shift = 11;
        (*pic.frame_hdr).film_grain.data.ar_coeff_shift = 6;
        let mut grained_pic = Dav1dPicture::default();
        assert_eq!(dav1d_apply_grain(c, &mut grained_pic, &pic), 0);
        assert!(!grained_pic.data[0].is_null());
        assert_ne!(
            grained_pic.data[0], pic.data[0],
            "etkin grain sentezinde yeni düzlem ayrılmalı"
        );
        assert_eq!((grained_pic.data[0] as usize) % 64, 0);
        dav1d_picture_unref(&mut grained_pic);

        dav1d_picture_unref(&mut pic);
        dav1d_close(&mut c);
        assert!(c.is_null());
        assert_eq!(user_frees.load(Ordering::SeqCst), 2);
    }
}

struct CustomAllocTracker {
    allocs: AtomicUsize,
    releases: AtomicUsize,
}

extern "C" fn custom_alloc_pic(pic: *mut Dav1dPicture, cookie: *mut c_void) -> c_int {
    unsafe {
        let tracker = &*(cookie as *const CustomAllocTracker);
        tracker.allocs.fetch_add(1, Ordering::SeqCst);
        let p = &mut *pic;
        let w = (p.p.w as usize).next_multiple_of(128);
        let h = (p.p.h as usize).next_multiple_of(128);
        let bps = if p.p.bpc > 8 { 2 } else { 1 };
        let luma_stride = (w * bps).next_multiple_of(64);
        let chroma_stride = ((w / 2) * bps).next_multiple_of(64);
        let luma_bytes = luma_stride * h + 64;
        let chroma_bytes = chroma_stride * (h / 2) + 64;

        let total = luma_bytes + 2 * chroma_bytes + 64;
        let mut raw = vec![0u8; total].into_boxed_slice();
        let base = raw.as_mut_ptr();
        let aligned_offset = base.align_offset(64);
        let y_ptr = base.add(aligned_offset);
        let u_ptr = y_ptr.add(luma_bytes);
        let v_ptr = u_ptr.add(chroma_bytes);

        p.data[0] = y_ptr as *mut c_void;
        p.data[1] = u_ptr as *mut c_void;
        p.data[2] = v_ptr as *mut c_void;
        p.stride[0] = luma_stride as isize;
        p.stride[1] = chroma_stride as isize;
        p.allocator_data = Box::into_raw(raw) as *mut u8 as *mut c_void;
        0
    }
}

extern "C" fn custom_release_pic(pic: *mut Dav1dPicture, cookie: *mut c_void) {
    unsafe {
        let tracker = &*(cookie as *const CustomAllocTracker);
        tracker.releases.fetch_add(1, Ordering::SeqCst);
        let p = &mut *pic;
        if !p.allocator_data.is_null() {
            let w = (p.p.w as usize).next_multiple_of(128);
            let h = (p.p.h as usize).next_multiple_of(128);
            let bps = if p.p.bpc > 8 { 2 } else { 1 };
            let luma_stride = (w * bps).next_multiple_of(64);
            let chroma_stride = ((w / 2) * bps).next_multiple_of(64);
            let total = (luma_stride * h + 64) + 2 * (chroma_stride * (h / 2) + 64) + 64;
            let slice_ptr = core::ptr::slice_from_raw_parts_mut(p.allocator_data as *mut u8, total);
            drop(Box::from_raw(slice_ptr));
            p.allocator_data = ptr::null_mut();
        }
    }
}

#[test]
fn capi_ivf_single_and_multithread_parity_and_custom_allocator() {
    let ivf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf",
    );
    let ivf_bytes = std::fs::read(&ivf_path).expect("test IVF akışı okunmalı");
    let packets = parse_ivf(&ivf_bytes);
    assert_eq!(packets.len(), 10);

    // Referans saf-Rust çıktıları
    let mut rust_dec = Decoder::new(Config::default()).unwrap();
    let mut expected_frames: Vec<Vec<Vec<u8>>> = Vec::new();
    for &(pts, payload) in &packets {
        rust_dec
            .send_packet(&Packet::with_timestamp(payload, pts))
            .unwrap();
        while let Some(f) = rust_dec.receive_frame().unwrap() {
            let mut planes = Vec::new();
            for pt in [PlaneType::Y, PlaneType::U, PlaneType::V] {
                let p = f.plane(pt).unwrap();
                let mut bytes = Vec::new();
                for y in 0..p.height() {
                    bytes.extend_from_slice(p.row(y).unwrap());
                }
                planes.push(bytes);
            }
            expected_frames.push(planes);
        }
    }
    rust_dec.flush();
    while let Some(f) = rust_dec.receive_frame().unwrap() {
        let mut planes = Vec::new();
        for pt in [PlaneType::Y, PlaneType::U, PlaneType::V] {
            let p = f.plane(pt).unwrap();
            let mut bytes = Vec::new();
            for y in 0..p.height() {
                bytes.extend_from_slice(p.row(y).unwrap());
            }
            planes.push(bytes);
        }
        expected_frames.push(planes);
    }
    assert_eq!(expected_frames.len(), 10);

    // 1. Tek-thread (`1, 1`) ve çok-thread (`4, 2`) varsayılan ayırıcı ile doğrulama
    for (n_threads, max_frame_delay) in [(1, 1), (4, 2)] {
        unsafe {
            let mut s = Dav1dSettings::default();
            dav1d_default_settings(&mut s);
            s.n_threads = n_threads;
            s.max_frame_delay = max_frame_delay;

            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!(dav1d_open(&mut c, &s), 0);

            let mut actual_frames = Vec::new();
            let mut pic = Dav1dPicture::default();
            for &(pts, payload) in &packets {
                let mut data = Dav1dData::default();
                let dst = dav1d_data_create(&mut data, payload.len());
                assert!(!dst.is_null());
                ptr::copy_nonoverlapping(payload.as_ptr(), dst, payload.len());
                data.m.timestamp = pts;
                assert_eq!(dav1d_send_data(c, &mut data), 0);

                loop {
                    let res = dav1d_get_picture(c, &mut pic);
                    if res == dav1d_err(EAGAIN) {
                        break;
                    }
                    assert_eq!(res, 0);
                    for (i, &plane_ptr) in pic.data.iter().enumerate() {
                        assert_eq!(
                            (plane_ptr as usize) % 64,
                            0,
                            "düzlem {i} 64-bayt hizalı olmalı"
                        );
                    }
                    actual_frames.push(copy_c_picture_planes(&pic));
                    dav1d_picture_unref(&mut pic);
                }
            }
            // EOS drain döngüsü (`dav1d.h` sözleşmesi)
            loop {
                let res = dav1d_get_picture(c, &mut pic);
                if res == dav1d_err(EAGAIN) {
                    break;
                }
                assert_eq!(res, 0);
                actual_frames.push(copy_c_picture_planes(&pic));
                dav1d_picture_unref(&mut pic);
            }
            dav1d_close(&mut c);
            assert_eq!(
                actual_frames, expected_frames,
                "C-ABI (threads={n_threads}, delay={max_frame_delay}) çıktısı Rust Decoder ile birebir aynı olmalı"
            );
        }
    }

    // 2. Özel `Dav1dPicAllocator` ile doğrulama
    unsafe {
        let tracker = CustomAllocTracker {
            allocs: AtomicUsize::new(0),
            releases: AtomicUsize::new(0),
        };
        let mut s = Dav1dSettings::default();
        dav1d_default_settings(&mut s);
        s.n_threads = 2;
        s.max_frame_delay = 1;
        s.allocator = Dav1dPicAllocator {
            cookie: (&tracker as *const CustomAllocTracker) as *mut c_void,
            alloc_picture_callback: Some(custom_alloc_pic),
            release_picture_callback: Some(custom_release_pic),
        };

        let mut c: *mut Dav1dContext = ptr::null_mut();
        assert_eq!(dav1d_open(&mut c, &s), 0);

        let mut actual_frames = Vec::new();
        let mut pic = Dav1dPicture::default();
        for &(pts, payload) in &packets {
            let mut data = Dav1dData::default();
            let dst = dav1d_data_create(&mut data, payload.len());
            ptr::copy_nonoverlapping(payload.as_ptr(), dst, payload.len());
            data.m.timestamp = pts;
            assert_eq!(dav1d_send_data(c, &mut data), 0);
            while dav1d_get_picture(c, &mut pic) == 0 {
                actual_frames.push(copy_c_picture_planes(&pic));
                dav1d_picture_unref(&mut pic);
            }
        }
        while dav1d_get_picture(c, &mut pic) == 0 {
            actual_frames.push(copy_c_picture_planes(&pic));
            dav1d_picture_unref(&mut pic);
        }
        dav1d_close(&mut c);

        assert_eq!(actual_frames, expected_frames);
        assert_eq!(tracker.allocs.load(Ordering::SeqCst), 10);
        assert_eq!(tracker.releases.load(Ordering::SeqCst), 10);
    }
}

#[test]
fn capi_c_compiler_struct_layout_and_staticlib_linking() {
    let Ok(cc_check) = Command::new("cc").arg("--version").output() else {
        return;
    };
    if !cc_check.status.success() {
        return;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let include_dir = manifest_dir.join("include");
    let scratch_dir = workspace_root.join(".cache/capi-abi-test");
    let _ = std::fs::remove_dir_all(&scratch_dir);
    std::fs::create_dir_all(&scratch_dir).unwrap();

    // 1. C derleyicisi (`cc`) ile `include/dav1d/dav1d.h` üzerinden `sizeof` ve `offsetof` kontrolü
    let c_layout_src = r#"
#include <stdio.h>
#include <stddef.h>
#include <dav1d/dav1d.h>

int main(void) {
    printf("sizeof_Dav1dUserData=%zu\n", sizeof(Dav1dUserData));
    printf("sizeof_Dav1dDataProps=%zu\n", sizeof(Dav1dDataProps));
    printf("sizeof_Dav1dData=%zu\n", sizeof(Dav1dData));
    printf("sizeof_Dav1dWarpedMotionParams=%zu\n", sizeof(Dav1dWarpedMotionParams));
    printf("sizeof_Dav1dContentLightLevel=%zu\n", sizeof(Dav1dContentLightLevel));
    printf("sizeof_Dav1dMasteringDisplay=%zu\n", sizeof(Dav1dMasteringDisplay));
    printf("sizeof_Dav1dITUTT35=%zu\n", sizeof(Dav1dITUTT35));
    printf("sizeof_Dav1dSequenceHeader=%zu\n", sizeof(Dav1dSequenceHeader));
    printf("offset_seq_operating_points=%zu\n", offsetof(Dav1dSequenceHeader, operating_points));
    printf("offset_seq_operating_parameter_info=%zu\n", offsetof(Dav1dSequenceHeader, operating_parameter_info));
    printf("sizeof_Dav1dSegmentationData=%zu\n", sizeof(Dav1dSegmentationData));
    printf("sizeof_Dav1dSegmentationDataSet=%zu\n", sizeof(Dav1dSegmentationDataSet));
    printf("sizeof_Dav1dLoopfilterModeRefDeltas=%zu\n", sizeof(Dav1dLoopfilterModeRefDeltas));
    printf("sizeof_Dav1dFilmGrainData=%zu\n", sizeof(Dav1dFilmGrainData));
    printf("sizeof_Dav1dFrameHeader=%zu\n", sizeof(Dav1dFrameHeader));
    printf("offset_fh_film_grain=%zu\n", offsetof(Dav1dFrameHeader, film_grain));
    printf("offset_fh_width=%zu\n", offsetof(Dav1dFrameHeader, width));
    printf("offset_fh_tiling=%zu\n", offsetof(Dav1dFrameHeader, tiling));
    printf("offset_fh_quant=%zu\n", offsetof(Dav1dFrameHeader, quant));
    printf("offset_fh_segmentation=%zu\n", offsetof(Dav1dFrameHeader, segmentation));
    printf("offset_fh_loopfilter=%zu\n", offsetof(Dav1dFrameHeader, loopfilter));
    printf("offset_fh_cdef=%zu\n", offsetof(Dav1dFrameHeader, cdef));
    printf("offset_fh_restoration=%zu\n", offsetof(Dav1dFrameHeader, restoration));
    printf("offset_fh_gmv=%zu\n", offsetof(Dav1dFrameHeader, gmv));
    printf("sizeof_Dav1dPictureParameters=%zu\n", sizeof(Dav1dPictureParameters));
    printf("sizeof_Dav1dPicture=%zu\n", sizeof(Dav1dPicture));
    printf("offset_pic_data=%zu\n", offsetof(Dav1dPicture, data));
    printf("offset_pic_stride=%zu\n", offsetof(Dav1dPicture, stride));
    printf("offset_pic_p=%zu\n", offsetof(Dav1dPicture, p));
    printf("offset_pic_m=%zu\n", offsetof(Dav1dPicture, m));
    printf("offset_pic_ref=%zu\n", offsetof(Dav1dPicture, ref));
    printf("offset_pic_allocator_data=%zu\n", offsetof(Dav1dPicture, allocator_data));
    printf("sizeof_Dav1dPicAllocator=%zu\n", sizeof(Dav1dPicAllocator));
    printf("sizeof_Dav1dLogger=%zu\n", sizeof(Dav1dLogger));
    printf("sizeof_Dav1dSettings=%zu\n", sizeof(Dav1dSettings));
    printf("offset_settings_allocator=%zu\n", offsetof(Dav1dSettings, allocator));
    printf("offset_settings_logger=%zu\n", offsetof(Dav1dSettings, logger));
    printf("offset_settings_inloop_filters=%zu\n", offsetof(Dav1dSettings, inloop_filters));
    printf("offset_settings_reserved=%zu\n", offsetof(Dav1dSettings, reserved));
    printf("c_eagain=%d\n", DAV1D_ERR(EAGAIN));
    return 0;
}
"#;
    let layout_c_path = scratch_dir.join("layout_check.c");
    let layout_bin_path = scratch_dir.join("layout_check");
    std::fs::write(&layout_c_path, c_layout_src).unwrap();

    let status = Command::new("cc")
        .arg("-I")
        .arg(&include_dir)
        .arg(&layout_c_path)
        .arg("-o")
        .arg(&layout_bin_path)
        .status()
        .expect("cc layout_check.c derlenmeli");
    assert!(status.success(), "cc layout_check.c derlemesi başarısız");

    let output = Command::new(&layout_bin_path).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();

    let rust_expected = [
        ("sizeof_Dav1dUserData", size_of::<Dav1dUserData>()),
        ("sizeof_Dav1dDataProps", size_of::<Dav1dDataProps>()),
        ("sizeof_Dav1dData", size_of::<Dav1dData>()),
        (
            "sizeof_Dav1dWarpedMotionParams",
            size_of::<Dav1dWarpedMotionParams>(),
        ),
        (
            "sizeof_Dav1dContentLightLevel",
            size_of::<Dav1dContentLightLevel>(),
        ),
        (
            "sizeof_Dav1dMasteringDisplay",
            size_of::<Dav1dMasteringDisplay>(),
        ),
        ("sizeof_Dav1dITUTT35", size_of::<Dav1dITUTT35>()),
        (
            "sizeof_Dav1dSequenceHeader",
            size_of::<Dav1dSequenceHeader>(),
        ),
        (
            "offset_seq_operating_points",
            offset_of!(Dav1dSequenceHeader, operating_points),
        ),
        (
            "offset_seq_operating_parameter_info",
            offset_of!(Dav1dSequenceHeader, operating_parameter_info),
        ),
        (
            "sizeof_Dav1dSegmentationData",
            size_of::<Dav1dSegmentationData>(),
        ),
        (
            "sizeof_Dav1dSegmentationDataSet",
            size_of::<Dav1dSegmentationDataSet>(),
        ),
        (
            "sizeof_Dav1dLoopfilterModeRefDeltas",
            size_of::<Dav1dLoopfilterModeRefDeltas>(),
        ),
        ("sizeof_Dav1dFilmGrainData", size_of::<Dav1dFilmGrainData>()),
        ("sizeof_Dav1dFrameHeader", size_of::<Dav1dFrameHeader>()),
        (
            "offset_fh_film_grain",
            offset_of!(Dav1dFrameHeader, film_grain),
        ),
        ("offset_fh_width", offset_of!(Dav1dFrameHeader, width)),
        ("offset_fh_tiling", offset_of!(Dav1dFrameHeader, tiling)),
        ("offset_fh_quant", offset_of!(Dav1dFrameHeader, quant)),
        (
            "offset_fh_segmentation",
            offset_of!(Dav1dFrameHeader, segmentation),
        ),
        (
            "offset_fh_loopfilter",
            offset_of!(Dav1dFrameHeader, loopfilter),
        ),
        ("offset_fh_cdef", offset_of!(Dav1dFrameHeader, cdef)),
        (
            "offset_fh_restoration",
            offset_of!(Dav1dFrameHeader, restoration),
        ),
        ("offset_fh_gmv", offset_of!(Dav1dFrameHeader, gmv)),
        (
            "sizeof_Dav1dPictureParameters",
            size_of::<Dav1dPictureParameters>(),
        ),
        ("sizeof_Dav1dPicture", size_of::<Dav1dPicture>()),
        ("offset_pic_data", offset_of!(Dav1dPicture, data)),
        ("offset_pic_stride", offset_of!(Dav1dPicture, stride)),
        ("offset_pic_p", offset_of!(Dav1dPicture, p)),
        ("offset_pic_m", offset_of!(Dav1dPicture, m)),
        ("offset_pic_ref", offset_of!(Dav1dPicture, r#ref)),
        (
            "offset_pic_allocator_data",
            offset_of!(Dav1dPicture, allocator_data),
        ),
        ("sizeof_Dav1dPicAllocator", size_of::<Dav1dPicAllocator>()),
        ("sizeof_Dav1dLogger", size_of::<Dav1dLogger>()),
        ("sizeof_Dav1dSettings", size_of::<Dav1dSettings>()),
        (
            "offset_settings_allocator",
            offset_of!(Dav1dSettings, allocator),
        ),
        ("offset_settings_logger", offset_of!(Dav1dSettings, logger)),
        (
            "offset_settings_inloop_filters",
            offset_of!(Dav1dSettings, inloop_filters),
        ),
        (
            "offset_settings_reserved",
            offset_of!(Dav1dSettings, reserved),
        ),
    ];

    for (key, expected_val) in rust_expected {
        let needle = format!("{key}={expected_val}");
        assert!(
            stdout.lines().any(|l| l.trim() == needle),
            "C/Rust ABI uyuşmazlığı: beklenen `{needle}`, C çıktısı:\n{stdout}"
        );
    }
    let eagain_line = format!("c_eagain={}", dav1d_err(EAGAIN));
    assert!(
        stdout.lines().any(|l| l.trim() == eagain_line),
        "DAV1D_ERR(EAGAIN) uyuşmazlığı: beklenen `{eagain_line}`, C çıktısı:\n{stdout}"
    );

    // 2. `--features capi --crate-type cdylib,staticlib` ile gerçek C programı bağlama ve çalıştırma testi
    let target_dir = scratch_dir.join("target");
    let cargo_status = Command::new("cargo")
        .current_dir(workspace_root)
        .arg("rustc")
        .arg("--release")
        .arg("-p")
        .arg("av1oxide")
        .arg("--features")
        .arg("capi")
        .arg("--crate-type")
        .arg("cdylib,staticlib")
        .arg("--target-dir")
        .arg(&target_dir)
        .status()
        .expect("cargo rustc --features capi çalışmalı");
    assert!(
        cargo_status.success(),
        "cdylib/staticlib derlemesi başarısız"
    );

    let staticlib_path = target_dir.join("release/libav1oxide.a");
    let cdylib_path = target_dir.join("release/libav1oxide.so");
    assert!(staticlib_path.exists(), "libav1oxide.a üretilmeli");
    if cdylib_path.exists()
        && let Ok(nm_out) = Command::new("nm").arg("-D").arg(&cdylib_path).output()
        && nm_out.status.success()
    {
        let nm_text = String::from_utf8_lossy(&nm_out.stdout);
        for sym in [
            "dav1d_version",
            "dav1d_version_api",
            "dav1d_default_settings",
            "dav1d_get_frame_delay",
            "dav1d_open",
            "dav1d_parse_sequence_header",
            "dav1d_send_data",
            "dav1d_get_picture",
            "dav1d_apply_grain",
            "dav1d_flush",
            "dav1d_close",
            "dav1d_get_event_flags",
            "dav1d_get_decode_error_data_props",
            "dav1d_data_create",
            "dav1d_data_wrap",
            "dav1d_data_wrap_user_data",
            "dav1d_data_unref",
            "dav1d_data_props_unref",
            "dav1d_picture_unref",
        ] {
            assert!(
                nm_text.contains(sym),
                "libav1oxide.so içinde `{sym}` sembolü dışa aktarılmalı"
            );
        }
    }

    let c_app_src = r#"
#include <assert.h>
#include <string.h>
#include <dav1d/dav1d.h>

static const uint8_t PKT0[] = {
    0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
    0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
    0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96
};

int main(void) {
    assert(dav1d_version_api() == 0x00070000U);
    assert(strcmp(dav1d_version(), "0.3.2") == 0);

    Dav1dSequenceHeader seq = { 0 };
    assert(dav1d_parse_sequence_header(&seq, PKT0, sizeof(PKT0)) == 0);
    assert(seq.max_width == 1 && seq.max_height == 1);

    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = 2;
    assert(dav1d_get_frame_delay(&s) >= 1);

    Dav1dContext *c = NULL;
    assert(dav1d_open(&c, &s) == 0);
    assert(c != NULL);

    Dav1dData data = { 0 };
    uint8_t *buf = dav1d_data_create(&data, sizeof(PKT0));
    assert(buf != NULL);
    memcpy(buf, PKT0, sizeof(PKT0));
    data.m.timestamp = 777;

    assert(dav1d_send_data(c, &data) == 0);

    Dav1dPicture pic = { 0 };
    int rc;
    do {
        rc = dav1d_get_picture(c, &pic);
    } while (rc == DAV1D_ERR(EAGAIN));
    assert(rc == 0);
    assert(pic.p.w == 1 && pic.p.h == 1 && pic.p.bpc == 8);
    assert(pic.m.timestamp == 777);
    assert(((uintptr_t)pic.data[0] & 63U) == 0U);
    assert(((const uint8_t *)pic.data[0])[0] == 190U);

    enum Dav1dEventFlags flags = 0;
    assert(dav1d_get_event_flags(c, &flags) == 0);
    assert((flags & DAV1D_EVENT_FLAG_NEW_SEQUENCE) != 0);

    dav1d_picture_unref(&pic);
    dav1d_close(&c);
    assert(c == NULL);
    return 0;
}
"#;
    let c_app_path = scratch_dir.join("c_app.c");
    let c_app_bin = scratch_dir.join("c_app");
    std::fs::write(&c_app_path, c_app_src).unwrap();

    let link_status = Command::new("cc")
        .arg("-I")
        .arg(&include_dir)
        .arg(&c_app_path)
        .arg(&staticlib_path)
        .arg("-lpthread")
        .arg("-ldl")
        .arg("-lm")
        .arg("-o")
        .arg(&c_app_bin)
        .status()
        .expect("cc c_app.c + libav1oxide.a bağlanmalı");
    assert!(
        link_status.success(),
        "C programı libav1oxide.a ile bağlanamadı"
    );

    let run_app = Command::new(&c_app_bin).status().unwrap();
    assert!(
        run_app.success(),
        "C programı çalışma zamanı doğrulaması başarısız"
    );

    let _ = std::fs::remove_dir_all(&scratch_dir);
}
