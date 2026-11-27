//! `libdav1d` (`1.5.4`, API `7.0.0`) C-ABI uyumluluk, yaşam döngüsü ve C derleyicisi (`cc`) ikili düzen testleri.

use core::ffi::{CStr, c_int, c_void};
use core::mem::{MaybeUninit, offset_of, size_of};
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashSet;
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
            // SAFETY: `base` null değildir ve `y * stride .. y * stride + row_bytes`
            // çözülmüş `Dav1dPicture` düzlem sınırları içindedir.
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
    // SAFETY: Test içindeki C-ABI çağrıları yerel yığın değişkenlerini ve statik
    // OBU dilimlerini geçerli yaşam süreleriyle kullanır.
    unsafe {
        let v = CStr::from_ptr(dav1d_version()).to_str().unwrap();
        assert_eq!(v, "0.5.1");
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
    // SAFETY: `cookie`, çağıran test çerçevesindeki canlı `AtomicUsize` adresidir.
    let counter = unsafe { &*(cookie as *const AtomicUsize) };
    counter.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn capi_data_wrap_user_data_error_props_and_flush() {
    // SAFETY: Test kapsamındaki tüm `Dav1dContext`, `Dav1dData` ve `Dav1dPicture`
    // işaretçileri `dav1d_close` / `dav1d_*_unref` çağrılarına kadar geçerlidir.
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
    // SAFETY: `pic` ve `cookie` (`CustomAllocTracker`) çözücü tarafından geçerli
    // adreslerle iletilir; ayrılan kutu `custom_release_pic` içinde serbest bırakılır.
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
    // SAFETY: `p.allocator_data`, `custom_alloc_pic` tarafından `Box::into_raw`
    // ile aynı `total` boyutta oluşturulmuş tekil tahsistir.
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
        // SAFETY: Her tur kendi bağımsız `Dav1dContext` örneğini açıp kapatır ve
        // paket yüklerini `dav1d_data_create` tamponuna kopyalar.
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
    // SAFETY: `tracker` yerel yığında `dav1d_close` tamamlanana dek canlı kalır.
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

    // 2. Resmî drop-in önek kurulum aracı (`examples/install_capi_prefix.rs`) ile mevcut
    //    ilgisiz dosyaları silmeden `prefix/` üzerine art arda iki kez kurulum, kök yol reddi,
    //    6 başlık, `dav1d.pc` / `pkg-config`, gerçek `DT_SONAME` / `install_name`, zorunlu `nm`
    //    tam sembol eşleşmesi ve hem statik hem dinamik C istemci doğrulaması.
    let prefix_dir = scratch_dir.join("prefix");
    let target_dir = scratch_dir.join("target");
    std::fs::create_dir_all(prefix_dir.join("lib")).unwrap();
    std::fs::create_dir_all(prefix_dir.join("include")).unwrap();
    let keep_root = prefix_dir.join("keep_root.txt");
    let keep_lib = prefix_dir.join("lib").join("libunrelated.a");
    let keep_inc = prefix_dir.join("include").join("unrelated.h");
    std::fs::write(&keep_root, b"keep-root-sentinel").unwrap();
    std::fs::write(&keep_lib, b"keep-lib-sentinel").unwrap();
    std::fs::write(&keep_inc, b"keep-inc-sentinel").unwrap();

    for pass in 0..2 {
        let install_status = Command::new("cargo")
            .current_dir(workspace_root)
            .arg("run")
            .arg("--locked")
            .arg("--release")
            .arg("-p")
            .arg("av1oxide")
            .arg("--example")
            .arg("install_capi_prefix")
            .arg("--")
            .arg(&prefix_dir)
            .arg("--target-dir")
            .arg(&target_dir)
            .status()
            .expect("resmî install_capi_prefix örneği çalışmalı");
        assert!(
            install_status.success(),
            "install_capi_prefix kurulumu {pass}. geçişte başarısız"
        );
        assert_eq!(
            std::fs::read(&keep_root).unwrap(),
            b"keep-root-sentinel",
            "ilgisiz kök dosyası korunmalı"
        );
        assert_eq!(
            std::fs::read(&keep_lib).unwrap(),
            b"keep-lib-sentinel",
            "ilgisiz lib dosyası korunmalı"
        );
        assert_eq!(
            std::fs::read(&keep_inc).unwrap(),
            b"keep-inc-sentinel",
            "ilgisiz include dosyası korunmalı"
        );
    }

    // Kök dizin `/` güvenlik nedeniyle reddedilmeli.
    #[cfg(unix)]
    {
        let root_reject = Command::new("cargo")
            .current_dir(workspace_root)
            .arg("run")
            .arg("--locked")
            .arg("--release")
            .arg("-p")
            .arg("av1oxide")
            .arg("--example")
            .arg("install_capi_prefix")
            .arg("--")
            .arg("/")
            .arg("--target-dir")
            .arg(&target_dir)
            .output()
            .expect("kök yol reddi çalıştırılmalı");
        assert!(
            !root_reject.status.success(),
            "kök dizin `/` hedef önek olarak reddedilmeli"
        );
    }

    // 2a. 6 resmî C başlık dosyasının varlığı ve kaynakla birebir eşleşmesi
    let prefix_inc_root = prefix_dir.join("include");
    let prefix_inc_dav1d = prefix_inc_root.join("dav1d");
    for hdr in [
        "common.h",
        "data.h",
        "dav1d.h",
        "headers.h",
        "picture.h",
        "version.h",
    ] {
        let installed = prefix_inc_dav1d.join(hdr);
        assert!(installed.is_file(), "eksik kurulu başlık: {hdr}");
        let expected_bytes = std::fs::read(include_dir.join("dav1d").join(hdr)).unwrap();
        let actual_bytes = std::fs::read(&installed).unwrap();
        assert_eq!(actual_bytes, expected_bytes, "başlık içeriği farklı: {hdr}");
    }

    // 2b. `dav1d.pc` içeriği ve (varsa) `pkg-config --modversion --cflags --libs dav1d`
    let prefix_lib = prefix_dir.join("lib");
    let pc_dir = prefix_lib.join("pkgconfig");
    let pc_path = pc_dir.join("dav1d.pc");
    let pc_text = std::fs::read_to_string(&pc_path).expect("dav1d.pc okunmalı");
    assert!(pc_text.contains("Version: 1.5.4"));
    assert!(pc_text.contains("Libs: -L${libdir} -ldav1d"));
    assert!(pc_text.contains("Cflags: -I${includedir}"));
    if let Ok(pc_ver) = Command::new("pkg-config")
        .env("PKG_CONFIG_PATH", &pc_dir)
        .args(["--modversion", "dav1d"])
        .output()
    {
        assert!(
            pc_ver.status.success(),
            "pkg-config --modversion dav1d başarısız"
        );
        assert_eq!(String::from_utf8_lossy(&pc_ver.stdout).trim(), "1.5.4");
        let pc_cflags = Command::new("pkg-config")
            .env("PKG_CONFIG_PATH", &pc_dir)
            .args(["--cflags", "dav1d"])
            .output()
            .expect("pkg-config --cflags çalışmalı");
        assert!(pc_cflags.status.success());
        assert!(
            String::from_utf8_lossy(&pc_cflags.stdout)
                .contains(&prefix_inc_root.to_string_lossy().into_owned()),
            "pkg-config --cflags önek include yolunu içermeli"
        );
        let pc_libs = Command::new("pkg-config")
            .env("PKG_CONFIG_PATH", &pc_dir)
            .args(["--libs", "dav1d"])
            .output()
            .expect("pkg-config --libs çalışmalı");
        assert!(pc_libs.status.success());
        assert!(
            String::from_utf8_lossy(&pc_libs.stdout).contains("-ldav1d"),
            "pkg-config --libs -ldav1d içermeli"
        );
    }

    let staticlib_path = prefix_lib.join("libdav1d.a");
    assert!(staticlib_path.is_file(), "libdav1d.a üretilmeli");

    // 2c. Gerçek paylaşımlı kütüphane, SONAME / install_name ve zorunlu `nm` 19 tam sembol denetimi
    #[cfg(target_os = "macos")]
    let (cdylib_path, nm_args): (PathBuf, &[&str]) =
        (prefix_lib.join("libdav1d.7.dylib"), &["-gU"]);
    #[cfg(all(unix, not(target_os = "macos")))]
    let (cdylib_path, nm_args): (PathBuf, &[&str]) =
        (prefix_lib.join("libdav1d.so.7"), &["-D", "--defined-only"]);

    #[cfg(target_os = "macos")]
    {
        assert!(cdylib_path.is_file(), "libdav1d.7.dylib üretilmeli");
        assert!(
            prefix_lib.join("libdav1d.dylib").exists(),
            "libdav1d.dylib bağı mevcut olmalı"
        );
        let otool_out = Command::new("otool")
            .arg("-D")
            .arg(&cdylib_path)
            .output()
            .expect("macOS üzerinde otool -D çalıştırılmalı");
        assert!(otool_out.status.success(), "otool -D başarısız");
        let install_name = String::from_utf8_lossy(&otool_out.stdout)
            .lines()
            .last()
            .unwrap_or("")
            .trim()
            .to_string();
        assert_eq!(
            install_name, "@rpath/libdav1d.7.dylib",
            "Mach-O install_name beklenenden farklı"
        );
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        assert!(
            prefix_lib.join("libdav1d.so.7.0.0").is_file(),
            "libdav1d.so.7.0.0 üretilmeli"
        );
        assert!(cdylib_path.exists(), "libdav1d.so.7 bağı mevcut olmalı");
        assert!(
            prefix_lib.join("libdav1d.so").exists(),
            "libdav1d.so bağı mevcut olmalı"
        );
        let readelf_out = Command::new("readelf")
            .arg("-d")
            .arg(&cdylib_path)
            .output()
            .expect("Linux üzerinde readelf -d çalıştırılmalı");
        assert!(readelf_out.status.success(), "readelf -d başarısız");
        let readelf_text = String::from_utf8_lossy(&readelf_out.stdout);
        assert!(
            readelf_text
                .lines()
                .any(|l| l.contains("(SONAME)") && l.contains("[libdav1d.so.7]")),
            "ELF DT_SONAME [libdav1d.so.7] bulunamadı:\n{readelf_text}"
        );
    }

    #[cfg(unix)]
    {
        let nm_out = Command::new("nm")
            .args(nm_args)
            .arg(&cdylib_path)
            .output()
            .expect("Unix hedefinde nm çalıştırılması zorunludur");
        assert!(
            nm_out.status.success(),
            "nm çalıştırması başarısız: {}",
            String::from_utf8_lossy(&nm_out.stderr)
        );
        let nm_text = String::from_utf8_lossy(&nm_out.stdout);
        let mut exported_syms: HashSet<String> = HashSet::new();
        for line in nm_text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            let (sym_type, raw_name) = if parts.len() >= 3 {
                (parts[1], parts[2])
            } else {
                (parts[0], parts[1])
            };
            if sym_type == "U" || sym_type == "u" {
                continue;
            }
            let base_name = raw_name.split('@').next().unwrap_or(raw_name);
            let normalized = if cfg!(target_os = "macos") {
                base_name.strip_prefix('_').unwrap_or(base_name)
            } else {
                base_name
            };
            exported_syms.insert(normalized.to_string());
        }

        let expected_syms = [
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
        ];
        for sym in expected_syms {
            assert!(
                exported_syms.contains(sym),
                "{} içinde `{sym}` sembolü tam ad eşleşmesiyle dışa aktarılmalı",
                cdylib_path.display()
            );
        }
        let exported_dav1d: HashSet<&str> = exported_syms
            .iter()
            .filter(|s| s.starts_with("dav1d_"))
            .map(String::as_str)
            .collect();
        let expected_set: HashSet<&str> = expected_syms.into_iter().collect();
        assert_eq!(
            exported_dav1d, expected_set,
            "dışa aktarılan dav1d_* sembol kümesi tam olarak 19 resmî sembolle eşleşmeli"
        );
    }

    let c_app_src = r#"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dav1d/dav1d.h>

static const uint8_t PKT0[] = {
    0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
    0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
    0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96
};

typedef struct {
    int alloc_calls;
    int release_calls;
} CustomAllocState;

typedef struct {
    int log_calls;
    char last_msg[512];
} LoggerState;

static void test_c_logger(void *cookie, const char *fmt, va_list ap) {
    LoggerState *st = (LoggerState *)cookie;
    assert(st != NULL && fmt != NULL);
    st->log_calls += 1;
    int n = vsnprintf(st->last_msg, sizeof(st->last_msg), fmt, ap);
    assert(n > 0);
}

static int test_c_alloc_picture(Dav1dPicture *p, void *cookie) {
    CustomAllocState *st = (CustomAllocState *)cookie;
    assert(st != NULL && p != NULL);
    /* FFmpeg libdav1d_picture_allocator gibi alloc sırasında seq_hdr/frame_hdr/m okunur */
    assert(p->seq_hdr != NULL);
    assert(p->frame_hdr != NULL);
    assert(p->seq_hdr->hbd == 0);
    assert(p->frame_hdr->width[1] == 1);
    assert(p->p.w == 1 && p->p.h == 1 && p->p.bpc == 8);
    assert(p->m.timestamp == 777);

    uint8_t *buf = (uint8_t *)calloc(3, 64);
    assert(buf != NULL);
    p->data[0] = buf;
    p->data[1] = buf + 64;
    p->data[2] = buf + 128;
    p->stride[0] = 64;
    p->stride[1] = 64;
    p->allocator_data = buf;
    st->alloc_calls += 1;
    return 0;
}

static void test_c_release_picture(Dav1dPicture *p, void *cookie) {
    CustomAllocState *st = (CustomAllocState *)cookie;
    assert(st != NULL && p != NULL && p->allocator_data != NULL);
    free(p->allocator_data);
    st->release_calls += 1;
}

int main(void) {
    assert(dav1d_version_api() == 0x00070000U);
    assert(strcmp(dav1d_version(), "0.5.1") == 0);

    Dav1dSequenceHeader seq = { 0 };
    assert(dav1d_parse_sequence_header(&seq, PKT0, sizeof(PKT0)) == 0);
    assert(seq.max_width == 1 && seq.max_height == 1);

    /* 1. Varsayılan ayırıcı + 64-bayt hizalama kontrolü */
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

    /* 2. FFmpeg tarzı özel ayırıcı (alloc_cb içinde seq_hdr/frame_hdr/m) + dav1d_apply_grain + C va_list logger */
    CustomAllocState alloc_st = { 0, 0 };
    LoggerState log_st = { 0, { 0 } };
    Dav1dSettings sc;
    dav1d_default_settings(&sc);
    sc.n_threads = 1;
    sc.max_frame_delay = 1;
    sc.apply_grain = 0;
    sc.allocator.cookie = &alloc_st;
    sc.allocator.alloc_picture_callback = test_c_alloc_picture;
    sc.allocator.release_picture_callback = test_c_release_picture;
    sc.logger.cookie = &log_st;
    sc.logger.callback = test_c_logger;

    Dav1dContext *cc = NULL;
    assert(dav1d_open(&cc, &sc) == 0);
    assert(cc != NULL);

    Dav1dData d2 = { 0 };
    uint8_t *b2 = dav1d_data_create(&d2, sizeof(PKT0));
    assert(b2 != NULL);
    memcpy(b2, PKT0, sizeof(PKT0));
    d2.m.timestamp = 777;
    assert(dav1d_send_data(cc, &d2) == 0);

    Dav1dPicture pic2 = { 0 };
    do {
        rc = dav1d_get_picture(cc, &pic2);
    } while (rc == DAV1D_ERR(EAGAIN));
    assert(rc == 0);
    assert(alloc_st.alloc_calls == 1);
    assert(((const uint8_t *)pic2.data[0])[0] == 190U);

    /* Film grain uygulamasında da alloc_cb seq_hdr/frame_hdr görmeli */
    Dav1dFrameHeader fg_hdr = *pic2.frame_hdr;
    fg_hdr.film_grain.present = 1;
    fg_hdr.film_grain.data.seed = 42;
    fg_hdr.film_grain.data.num_y_points = 2;
    fg_hdr.film_grain.data.y_points[0][0] = 0;
    fg_hdr.film_grain.data.y_points[0][1] = 16;
    fg_hdr.film_grain.data.y_points[1][0] = 255;
    fg_hdr.film_grain.data.y_points[1][1] = 16;
    fg_hdr.film_grain.data.scaling_shift = 8;
    fg_hdr.film_grain.data.ar_coeff_shift = 6;
    Dav1dPicture grain_in = pic2;
    grain_in.frame_hdr = &fg_hdr;
    Dav1dPicture grain_out = { 0 };
    assert(dav1d_apply_grain(cc, &grain_out, &grain_in) == 0);
    assert(alloc_st.alloc_calls == 2);
    assert(grain_out.seq_hdr != NULL && grain_out.frame_hdr != NULL);
    dav1d_picture_unref(&grain_out);
    dav1d_picture_unref(&pic2);
    assert(alloc_st.release_calls == 2);

    /* Bozuk paketle gerçek C va_list vsnprintf logger geri çağrısı */
    const uint8_t bad_obu[] = { 0x0A, 0x7F, 0xFF };
    Dav1dData dbad = { 0 };
    uint8_t *bb = dav1d_data_create(&dbad, sizeof(bad_obu));
    assert(bb != NULL);
    memcpy(bb, bad_obu, sizeof(bad_obu));
    assert(dav1d_send_data(cc, &dbad) < 0);
    dav1d_data_unref(&dbad);
    assert(log_st.log_calls >= 1);
    assert(strlen(log_st.last_msg) > 0);

    dav1d_close(&cc);
    assert(cc == NULL);
    return 0;
}
"#;
    let c_app_path = scratch_dir.join("c_app.c");
    let c_app_static_bin = scratch_dir.join("c_app_static");
    let c_app_dynamic_bin = scratch_dir.join("c_app_dynamic");
    std::fs::write(&c_app_path, c_app_src).unwrap();

    // 2d. Statik (`libdav1d.a`) bağlama ve çalıştırma
    let link_static = Command::new("cc")
        .arg("-I")
        .arg(&prefix_inc_root)
        .arg(&c_app_path)
        .arg(&staticlib_path)
        .arg("-lpthread")
        .arg("-ldl")
        .arg("-lm")
        .arg("-o")
        .arg(&c_app_static_bin)
        .status()
        .expect("cc c_app.c + libdav1d.a bağlanmalı");
    assert!(
        link_static.success(),
        "C programı libdav1d.a ile bağlanamadı"
    );
    let run_static = Command::new(&c_app_static_bin).status().unwrap();
    assert!(
        run_static.success(),
        "statik bağlı C programı çalışma zamanı doğrulaması başarısız"
    );

    // 2e. Dinamik (`-L<prefix>/lib -ldav1d -Wl,-rpath,<prefix>/lib`) bağlama ve çalıştırma
    #[cfg(unix)]
    {
        let rpath_arg = format!("-Wl,-rpath,{}", prefix_lib.display());
        let link_dynamic = Command::new("cc")
            .arg("-I")
            .arg(&prefix_inc_root)
            .arg(&c_app_path)
            .arg("-L")
            .arg(&prefix_lib)
            .arg("-ldav1d")
            .arg(&rpath_arg)
            .arg("-o")
            .arg(&c_app_dynamic_bin)
            .status()
            .expect("cc c_app.c + -ldav1d dinamik bağlanmalı");
        assert!(
            link_dynamic.success(),
            "C programı -ldav1d ile dinamik bağlanamadı"
        );
        let run_dynamic = Command::new(&c_app_dynamic_bin).status().unwrap();
        assert!(
            run_dynamic.success(),
            "dinamik bağlı C programı çalışma zamanı doğrulaması başarısız"
        );
    }

    let _ = std::fs::remove_dir_all(&scratch_dir);
}

#[test]
fn capi_apply_grain_in_place_out_equals_in_has_no_uaf_or_leak() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    extern "C" fn count_user_free(_ptr: *const u8, cookie: *mut core::ffi::c_void) {
        // SAFETY: `cookie` bu test boyunca canlı duran `AtomicUsize` örneğinin adresidir.
        let counter = unsafe { &*(cookie as *const AtomicUsize) };
        counter.fetch_add(1, Ordering::SeqCst);
    }

    // `PKT0` önüne CLL, MDCV ve ITU-T T.35 metadata OBU'ları ekleyerek 7 `Dav1dRef` yuvasının
    // tamamının (pic, seq_hdr, frame_hdr, cll, mdcv, itut_t35, user_data) strong_count == 1 ile
    // doldurulmasını sağla.
    fn make_meta_obu(payload: &[u8]) -> Vec<u8> {
        // OBU_METADATA = 5, has_size_field = 1 -> header byte = (5 << 3) | 2 = 0x2A
        let mut out = Vec::with_capacity(2 + payload.len());
        out.push(0x2A);
        assert!(payload.len() < 128);
        out.push(payload.len() as u8);
        out.extend_from_slice(payload);
        out
    }

    let mut cll_payload = vec![1u8, 0x03, 0xE8, 0x01, 0x90, 0x80];
    let _ = &mut cll_payload;
    let mut mdcv_payload = vec![2u8];
    mdcv_payload.extend_from_slice(&[0x12u8; 24]);
    mdcv_payload.push(0x80);
    let itut_payload = vec![4u8, 0xB5, 0x00, 0x31, 0x41, 0x80];

    const PKT0_1X1: &[u8] = &[
        0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
        0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
        0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ];
    let mut rich_pkt = Vec::new();
    rich_pkt.extend_from_slice(&make_meta_obu(&cll_payload));
    rich_pkt.extend_from_slice(&make_meta_obu(&mdcv_payload));
    rich_pkt.extend_from_slice(&make_meta_obu(&itut_payload));
    rich_pkt.extend_from_slice(PKT0_1X1);

    let free_counter = AtomicUsize::new(0);
    let user_sentinel: [u8; 4] = [1, 2, 3, 4];

    // SAFETY: C-ABI çağrıları bu testte ayrılan geçerli `Dav1dContext`, `Dav1dData` ve `Dav1dPicture` yapılarıyla yapılır.
    unsafe {
        let mut settings = Dav1dSettings::default();
        dav1d_default_settings(&mut settings);
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        settings.apply_grain = 0;

        let mut ctx: *mut Dav1dContext = core::ptr::null_mut();
        assert_eq!(dav1d_open(&mut ctx, &settings), 0);

        // 1. `!has_grain_c` kolu: `out == in_pic` (strong_count == 1).
        let mut data = Dav1dData::default();
        let buf = dav1d_data_create(&mut data, rich_pkt.len());
        assert!(!buf.is_null());
        core::ptr::copy_nonoverlapping(rich_pkt.as_ptr(), buf, rich_pkt.len());
        assert_eq!(
            dav1d_data_wrap_user_data(
                &mut data,
                user_sentinel.as_ptr(),
                Some(count_user_free),
                (&free_counter as *const AtomicUsize) as *mut core::ffi::c_void,
            ),
            0
        );
        assert_eq!(dav1d_send_data(ctx, &mut data), 0);

        let mut pic = Dav1dPicture::default();
        assert_eq!(dav1d_get_picture(ctx, &mut pic), 0);
        assert!(!pic.seq_hdr.is_null());
        assert!(!pic.frame_hdr.is_null());
        assert!(!pic.content_light.is_null());
        assert!(!pic.mastering_display.is_null());
        assert!(!pic.itut_t35.is_null());
        assert_eq!(pic.n_itut_t35, 1);
        assert_eq!(free_counter.load(Ordering::SeqCst), 0);

        let pic_ptr: *mut Dav1dPicture = &mut pic;
        assert_eq!(dav1d_apply_grain(ctx, pic_ptr, pic_ptr as *const _), 0);
        assert_eq!(free_counter.load(Ordering::SeqCst), 0);
        assert_eq!((*pic.content_light).max_content_light_level, 1000);
        assert_eq!((*pic.itut_t35).country_code, 0xB5);
        assert_eq!(*(pic.data[0] as *const u8), 190);

        // 2. `has_grain_c` kolu: `out == in_pic` (strong_count == 1).
        let fh = &mut *pic.frame_hdr;
        fh.film_grain.present = 1;
        fh.film_grain.data.seed = 7391;
        fh.film_grain.data.num_y_points = 2;
        fh.film_grain.data.y_points[0] = [0, 16];
        fh.film_grain.data.y_points[1] = [255, 16];
        fh.film_grain.data.scaling_shift = 8;
        fh.film_grain.data.grain_scale_shift = 0;

        assert_eq!(dav1d_apply_grain(ctx, pic_ptr, pic_ptr as *const _), 0);
        assert_eq!(free_counter.load(Ordering::SeqCst), 0);
        assert!(!pic.seq_hdr.is_null());
        assert!(!pic.frame_hdr.is_null());
        assert!(!pic.content_light.is_null());
        assert!(!pic.mastering_display.is_null());
        assert!(!pic.itut_t35.is_null());
        assert_eq!((*pic.content_light).max_content_light_level, 1000);
        assert_eq!((*pic.itut_t35).country_code, 0xB5);
        assert_eq!(pic.m.user_data.data, user_sentinel.as_ptr());

        dav1d_picture_unref(&mut pic);
        assert_eq!(
            free_counter.load(Ordering::SeqCst),
            1,
            "tüm referanslar bırakıldığında user_data tam olarak bir kez serbest bırakılmalı"
        );
        dav1d_close(&mut ctx);
    }
}

#[test]
fn capi_send_data_returns_eagain_when_pending_slot_full_and_matches_rust_decoder() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ivf_path = manifest_dir.join(
        "tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_001_vq_aom_ctest_4.2.stripped.ivf",
    );
    let raw = std::fs::read(&ivf_path).expect("IVF fikstürü okunmalı");
    assert!(raw.len() > 32 && &raw[..4] == b"DKIF");
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut pos = 32usize;
    while pos + 12 <= raw.len() {
        let sz = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 12;
        assert!(pos + sz <= raw.len());
        packets.push(raw[pos..pos + sz].to_vec());
        pos += sz;
    }
    assert!(packets.len() >= 4, "en az 4 IVF paketi gerekli");

    // Referans: Rust `Decoder` ile tüm karelerin Y/U/V özetleri.
    let mut ref_dec = av1oxide::Decoder::new(av1oxide::Config {
        n_threads: 1,
        max_frame_delay: 1,
        ..av1oxide::Config::default()
    })
    .unwrap();
    let mut ref_luma_bytes: Vec<Vec<u8>> = Vec::new();
    for pkt in &packets {
        ref_dec.send_packet(&av1oxide::Packet::new(pkt)).unwrap();
        while let Some(frame) = ref_dec.receive_frame().unwrap() {
            let y = frame.plane(av1oxide::frame::PlaneType::Y).unwrap();
            let mut rows = Vec::with_capacity(y.width() * y.height());
            for r in 0..y.height() {
                rows.extend_from_slice(&y.row(r).unwrap()[..y.width()]);
            }
            ref_luma_bytes.push(rows);
        }
    }
    ref_dec.flush();
    while let Some(frame) = ref_dec.receive_frame().unwrap() {
        let y = frame.plane(av1oxide::frame::PlaneType::Y).unwrap();
        let mut rows = Vec::with_capacity(y.width() * y.height());
        for r in 0..y.height() {
            rows.extend_from_slice(&y.row(r).unwrap()[..y.width()]);
        }
        ref_luma_bytes.push(rows);
    }
    assert!(!ref_luma_bytes.is_empty());

    // SAFETY: C-ABI çağrıları bu testte ayrılan geçerli `Dav1dContext`, `Dav1dData` ve `Dav1dPicture` yapılarıyla yapılır.
    unsafe {
        let mut settings = Dav1dSettings::default();
        dav1d_default_settings(&mut settings);
        settings.n_threads = 1;
        settings.max_frame_delay = 1;

        let mut ctx: *mut Dav1dContext = core::ptr::null_mut();
        assert_eq!(dav1d_open(&mut ctx, &settings), 0);

        let make_data = |bytes: &[u8]| -> Dav1dData {
            let mut d = Dav1dData::default();
            let p = dav1d_data_create(&mut d, bytes.len());
            assert!(!p.is_null());
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
            d
        };

        // 1. paket: çözülür ve çıktı hazır olur -> 0 (tüketildi).
        let mut d0 = make_data(&packets[0]);
        assert_eq!(dav1d_send_data(ctx, &mut d0), 0);
        assert!(d0.data.is_null() && d0.sz == 0);

        // 2. paket (`dav1d_get_picture` çağrılmadan): bekleyen giriş yuvasına alınır -> 0 (tüketildi).
        let mut d1 = make_data(&packets[1]);
        assert_eq!(dav1d_send_data(ctx, &mut d1), 0);
        assert!(d1.data.is_null() && d1.sz == 0);

        // 3. paket (çıktı hâlâ alınmadığı ve bekleyen yuva dolu olduğu için): `-EAGAIN` ve
        // `d2` çağıranda bozulmadan kalır!
        let mut d2 = make_data(&packets[2]);
        let orig_ptr = d2.data;
        let orig_sz = d2.sz;
        let orig_ref = d2.r#ref;
        assert_eq!(dav1d_send_data(ctx, &mut d2), dav1d_err(EAGAIN));
        assert_eq!(d2.data, orig_ptr);
        assert_eq!(d2.sz, orig_sz);
        assert_eq!(d2.r#ref, orig_ref);

        // Şimdi standart `dav1d` send/get döngüsüyle `d2` ve kalan paketleri tamamla.
        let mut capi_luma_bytes: Vec<Vec<u8>> = Vec::new();
        let drain_one = |ctx: *mut Dav1dContext, out_list: &mut Vec<Vec<u8>>| -> i32 {
            let mut pic = Dav1dPicture::default();
            let rc = dav1d_get_picture(ctx, &mut pic);
            if rc == 0 {
                let w = pic.p.w as usize;
                let h = pic.p.h as usize;
                let stride = pic.stride[0] as usize;
                let base = pic.data[0] as *const u8;
                let mut rows = Vec::with_capacity(w * h);
                for r in 0..h {
                    let slice = core::slice::from_raw_parts(base.add(r * stride), w);
                    rows.extend_from_slice(slice);
                }
                out_list.push(rows);
                dav1d_picture_unref(&mut pic);
            }
            rc
        };

        let mut remaining: Vec<Dav1dData> = Vec::new();
        remaining.push(d2);
        for pkt in &packets[3..] {
            remaining.push(make_data(pkt));
        }

        for mut d in remaining {
            loop {
                let s_rc = dav1d_send_data(ctx, &mut d);
                if s_rc == 0 {
                    break;
                }
                assert_eq!(s_rc, dav1d_err(EAGAIN));
                let g_rc = drain_one(ctx, &mut capi_luma_bytes);
                assert_eq!(
                    g_rc, 0,
                    "EAGAIN sonrası dav1d_get_picture kare teslim etmeli"
                );
            }
        }
        loop {
            let g_rc = drain_one(ctx, &mut capi_luma_bytes);
            if g_rc == dav1d_err(EAGAIN) {
                break;
            }
            assert_eq!(g_rc, 0);
        }

        assert_eq!(
            capi_luma_bytes, ref_luma_bytes,
            "C-ABI send/get -EAGAIN döngüsü Rust Decoder ile piksel-özdeş olmalı"
        );
        dav1d_close(&mut ctx);
    }
}

#[test]
fn capi_inloop_filters_decode_frame_type_and_logger_callback() {
    use av1oxide::capi::types::{
        DAV1D_DECODEFRAMETYPE_ALL, DAV1D_DECODEFRAMETYPE_INTRA, DAV1D_DECODEFRAMETYPE_KEY,
        DAV1D_DECODEFRAMETYPE_REFERENCE, DAV1D_INLOOPFILTER_ALL, DAV1D_INLOOPFILTER_CDEF,
        DAV1D_INLOOPFILTER_DEBLOCK, DAV1D_INLOOPFILTER_NONE, DAV1D_INLOOPFILTER_RESTORATION,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    unsafe extern "C" fn test_logger_cb(
        cookie: *mut core::ffi::c_void,
        fmt: *const core::ffi::c_char,
        _ap: core::ffi::VaList<'_>,
    ) {
        if !cookie.is_null() && !fmt.is_null() {
            // SAFETY: `cookie` bu test boyunca canlı `AtomicUsize` örneğidir.
            let hits = unsafe { &*(cookie as *const AtomicUsize) };
            hits.fetch_add(1, Ordering::SeqCst);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ivf_path = manifest_dir.join(
        "tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_001_vq_aom_ctest_4.2.stripped.ivf",
    );
    let raw = std::fs::read(&ivf_path).expect("IVF fikstürü okunmalı");
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut pos = 32usize;
    while pos + 12 <= raw.len() {
        let sz = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 12;
        packets.push(raw[pos..pos + sz].to_vec());
        pos += sz;
    }

    let log_hits = AtomicUsize::new(0);

    // SAFETY: C-ABI çağrıları bu test içinde ayrılan geçerli yapılarla yapılır.
    unsafe {
        // 1. Geçersiz `inloop_filters` ve `decode_frame_type` reddedilir ve logger çağrılır.
        let mut bad_s = Dav1dSettings::default();
        dav1d_default_settings(&mut bad_s);
        bad_s.logger.cookie = (&log_hits as *const AtomicUsize) as *mut core::ffi::c_void;
        bad_s.logger.callback = test_logger_cb as *const core::ffi::c_void;
        bad_s.inloop_filters = DAV1D_INLOOPFILTER_ALL | 0x8;
        let mut ctx: *mut Dav1dContext = core::ptr::null_mut();
        assert_eq!(dav1d_open(&mut ctx, &bad_s), dav1d_err(EINVAL));
        assert!(ctx.is_null());
        assert!(log_hits.load(Ordering::SeqCst) >= 1);

        bad_s.inloop_filters = DAV1D_INLOOPFILTER_ALL;
        bad_s.decode_frame_type = DAV1D_DECODEFRAMETYPE_KEY + 1;
        assert_eq!(dav1d_open(&mut ctx, &bad_s), dav1d_err(EINVAL));
        assert!(ctx.is_null());

        // 2. Tüm geçerli `inloop_filters` kombinasyonları (`NONE`, `DEBLOCK`, `CDEF`, `RESTORATION`,
        //    `DEBLOCK|RESTORATION`, `ALL`) kabul edilip çalışır.
        let mut decoded_none = Vec::new();
        let mut decoded_all = Vec::new();
        for mask in [
            DAV1D_INLOOPFILTER_NONE,
            DAV1D_INLOOPFILTER_DEBLOCK,
            DAV1D_INLOOPFILTER_CDEF,
            DAV1D_INLOOPFILTER_RESTORATION,
            DAV1D_INLOOPFILTER_DEBLOCK | DAV1D_INLOOPFILTER_RESTORATION,
            DAV1D_INLOOPFILTER_ALL,
        ] {
            let mut s = Dav1dSettings::default();
            dav1d_default_settings(&mut s);
            s.n_threads = 1;
            s.max_frame_delay = 1;
            s.inloop_filters = mask;
            let mut c: *mut Dav1dContext = core::ptr::null_mut();
            assert_eq!(dav1d_open(&mut c, &s), 0, "mask={mask} açılmalı");
            let mut d = Dav1dData::default();
            let p = dav1d_data_create(&mut d, packets[0].len());
            core::ptr::copy_nonoverlapping(packets[0].as_ptr(), p, packets[0].len());
            assert_eq!(dav1d_send_data(c, &mut d), 0);
            let mut pic = Dav1dPicture::default();
            assert_eq!(dav1d_get_picture(c, &mut pic), 0);
            if mask == DAV1D_INLOOPFILTER_NONE || mask == DAV1D_INLOOPFILTER_ALL {
                let w = pic.p.w as usize;
                let h = pic.p.h as usize;
                let stride = pic.stride[0] as usize;
                let base = pic.data[0] as *const u8;
                let mut rows = Vec::with_capacity(w * h);
                for r in 0..h {
                    rows.extend_from_slice(core::slice::from_raw_parts(base.add(r * stride), w));
                }
                if mask == DAV1D_INLOOPFILTER_NONE {
                    decoded_none = rows;
                } else {
                    decoded_all = rows;
                }
            }
            dav1d_picture_unref(&mut pic);
            dav1d_close(&mut c);
        }
        assert!(!decoded_none.is_empty() && !decoded_all.is_empty());
        assert_ne!(
            decoded_none, decoded_all,
            "inloop_filters=NONE çıktısı ALL çıktısından farklı olmalıdır"
        );

        // 3. `decode_frame_type = KEY`: çok kareli IVF'den yalnızca anahtar kareler çıkar.
        let count_for_ft = |ft: u32| -> usize {
            let mut s = Dav1dSettings::default();
            dav1d_default_settings(&mut s);
            s.n_threads = 1;
            s.max_frame_delay = 1;
            s.decode_frame_type = ft;
            let mut c: *mut Dav1dContext = core::ptr::null_mut();
            assert_eq!(dav1d_open(&mut c, &s), 0);
            let mut count = 0usize;
            for pkt in &packets {
                let mut d = Dav1dData::default();
                let p = dav1d_data_create(&mut d, pkt.len());
                core::ptr::copy_nonoverlapping(pkt.as_ptr(), p, pkt.len());
                assert_eq!(dav1d_send_data(c, &mut d), 0);
                let mut pic = Dav1dPicture::default();
                while dav1d_get_picture(c, &mut pic) == 0 {
                    count += 1;
                    dav1d_picture_unref(&mut pic);
                }
            }
            let mut pic = Dav1dPicture::default();
            while dav1d_get_picture(c, &mut pic) == 0 {
                count += 1;
                dav1d_picture_unref(&mut pic);
            }
            dav1d_close(&mut c);
            count
        };
        let all_cnt = count_for_ft(DAV1D_DECODEFRAMETYPE_ALL);
        let ref_cnt = count_for_ft(DAV1D_DECODEFRAMETYPE_REFERENCE);
        let intra_cnt = count_for_ft(DAV1D_DECODEFRAMETYPE_INTRA);
        let key_cnt = count_for_ft(DAV1D_DECODEFRAMETYPE_KEY);
        assert!(all_cnt >= 5);
        assert!(ref_cnt <= all_cnt);
        assert!(intra_cnt >= key_cnt && key_cnt >= 1 && key_cnt < all_cnt);

        // 4. `logger` bozuk paket hatasında çağrılır.
        let before = log_hits.load(Ordering::SeqCst);
        let mut s = Dav1dSettings::default();
        dav1d_default_settings(&mut s);
        s.n_threads = 1;
        s.max_frame_delay = 1;
        s.logger.cookie = (&log_hits as *const AtomicUsize) as *mut core::ffi::c_void;
        s.logger.callback = test_logger_cb as *const core::ffi::c_void;
        let mut c: *mut Dav1dContext = core::ptr::null_mut();
        assert_eq!(dav1d_open(&mut c, &s), 0);
        let bad_obu = [0x0Au8, 0x7F, 0xFF]; // kesik OBU
        let mut d = Dav1dData::default();
        let p = dav1d_data_create(&mut d, bad_obu.len());
        core::ptr::copy_nonoverlapping(bad_obu.as_ptr(), p, bad_obu.len());
        assert!(dav1d_send_data(c, &mut d) < 0);
        dav1d_data_unref(&mut d);
        assert!(log_hits.load(Ordering::SeqCst) > before);
        dav1d_close(&mut c);
    }
}
