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
    /* `grain_out` hemen serbest kalır; `pic2` KEY referans yuvasında tutulduğu için `dav1d_close` ile serbest kalır. */
    assert(alloc_st.release_calls == 1);

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
    assert(alloc_st.release_calls == 2);
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
    use std::sync::Mutex;

    unsafe extern "C" {
        fn vsnprintf(
            s: *mut core::ffi::c_char,
            n: usize,
            format: *const core::ffi::c_char,
            ap: core::ffi::VaList<'_>,
        ) -> core::ffi::c_int;
    }

    unsafe extern "C" fn test_logger_cb(
        cookie: *mut core::ffi::c_void,
        fmt: *const core::ffi::c_char,
        ap: core::ffi::VaList<'_>,
    ) {
        if !cookie.is_null() && !fmt.is_null() {
            let mut buf = [0u8; 512];
            // SAFETY: `buf` 512 bayt yazılabilir yerel tampondur; `fmt` ve `ap` C variadic
            // `Dav1dLogger.callback` çağrısından geçerli olarak gelir; `cookie` canlı `Mutex<Vec<String>>` adresidir.
            unsafe {
                let n = vsnprintf(
                    buf.as_mut_ptr() as *mut core::ffi::c_char,
                    buf.len(),
                    fmt,
                    ap,
                );
                assert!(n > 0, "vsnprintf geçerli günlük mesajı biçimlendirmeli");
                let text = CStr::from_ptr(buf.as_ptr() as *const core::ffi::c_char)
                    .to_string_lossy()
                    .into_owned();
                let logs = &*(cookie as *const Mutex<Vec<String>>);
                logs.lock().unwrap().push(text);
            }
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

    let log_messages: Mutex<Vec<String>> = Mutex::new(Vec::new());

    // SAFETY: C-ABI çağrıları bu test içinde ayrılan geçerli yapılarla yapılır.
    unsafe {
        // 1. Geçersiz `inloop_filters` ve `decode_frame_type` reddedilir ve logger `vsnprintf` ile gerçek mesaj biçimlendirir.
        let mut bad_s = Dav1dSettings::default();
        dav1d_default_settings(&mut bad_s);
        bad_s.logger.cookie =
            (&log_messages as *const Mutex<Vec<String>>) as *mut core::ffi::c_void;
        bad_s.logger.callback = test_logger_cb as *const core::ffi::c_void;
        bad_s.inloop_filters = DAV1D_INLOOPFILTER_ALL | 0x8;
        let mut ctx: *mut Dav1dContext = core::ptr::null_mut();
        assert_eq!(dav1d_open(&mut ctx, &bad_s), dav1d_err(EINVAL));
        assert!(ctx.is_null());
        {
            let msgs = log_messages.lock().unwrap();
            assert!(!msgs.is_empty(), "logger en az bir mesaj kaydetmeli");
            assert!(
                msgs[0].ends_with('\n') && msgs[0].contains("invalid settings"),
                "biçimlendirilen mesaj beklenenden farklı: {:?}",
                msgs[0]
            );
        }

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
        let count_for_ft = |ft: u32, n_threads: c_int, mfd: c_int| -> usize {
            let mut s = Dav1dSettings::default();
            dav1d_default_settings(&mut s);
            s.n_threads = n_threads;
            s.max_frame_delay = mfd;
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
        for (nt, mfd) in [(1, 1), (4, 2)] {
            let all_cnt = count_for_ft(DAV1D_DECODEFRAMETYPE_ALL, nt, mfd);
            let ref_cnt = count_for_ft(DAV1D_DECODEFRAMETYPE_REFERENCE, nt, mfd);
            let intra_cnt = count_for_ft(DAV1D_DECODEFRAMETYPE_INTRA, nt, mfd);
            let key_cnt = count_for_ft(DAV1D_DECODEFRAMETYPE_KEY, nt, mfd);
            assert!(all_cnt >= 5);
            assert!(ref_cnt <= all_cnt);
            assert!(intra_cnt >= key_cnt && key_cnt >= 1 && key_cnt < all_cnt);
        }

        // 4. `logger` bozuk OBU başlık/boyut hatasında C `dav1d` (`src/obu.c`) ile birebir aynı
        //    İngilizce `"Error parsing OBU data\n"` mesajını üretir; karo/MSAC çözme hatasında ise
        //    C `dav1d` gibi hiç günlük yazmaz (`0` mesaj).
        let before = log_messages.lock().unwrap().len();
        let mut s = Dav1dSettings::default();
        dav1d_default_settings(&mut s);
        s.n_threads = 1;
        s.max_frame_delay = 1;
        s.logger.cookie = (&log_messages as *const Mutex<Vec<String>>) as *mut core::ffi::c_void;
        s.logger.callback = test_logger_cb as *const core::ffi::c_void;
        let mut c: *mut Dav1dContext = core::ptr::null_mut();
        assert_eq!(dav1d_open(&mut c, &s), 0);
        let bad_obu = [0x0Au8, 0x7F, 0xFF]; // kesik OBU
        let mut d = Dav1dData::default();
        let p = dav1d_data_create(&mut d, bad_obu.len());
        core::ptr::copy_nonoverlapping(bad_obu.as_ptr(), p, bad_obu.len());
        assert!(dav1d_send_data(c, &mut d) < 0);
        dav1d_data_unref(&mut d);
        {
            let msgs = log_messages.lock().unwrap();
            assert_eq!(msgs.len(), before + 1);
            let last = msgs.last().unwrap();
            assert_eq!(
                last, "Error parsing OBU data\n",
                "bozuk paket log mesajı C dav1d ile birebir aynı olmalı: {last:?}"
            );
        }

        // Karo/MSAC bozuk kare paketi: geçerli TD + Sequence Header + Frame Header korunup
        // karo verisi 1 bayt `0x00`'a kısaltılır -> başlık ayrışır, karo çözümü hata verir,
        // ancak `Dav1dLogger`'a yeni mesaj yazılmaz.
        let mut corrupt_tile_pkt = Vec::new();
        {
            let pkt0 = &packets[0];
            let seq = av1oxide::bitstream::parse_sequence_header_obus(pkt0)
                .unwrap()
                .unwrap();
            let mut off = 0usize;
            while off < pkt0.len() {
                let start = off;
                let Some(obu) = av1oxide::bitstream::split_next_with(pkt0, &mut off, true).unwrap()
                else {
                    break;
                };
                if obu.header.obu_type == av1oxide::bitstream::ObuType::Frame {
                    let mut br = av1oxide::bitstream::BitReader::new(obu.payload);
                    let refs0: [Option<av1oxide::bitstream::FrameHeader>; 8] = Default::default();
                    let ctx_fh = av1oxide::bitstream::FrameHeaderContext {
                        sequence_header: &seq,
                        refs: &refs0,
                        temporal_id: 0,
                        spatial_id: 0,
                        strict_std_compliance: false,
                    };
                    let fh = av1oxide::bitstream::FrameHeader::parse(&mut br, &ctx_fh).unwrap();
                    br.align_to_byte();
                    let fh_end = br.byte_position();
                    let n_tiles = usize::from(fh.tiling.cols) * usize::from(fh.tiling.rows);
                    let size_bytes = usize::from(fh.tiling.n_bytes);
                    let mut body = obu.payload[..fh_end].to_vec();
                    for t in 0..n_tiles {
                        if t + 1 < n_tiles {
                            body.resize(body.len() + size_bytes, 0u8);
                        }
                        body.push(0x00);
                    }
                    corrupt_tile_pkt.push(0x32); // OBU_FRAME, has_size_field=1
                    let mut rem = body.len();
                    loop {
                        let mut b = (rem & 0x7F) as u8;
                        rem >>= 7;
                        if rem != 0 {
                            b |= 0x80;
                        }
                        corrupt_tile_pkt.push(b);
                        if rem == 0 {
                            break;
                        }
                    }
                    corrupt_tile_pkt.extend_from_slice(&body);
                } else {
                    corrupt_tile_pkt.extend_from_slice(&pkt0[start..off]);
                }
            }
        }
        assert!(!corrupt_tile_pkt.is_empty());

        let before_tile_err = log_messages.lock().unwrap().len();
        let mut d_tile = Dav1dData::default();
        let p_tile = dav1d_data_create(&mut d_tile, corrupt_tile_pkt.len());
        core::ptr::copy_nonoverlapping(corrupt_tile_pkt.as_ptr(), p_tile, corrupt_tile_pkt.len());
        let send_rc = dav1d_send_data(c, &mut d_tile);
        if d_tile.sz > 0 {
            dav1d_data_unref(&mut d_tile);
        }
        let mut pic_tile = Dav1dPicture::default();
        let get_rc = dav1d_get_picture(c, &mut pic_tile);
        assert!(
            send_rc < 0 || (get_rc < 0 && get_rc != dav1d_err(EAGAIN)),
            "bozuk karo verisi hata döndürmeli: send_rc={send_rc}, get_rc={get_rc}"
        );
        assert_eq!(
            log_messages.lock().unwrap().len(),
            before_tile_err,
            "karo/MSAC çözme hatasında Dav1dLogger'a mesaj yazılmamalı"
        );
        dav1d_close(&mut c);

        // Referans C `libdav1d.so.7` mevcutsa hem `bad_obu` hem `corrupt_tile_pkt` için
        // üretilen günlük satırlarının birebir aynı olduğunu doğrula.
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            use core::ffi::c_char;
            use std::ffi::CString;
            unsafe extern "C" {
                fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
                fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
                fn dlclose(handle: *mut c_void) -> c_int;
            }
            let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
            let so_path = workspace_root.join(".cache/dav1d-build-asm/src/libdav1d.so.7");
            if so_path.is_file() {
                let cpath = CString::new(so_path.to_str().unwrap()).unwrap();
                let handle = dlopen(cpath.as_ptr(), 2);
                if !handle.is_null() {
                    let sym = |name: &CStr| -> *mut c_void { dlsym(handle, name.as_ptr()) };
                    let f_def = core::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut Dav1dSettings),
                    >(sym(c"dav1d_default_settings"));
                    let f_open = core::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut *mut Dav1dContext, *const Dav1dSettings) -> c_int,
                    >(sym(c"dav1d_open"));
                    let f_close = core::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut *mut Dav1dContext),
                    >(sym(c"dav1d_close"));
                    let f_create = core::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut Dav1dData, usize) -> *mut u8,
                    >(sym(c"dav1d_data_create"));
                    let f_dunref = core::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut Dav1dData),
                    >(sym(c"dav1d_data_unref"));
                    let f_send = core::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dData) -> c_int,
                    >(sym(c"dav1d_send_data"));
                    let f_get = core::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dPicture) -> c_int,
                    >(sym(c"dav1d_get_picture"));

                    let ref_logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
                    let mut rs = Dav1dSettings::default();
                    f_def(&mut rs);
                    rs.n_threads = 1;
                    rs.max_frame_delay = 1;
                    rs.logger.cookie = (&ref_logs as *const Mutex<Vec<String>>) as *mut c_void;
                    rs.logger.callback = test_logger_cb as *const c_void;
                    let mut rc_ctx: *mut Dav1dContext = core::ptr::null_mut();
                    assert_eq!(f_open(&mut rc_ctx, &rs), 0);

                    let mut rd = Dav1dData::default();
                    let rp = f_create(&mut rd, bad_obu.len());
                    core::ptr::copy_nonoverlapping(bad_obu.as_ptr(), rp, bad_obu.len());
                    assert!(f_send(rc_ctx, &mut rd) < 0);
                    if rd.sz > 0 {
                        f_dunref(&mut rd);
                    }
                    assert_eq!(
                        *ref_logs.lock().unwrap(),
                        vec!["Error parsing OBU data\n".to_string()],
                        "referans C libdav1d bad_obu log çıktısı eşleşmeli"
                    );

                    let mut rd_tile = Dav1dData::default();
                    let rp_tile = f_create(&mut rd_tile, corrupt_tile_pkt.len());
                    core::ptr::copy_nonoverlapping(
                        corrupt_tile_pkt.as_ptr(),
                        rp_tile,
                        corrupt_tile_pkt.len(),
                    );
                    let r_send = f_send(rc_ctx, &mut rd_tile);
                    if rd_tile.sz > 0 {
                        f_dunref(&mut rd_tile);
                    }
                    let mut r_pic = Dav1dPicture::default();
                    let r_get = f_get(rc_ctx, &mut r_pic);
                    assert!(r_send < 0 || (r_get < 0 && r_get != dav1d_err(EAGAIN)));
                    assert_eq!(
                        ref_logs.lock().unwrap().len(),
                        1,
                        "referans C libdav1d karo/MSAC çözme hatasında ek log yazmamalı"
                    );

                    f_close(&mut rc_ctx);
                    let _ = dlclose(handle);
                }
            }
        }
    }
}

#[test]
fn capi_custom_allocator_enomem_timing_and_error_props() {
    struct FailAllocState {
        alloc_calls: AtomicUsize,
        release_calls: AtomicUsize,
    }

    extern "C" fn fail_enomem_alloc(_pic: *mut Dav1dPicture, cookie: *mut c_void) -> c_int {
        // SAFETY: `cookie`, test çerçevesinde canlı tutulan `FailAllocState` adresidir.
        let st = unsafe { &*(cookie as *const FailAllocState) };
        st.alloc_calls.fetch_add(1, Ordering::SeqCst);
        dav1d_err(ENOMEM)
    }

    extern "C" fn fail_enomem_release(_pic: *mut Dav1dPicture, cookie: *mut c_void) {
        // SAFETY: `cookie`, test çerçevesinde canlı tutulan `FailAllocState` adresidir.
        let st = unsafe { &*(cookie as *const FailAllocState) };
        st.release_calls.fetch_add(1, Ordering::SeqCst);
    }

    for (n_threads, max_frame_delay) in [(1, 1), (4, 2)] {
        let alloc_st = FailAllocState {
            alloc_calls: AtomicUsize::new(0),
            release_calls: AtomicUsize::new(0),
        };
        let user_frees = AtomicUsize::new(0);
        let user_sentinel = [0xDEu8, 0xAD, 0xBE, 0xEF];

        // SAFETY: Test boyunca tüm yapılar yerel yığında canlıdır ve `dav1d_*_unref`/`dav1d_close` ile temizlenir.
        unsafe {
            let mut s = Dav1dSettings::default();
            dav1d_default_settings(&mut s);
            s.n_threads = n_threads;
            s.max_frame_delay = max_frame_delay;
            s.allocator = Dav1dPicAllocator {
                cookie: (&alloc_st as *const FailAllocState) as *mut c_void,
                alloc_picture_callback: Some(fail_enomem_alloc),
                release_picture_callback: Some(fail_enomem_release),
            };

            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!(dav1d_open(&mut c, &s), 0);

            let mut data = Dav1dData::default();
            let dst = dav1d_data_create(&mut data, SAMPLE_OBU_PACKETS[0].len());
            assert!(!dst.is_null());
            ptr::copy_nonoverlapping(
                SAMPLE_OBU_PACKETS[0].as_ptr(),
                dst,
                SAMPLE_OBU_PACKETS[0].len(),
            );
            assert_eq!(
                dav1d_data_wrap_user_data(
                    &mut data,
                    user_sentinel.as_ptr(),
                    Some(count_free_cb),
                    (&user_frees as *const AtomicUsize) as *mut c_void,
                ),
                0
            );
            data.m.timestamp = 442;
            data.m.offset = 123;

            // Özel ayırıcı kare gönderimi sırasında (`dav1d_send_data` içinde) çalışmalı
            // ve `-ENOMEM` döndürürken çağıranın `Dav1dData` sahipliğini korumalı.
            let rc = dav1d_send_data(c, &mut data);
            assert_eq!(
                rc,
                dav1d_err(ENOMEM),
                "t={n_threads}, mfd={max_frame_delay}: dav1d_send_data -ENOMEM döndürmeli"
            );
            assert!(
                !data.data.is_null(),
                "t={n_threads}, mfd={max_frame_delay}: hata durumunda in_data tüketilmemeli"
            );
            assert_eq!(alloc_st.alloc_calls.load(Ordering::SeqCst), 1);
            assert_eq!(alloc_st.release_calls.load(Ordering::SeqCst), 0);

            dav1d_data_unref(&mut data);
            assert_eq!(
                user_frees.load(Ordering::SeqCst),
                0,
                "cached_error_props user_data referansını hâlâ canlı tutmalı"
            );

            let mut err_props = Dav1dDataProps::default();
            assert_eq!(dav1d_get_decode_error_data_props(c, &mut err_props), 0);
            assert_eq!(err_props.timestamp, 442);
            assert_eq!(err_props.offset, 123);
            assert_eq!(err_props.user_data.data, user_sentinel.as_ptr());

            dav1d_data_props_unref(&mut err_props);
            assert_eq!(user_frees.load(Ordering::SeqCst), 1);
            dav1d_close(&mut c);
        }
    }
}

#[test]
fn capi_decode_frame_type_and_svc_per_obu_parity() {
    use av1oxide::capi::types::{
        DAV1D_DECODEFRAMETYPE_ALL, DAV1D_DECODEFRAMETYPE_INTRA, DAV1D_DECODEFRAMETYPE_KEY,
        DAV1D_DECODEFRAMETYPE_REFERENCE,
    };

    #[derive(Clone, Copy)]
    struct Dav1dApi {
        default_settings: unsafe extern "C" fn(*mut Dav1dSettings),
        open: unsafe extern "C" fn(*mut *mut Dav1dContext, *const Dav1dSettings) -> c_int,
        close: unsafe extern "C" fn(*mut *mut Dav1dContext),
        data_create: unsafe extern "C" fn(*mut Dav1dData, usize) -> *mut u8,
        send_data: unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dData) -> c_int,
        get_picture: unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dPicture) -> c_int,
        picture_unref: unsafe extern "C" fn(*mut Dav1dPicture),
    }

    let av1oxide_api = Dav1dApi {
        default_settings: dav1d_default_settings,
        open: dav1d_open,
        close: dav1d_close,
        data_create: dav1d_data_create,
        send_data: dav1d_send_data,
        get_picture: dav1d_get_picture,
        picture_unref: dav1d_picture_unref,
    };

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    struct RefDav1dLib {
        handle: *mut c_void,
        api: Dav1dApi,
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    impl RefDav1dLib {
        fn try_load(so_path: &std::path::Path) -> Option<Self> {
            use core::ffi::c_char;
            use std::ffi::CString;
            unsafe extern "C" {
                fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
                fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            }
            const RTLD_NOW: c_int = 2;
            const RTLD_LOCAL: c_int = 0;
            if !so_path.is_file() {
                return None;
            }
            let cpath = CString::new(so_path.to_str()?).ok()?;
            // SAFETY: Referans C `libdav1d.so.7` paylaşımlı kütüphanesi salt test karşılaştırması için yüklenir.
            unsafe {
                let handle = dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_LOCAL);
                if handle.is_null() {
                    return None;
                }
                let sym = |name: &CStr| -> *mut c_void { dlsym(handle, name.as_ptr()) };
                let f_def = sym(c"dav1d_default_settings");
                let f_open = sym(c"dav1d_open");
                let f_close = sym(c"dav1d_close");
                let f_create = sym(c"dav1d_data_create");
                let f_send = sym(c"dav1d_send_data");
                let f_get = sym(c"dav1d_get_picture");
                let f_unref = sym(c"dav1d_picture_unref");
                if [f_def, f_open, f_close, f_create, f_send, f_get, f_unref]
                    .iter()
                    .any(|p| p.is_null())
                {
                    return None;
                }
                Some(Self {
                    handle,
                    api: Dav1dApi {
                        default_settings: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dSettings),
                        >(f_def),
                        open: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(
                                *mut *mut Dav1dContext,
                                *const Dav1dSettings,
                            ) -> c_int,
                        >(f_open),
                        close: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut *mut Dav1dContext),
                        >(f_close),
                        data_create: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dData, usize) -> *mut u8,
                        >(f_create),
                        send_data: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dData) -> c_int,
                        >(f_send),
                        get_picture: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dPicture) -> c_int,
                        >(f_get),
                        picture_unref: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dPicture),
                        >(f_unref),
                    },
                })
            }
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    impl Drop for RefDav1dLib {
        fn drop(&mut self) {
            unsafe extern "C" {
                fn dlclose(handle: *mut c_void) -> c_int;
            }
            // SAFETY: `self.handle`, `dlopen` ile açılmış geçerli tanıtıcıdır.
            unsafe {
                let _ = dlclose(self.handle);
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct DecodedCapiFrame {
        planes: Vec<Vec<u8>>,
        timestamp: i64,
        offset: i64,
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_packets_capi(
        api: Dav1dApi,
        packets: &[&[u8]],
        n_threads: c_int,
        mfd: c_int,
        decode_frame_type: u32,
        all_layers: c_int,
        apply_grain: c_int,
        while_drain: bool,
    ) -> Vec<DecodedCapiFrame> {
        // SAFETY: Her çağrı bağımsız bir `Dav1dContext` açıp kapatır ve tamponları güvenle yönetir.
        unsafe {
            let mut s = Dav1dSettings::default();
            (api.default_settings)(&mut s);
            s.n_threads = n_threads;
            s.max_frame_delay = mfd;
            s.decode_frame_type = decode_frame_type;
            s.all_layers = all_layers;
            s.apply_grain = apply_grain;

            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!((api.open)(&mut c, &s), 0);

            let mut out = Vec::new();
            let mut pic = Dav1dPicture::default();
            for (idx, &pkt) in packets.iter().enumerate() {
                let mut data = Dav1dData::default();
                let dst = (api.data_create)(&mut data, pkt.len());
                assert!(!dst.is_null());
                ptr::copy_nonoverlapping(pkt.as_ptr(), dst, pkt.len());
                data.m.timestamp = 100 + idx as i64;
                data.m.offset = 1000 + idx as i64;
                loop {
                    let s_rc = (api.send_data)(c, &mut data);
                    assert!(
                        s_rc == 0 || s_rc == dav1d_err(EAGAIN),
                        "dav1d_send_data beklenmeyen hata: {s_rc}"
                    );
                    if while_drain {
                        loop {
                            let g_rc = (api.get_picture)(c, &mut pic);
                            if g_rc == 0 {
                                out.push(DecodedCapiFrame {
                                    planes: copy_c_picture_planes(&pic),
                                    timestamp: pic.m.timestamp,
                                    offset: pic.m.offset,
                                });
                                (api.picture_unref)(&mut pic);
                            } else {
                                assert_eq!(g_rc, dav1d_err(EAGAIN));
                                break;
                            }
                        }
                    } else {
                        let g_rc = (api.get_picture)(c, &mut pic);
                        if g_rc == 0 {
                            out.push(DecodedCapiFrame {
                                planes: copy_c_picture_planes(&pic),
                                timestamp: pic.m.timestamp,
                                offset: pic.m.offset,
                            });
                            (api.picture_unref)(&mut pic);
                        } else {
                            assert_eq!(g_rc, dav1d_err(EAGAIN));
                        }
                    }
                    if s_rc == 0 {
                        break;
                    }
                }
            }
            // EOS drain döngüsü
            loop {
                let rc = (api.get_picture)(c, &mut pic);
                if rc == dav1d_err(EAGAIN) {
                    break;
                }
                assert_eq!(rc, 0);
                out.push(DecodedCapiFrame {
                    planes: copy_c_picture_planes(&pic),
                    timestamp: pic.m.timestamp,
                    offset: pic.m.offset,
                });
                (api.picture_unref)(&mut pic);
            }
            (api.close)(&mut c);
            out
        }
    }

    fn split_to_single_obu_packets<'a>(ivf_packets: &[(i64, &'a [u8])]) -> Vec<&'a [u8]> {
        let mut out = Vec::new();
        for &(_, payload) in ivf_packets {
            let mut offset = 0usize;
            while offset < payload.len() {
                let start = offset;
                let Some(_) = av1oxide::bitstream::split_next_with(payload, &mut offset, true)
                    .expect("geçerli OBU ayrışmalı")
                else {
                    break;
                };
                out.push(&payload[start..offset]);
            }
        }
        out
    }

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let ref_dav1d =
        RefDav1dLib::try_load(&workspace_root.join(".cache/dav1d-build-asm/src/libdav1d.so.7"));
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    let ref_dav1d: Option<()> = None;

    // 1. `ccvb_film_grain.ivf` ve 10-bit `av1-1-b10-23-film_grain-50.ivf` (varsa):
    //    `decode_frame_type = ALL/REFERENCE/INTRA/KEY` referans `dav1d` kare sayısı eşliği.
    let ft_cases = [
        (
            workspace_root.join(".cache/dav1d-test-data/8-bit/features/ccvb_film_grain.ivf"),
            100usize,
            94usize,
            1usize,
            1usize,
        ),
        (
            workspace_root
                .join(".cache/dav1d-test-data/10-bit/film_grain/av1-1-b10-23-film_grain-50.ivf"),
            10usize,
            10usize,
            1usize,
            1usize,
        ),
    ];
    for (path, exp_all, exp_ref, exp_intra, exp_key) in ft_cases {
        if !path.is_file() {
            continue;
        }
        let raw = std::fs::read(&path).unwrap();
        let ivf = parse_ivf(&raw);
        let pkts: Vec<&[u8]> = ivf.iter().map(|&(_, p)| p).collect();
        for (nt, mfd) in [(1, 1), (4, 2)] {
            let all_frames = decode_packets_capi(
                av1oxide_api,
                &pkts,
                nt,
                mfd,
                DAV1D_DECODEFRAMETYPE_ALL,
                1,
                0,
                false,
            );
            let ref_frames = decode_packets_capi(
                av1oxide_api,
                &pkts,
                nt,
                mfd,
                DAV1D_DECODEFRAMETYPE_REFERENCE,
                1,
                0,
                false,
            );
            let intra_frames = decode_packets_capi(
                av1oxide_api,
                &pkts,
                nt,
                mfd,
                DAV1D_DECODEFRAMETYPE_INTRA,
                1,
                0,
                false,
            );
            let key_frames = decode_packets_capi(
                av1oxide_api,
                &pkts,
                nt,
                mfd,
                DAV1D_DECODEFRAMETYPE_KEY,
                1,
                0,
                false,
            );
            assert_eq!(all_frames.len(), exp_all, "{:?} t={nt}", path.file_name());
            assert_eq!(ref_frames.len(), exp_ref, "{:?} t={nt}", path.file_name());
            assert_eq!(
                intra_frames.len(),
                exp_intra,
                "{:?} t={nt}",
                path.file_name()
            );
            assert_eq!(key_frames.len(), exp_key, "{:?} t={nt}", path.file_name());
            assert_eq!(key_frames[0], all_frames[0]);
            assert_eq!(intra_frames[0], all_frames[0]);
        }
    }

    // 2. SVC akışlarında (`L1T2`, `L2T1`, `L2T2` varsa) IVF ve 1-OBU-başına-paket,
    //    `all_layers = 0` / `1`, `while_drain = false` / `true` ve `(1,1)/(4,1)/(4,2)/(4,4)`
    //    matrisinde gerçek C `libdav1d` ile birebir kare sayısı + piksel düzlemi eşliği.
    let svc_cases = [
        ("av1-1-b8-22-svc-L1T2.ivf", false),
        ("av1-1-b8-22-svc-L2T1.ivf", true),
        ("av1-1-b8-22-svc-L2T2.ivf", true),
    ];
    for (svc_name, has_two_spatial_layers) in svc_cases {
        let svc_path = workspace_root
            .join(".cache/dav1d-test-data/8-bit/svc")
            .join(svc_name);
        if !svc_path.is_file() {
            continue;
        }
        let raw = std::fs::read(&svc_path).unwrap();
        let ivf = parse_ivf(&raw);
        let tu_pkts: Vec<&[u8]> = ivf.iter().map(|&(_, p)| p).collect();
        let obu_pkts = split_to_single_obu_packets(&ivf);
        assert!(obu_pkts.len() > tu_pkts.len());

        let ref_tu_l0 = decode_packets_capi(
            av1oxide_api,
            &tu_pkts,
            1,
            1,
            DAV1D_DECODEFRAMETYPE_ALL,
            0,
            0,
            false,
        );
        let ref_tu_l1 = decode_packets_capi(
            av1oxide_api,
            &tu_pkts,
            1,
            1,
            DAV1D_DECODEFRAMETYPE_ALL,
            1,
            0,
            false,
        );
        assert_eq!(ref_tu_l0.len(), 8, "{svc_name} IVF all_layers=0");
        assert_eq!(
            ref_tu_l1.len(),
            if has_two_spatial_layers { 16 } else { 8 },
            "{svc_name} IVF all_layers=1"
        );

        for (mode_name, pkts, is_per_obu) in [
            ("IVF", tu_pkts.as_slice(), false),
            ("OBU", obu_pkts.as_slice(), true),
        ] {
            for all_layers in [0, 1] {
                for while_drain in [false, true] {
                    for (nt, mfd) in [(1, 1), (4, 1), (4, 2), (4, 4)] {
                        let expected_count = if !has_two_spatial_layers {
                            8usize
                        } else if all_layers == 1 {
                            16usize
                        } else if !is_per_obu {
                            8usize
                        } else if mfd == 1 {
                            16usize
                        } else if !while_drain {
                            8usize
                        } else if mfd == 2 {
                            11usize
                        } else {
                            10usize
                        };

                        let got = decode_packets_capi(
                            av1oxide_api,
                            pkts,
                            nt,
                            mfd,
                            DAV1D_DECODEFRAMETYPE_ALL,
                            all_layers,
                            0,
                            while_drain,
                        );
                        assert_eq!(
                            got.len(),
                            expected_count,
                            "{svc_name} {mode_name} all_layers={all_layers} wd={while_drain} t={nt} mfd={mfd}"
                        );
                        if expected_count == ref_tu_l0.len() {
                            for (a, b) in got.iter().zip(ref_tu_l0.iter()) {
                                assert_eq!(
                                    a.planes, b.planes,
                                    "{svc_name} {mode_name} all_layers={all_layers} wd={while_drain} t={nt} mfd={mfd} ref_tu_l0 ile piksel-özdeş olmalı"
                                );
                            }
                        } else if expected_count == ref_tu_l1.len() {
                            for (a, b) in got.iter().zip(ref_tu_l1.iter()) {
                                assert_eq!(
                                    a.planes, b.planes,
                                    "{svc_name} {mode_name} all_layers={all_layers} wd={while_drain} t={nt} mfd={mfd} ref_tu_l1 ile piksel-özdeş olmalı"
                                );
                            }
                        }
                        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
                        if let Some(ref_lib) = ref_dav1d.as_ref() {
                            let expected_c = decode_packets_capi(
                                ref_lib.api,
                                pkts,
                                nt,
                                mfd,
                                DAV1D_DECODEFRAMETYPE_ALL,
                                all_layers,
                                0,
                                while_drain,
                            );
                            assert_eq!(
                                got, expected_c,
                                "{svc_name} {mode_name} all_layers={all_layers} wd={while_drain} t={nt} mfd={mfd} gerçek C libdav1d ile piksel + timestamp + offset özdeş olmalı"
                            );
                        }
                        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
                        let _ = &ref_dav1d;
                    }
                }
            }
        }
    }

    // 3. Çoklu `OBU_TILE_GRP` paketine bölünmüş karelerde (`6.F.1`) `m.timestamp` ve `m.offset`
    //    değerleri kareyi tamamlayan son karo grubundan değil ilk karo grubundan (`tile[0].data.m`)
    //    alınmalıdır (`103 / 1003` vs `105 / 1005`).
    let multi_tg_path = workspace_root.join(".cache/dav1d-test-data/8-bit/data/00000019.ivf");
    if multi_tg_path.is_file() {
        let raw = std::fs::read(&multi_tg_path).unwrap();
        let ivf = parse_ivf(&raw);
        let obu_pkts = split_to_single_obu_packets(&ivf);
        // İlk paketteki ilk OBU_TILE_GRP (start == 0) indeksini bul.
        let mut first_tg_idx = None;
        for (i, &pkt) in obu_pkts.iter().enumerate() {
            let obu_type = (pkt[0] >> 3) & 0x0F;
            if obu_type == 4 {
                first_tg_idx = Some(i as i64);
                break;
            }
        }
        let first_tg_idx = first_tg_idx.expect("00000019.ivf içinde OBU_TILE_GRP bulunmalı");
        for (nt, mfd) in [(1, 1), (4, 2)] {
            let got = decode_packets_capi(
                av1oxide_api,
                &obu_pkts,
                nt,
                mfd,
                DAV1D_DECODEFRAMETYPE_ALL,
                1,
                0,
                true,
            );
            assert!(!got.is_empty());
            assert_eq!(
                got[0].timestamp,
                100 + first_tg_idx,
                "00000019.ivf t={nt} mfd={mfd} ilk kare timestamp ilk tile group'tan gelmeli"
            );
            assert_eq!(
                got[0].offset,
                1000 + first_tg_idx,
                "00000019.ivf t={nt} mfd={mfd} ilk kare offset ilk tile group'tan gelmeli"
            );
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            if let Some(ref_lib) = ref_dav1d.as_ref() {
                let expected_c = decode_packets_capi(
                    ref_lib.api,
                    &obu_pkts,
                    nt,
                    mfd,
                    DAV1D_DECODEFRAMETYPE_ALL,
                    1,
                    0,
                    true,
                );
                assert_eq!(
                    got, expected_c,
                    "00000019.ivf t={nt} mfd={mfd} gerçek C libdav1d ile piksel + timestamp + offset özdeş olmalı"
                );
            }
        }
    }
}

/// `6.F.2`: `show_existing_frame` yolunda özel `Dav1dPicAllocator` yeniden kullanımı.
/// `ccvb_film_grain.ivf`'in 3. paketinde (`pkt_idx == 2`, `show_existing_frame`) ayırıcı `-ENOMEM`
/// döndürse dahi `apply_grain = 0` altında `alloc_picture_callback` hiç çağrılmadığı için
/// 100 karenin tamamı hatasız çözülür ve kapanışta `live_allocs == 0` sağlanır.
#[test]
fn capi_custom_allocator_show_existing_reuses_buffer_and_no_leak() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let ivf_path = workspace_root.join(".cache/dav1d-test-data/8-bit/features/ccvb_film_grain.ivf");
    if !ivf_path.is_file() {
        return;
    }
    let raw = std::fs::read(&ivf_path).unwrap();
    let ivf = parse_ivf(&raw);
    let pkts: Vec<&[u8]> = ivf.iter().map(|&(_, p)| p).collect();

    struct ReuseAllocState {
        cur_pkt_idx: AtomicUsize,
        fail_on_pkt_idx: usize,
        alloc_calls: AtomicUsize,
        release_calls: AtomicUsize,
        live_allocs: AtomicUsize,
    }

    struct RawBuf {
        _y: Vec<u8>,
        _u: Vec<u8>,
        _v: Vec<u8>,
    }

    extern "C" fn reuse_alloc_cb(pic: *mut Dav1dPicture, cookie: *mut c_void) -> c_int {
        // SAFETY: `cookie` canlı `ReuseAllocState` ve `pic` geçerli `Dav1dPicture` işaretçisidir.
        unsafe {
            let st = &*(cookie as *const ReuseAllocState);
            let p = &mut *pic;
            assert!(!p.seq_hdr.is_null(), "alloc_cb içinde seq_hdr dolu olmalı");
            assert!(
                !p.frame_hdr.is_null(),
                "alloc_cb içinde frame_hdr dolu olmalı"
            );
            if st.cur_pkt_idx.load(Ordering::SeqCst) == st.fail_on_pkt_idx {
                return dav1d_err(ENOMEM);
            }
            let w = p.p.w as usize;
            let h = p.p.h as usize;
            let bps = if p.p.bpc > 8 { 2 } else { 1 };
            let stride_y = (w * bps + 63) & !63;
            let cw = w.div_ceil(2);
            let ch = h.div_ceil(2);
            let stride_uv = (cw * bps + 63) & !63;
            let mut y = vec![0u8; stride_y * h + 64];
            let mut u = vec![0u8; stride_uv * ch + 64];
            let mut v = vec![0u8; stride_uv * ch + 64];
            p.data[0] = y.as_mut_ptr() as *mut c_void;
            p.data[1] = u.as_mut_ptr() as *mut c_void;
            p.data[2] = v.as_mut_ptr() as *mut c_void;
            p.stride[0] = stride_y as isize;
            p.stride[1] = stride_uv as isize;
            let boxed = Box::new(RawBuf {
                _y: y,
                _u: u,
                _v: v,
            });
            p.allocator_data = Box::into_raw(boxed) as *mut c_void;
            st.alloc_calls.fetch_add(1, Ordering::SeqCst);
            st.live_allocs.fetch_add(1, Ordering::SeqCst);
            0
        }
    }

    extern "C" fn reuse_release_cb(pic: *mut Dav1dPicture, cookie: *mut c_void) {
        // SAFETY: `cookie` canlı `ReuseAllocState` ve `pic.allocator_data` `Box::into_raw(RawBuf)`'tır.
        unsafe {
            let st = &*(cookie as *const ReuseAllocState);
            let p = &mut *pic;
            if !p.allocator_data.is_null() {
                drop(Box::from_raw(p.allocator_data as *mut RawBuf));
                p.allocator_data = ptr::null_mut();
                st.release_calls.fetch_add(1, Ordering::SeqCst);
                st.live_allocs.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }

    let decode_default = |apply_grain: c_int| -> Vec<Vec<Vec<u8>>> {
        // SAFETY: Bağımsız varsayılan ayırıcılı `Dav1dContext` referans çözümü.
        unsafe {
            let mut s = Dav1dSettings::default();
            dav1d_default_settings(&mut s);
            s.n_threads = 1;
            s.max_frame_delay = 1;
            s.apply_grain = apply_grain;
            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!(dav1d_open(&mut c, &s), 0);
            let mut out = Vec::new();
            let mut pic = Dav1dPicture::default();
            for &pkt in &pkts {
                let mut data = Dav1dData::default();
                let dst = dav1d_data_create(&mut data, pkt.len());
                ptr::copy_nonoverlapping(pkt.as_ptr(), dst, pkt.len());
                loop {
                    let s_rc = dav1d_send_data(c, &mut data);
                    assert!(s_rc == 0 || s_rc == dav1d_err(EAGAIN));
                    loop {
                        let g_rc = dav1d_get_picture(c, &mut pic);
                        if g_rc == 0 {
                            out.push(copy_c_picture_planes(&pic));
                            dav1d_picture_unref(&mut pic);
                        } else {
                            assert_eq!(g_rc, dav1d_err(EAGAIN));
                            break;
                        }
                    }
                    if s_rc == 0 {
                        break;
                    }
                }
            }
            loop {
                let g_rc = dav1d_get_picture(c, &mut pic);
                if g_rc == dav1d_err(EAGAIN) {
                    break;
                }
                assert_eq!(g_rc, 0);
                out.push(copy_c_picture_planes(&pic));
                dav1d_picture_unref(&mut pic);
            }
            dav1d_close(&mut c);
            out
        }
    };

    let ref_no_grain = decode_default(0);
    let ref_with_grain = decode_default(1);
    assert_eq!(ref_no_grain.len(), 100);
    assert_eq!(ref_with_grain.len(), 100);

    for (apply_grain, fail_on_pkt, expected) in
        [(0, 2usize, &ref_no_grain), (1, usize::MAX, &ref_with_grain)]
    {
        for (nt, mfd) in [(1, 1), (4, 2)] {
            let st = ReuseAllocState {
                cur_pkt_idx: AtomicUsize::new(0),
                fail_on_pkt_idx: fail_on_pkt,
                alloc_calls: AtomicUsize::new(0),
                release_calls: AtomicUsize::new(0),
                live_allocs: AtomicUsize::new(0),
            };
            // SAFETY: Özel ayırıcı ömrü `dav1d_close` sonuna kadar geçerlidir.
            unsafe {
                let mut s = Dav1dSettings::default();
                dav1d_default_settings(&mut s);
                s.n_threads = nt;
                s.max_frame_delay = mfd;
                s.apply_grain = apply_grain;
                s.allocator = Dav1dPicAllocator {
                    cookie: (&st as *const ReuseAllocState) as *mut c_void,
                    alloc_picture_callback: Some(reuse_alloc_cb),
                    release_picture_callback: Some(reuse_release_cb),
                };

                let mut c: *mut Dav1dContext = ptr::null_mut();
                assert_eq!(dav1d_open(&mut c, &s), 0);
                let mut out = Vec::new();
                let mut pic = Dav1dPicture::default();
                for (idx, &pkt) in pkts.iter().enumerate() {
                    st.cur_pkt_idx.store(idx, Ordering::SeqCst);
                    let mut data = Dav1dData::default();
                    let dst = dav1d_data_create(&mut data, pkt.len());
                    ptr::copy_nonoverlapping(pkt.as_ptr(), dst, pkt.len());
                    data.m.timestamp = 500 + idx as i64;
                    loop {
                        let s_rc = dav1d_send_data(c, &mut data);
                        assert!(
                            s_rc == 0 || s_rc == dav1d_err(EAGAIN),
                            "apply_grain={apply_grain} t={nt} mfd={mfd} pkt={idx} send_data={s_rc}"
                        );
                        loop {
                            let g_rc = dav1d_get_picture(c, &mut pic);
                            if g_rc == 0 {
                                out.push(copy_c_picture_planes(&pic));
                                dav1d_picture_unref(&mut pic);
                            } else {
                                assert_eq!(g_rc, dav1d_err(EAGAIN));
                                break;
                            }
                        }
                        if s_rc == 0 {
                            break;
                        }
                    }
                }
                st.cur_pkt_idx.store(pkts.len(), Ordering::SeqCst);
                loop {
                    let g_rc = dav1d_get_picture(c, &mut pic);
                    if g_rc == dav1d_err(EAGAIN) {
                        break;
                    }
                    assert_eq!(g_rc, 0);
                    out.push(copy_c_picture_planes(&pic));
                    dav1d_picture_unref(&mut pic);
                }
                dav1d_close(&mut c);

                assert_eq!(
                    out.len(),
                    100,
                    "apply_grain={apply_grain} t={nt} mfd={mfd} 100 kare çözülmeli"
                );
                assert_eq!(
                    out, *expected,
                    "apply_grain={apply_grain} t={nt} mfd={mfd} piksel-özdeş olmalı"
                );
                assert!(st.alloc_calls.load(Ordering::SeqCst) > 0);
                assert_eq!(
                    st.alloc_calls.load(Ordering::SeqCst),
                    st.release_calls.load(Ordering::SeqCst),
                    "apply_grain={apply_grain} t={nt} mfd={mfd} alloc == release olmalı"
                );
                assert_eq!(
                    st.live_allocs.load(Ordering::SeqCst),
                    0,
                    "apply_grain={apply_grain} t={nt} mfd={mfd} sızıntı olmamalı"
                );
            }
        }
    }
}

#[test]
fn capi_install_prefix_transactional_preserves_existing_on_conflict() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let scratch_dir = workspace_root.join(".cache/capi-prefix-rollback-test");
    let _ = std::fs::remove_dir_all(&scratch_dir);
    let prefix_dir = scratch_dir.join("prefix");
    let target_dir = scratch_dir.join("target");

    let inc_dav1d = prefix_dir.join("include").join("dav1d");
    let lib_dir = prefix_dir.join("lib");
    let pc_dir = lib_dir.join("pkgconfig");
    std::fs::create_dir_all(&inc_dav1d).unwrap();
    std::fs::create_dir_all(&pc_dir).unwrap();

    // Mevcut sentinel dosyaları ve `lib/pkgconfig/dav1d.pc` dizin çakışmasını hazırla.
    let sentinel_hdr = inc_dav1d.join("dav1d.h");
    let sentinel_static = lib_dir.join("libdav1d.a");
    let sentinel_keep = prefix_dir.join("keep_unrelated.txt");
    #[cfg(target_os = "macos")]
    let sentinel_shared = lib_dir.join("libdav1d.7.dylib");
    #[cfg(not(target_os = "macos"))]
    let sentinel_shared = lib_dir.join("libdav1d.so.7.0.0");

    std::fs::write(&sentinel_hdr, b"OLD_DAV1D_H_SENTINEL").unwrap();
    std::fs::write(&sentinel_static, b"OLD_LIBDAV1D_A_SENTINEL").unwrap();
    std::fs::write(&sentinel_shared, b"OLD_SHARED_SENTINEL").unwrap();
    std::fs::write(&sentinel_keep, b"KEEP_UNRELATED_SENTINEL").unwrap();
    // Çakışma: `dav1d.pc` bir dosya yerine dizin!
    std::fs::create_dir(pc_dir.join("dav1d.pc")).unwrap();

    let output = Command::new("cargo")
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
        .output()
        .expect("install_capi_prefix çalıştırılmalı");

    assert!(
        !output.status.success(),
        "dav1d.pc dizin çakışmasında install_capi_prefix hata döndürmeli"
    );
    assert_eq!(
        std::fs::read(&sentinel_hdr).unwrap(),
        b"OLD_DAV1D_H_SENTINEL",
        "mevcut dav1d.h bayt-birebir korunmalı"
    );
    assert_eq!(
        std::fs::read(&sentinel_static).unwrap(),
        b"OLD_LIBDAV1D_A_SENTINEL",
        "mevcut libdav1d.a bayt-birebir korunmalı"
    );
    assert_eq!(
        std::fs::read(&sentinel_shared).unwrap(),
        b"OLD_SHARED_SENTINEL",
        "mevcut paylaşımlı kütüphane bayt-birebir korunmalı"
    );
    assert_eq!(
        std::fs::read(&sentinel_keep).unwrap(),
        b"KEEP_UNRELATED_SENTINEL",
        "ilgisiz dosya bayt-birebir korunmalı"
    );
    assert!(
        pc_dir.join("dav1d.pc").is_dir(),
        "çakışan dizine dokunulmamış olmalı"
    );
    // Artakalan sahne/yedek dosyası olmamalı
    for dir in [&inc_dav1d, &lib_dir, &pc_dir] {
        for entry in std::fs::read_dir(dir).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".av1oxide_"),
                "artakalan geçici dosya olmamalı: {name}"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&scratch_dir);
}

/// `6.F.5` (`P3`): `RLIMIT_FSIZE = 16 KiB` altında `install_capi_prefix` kopyalama ortasında
/// yazma hatasına düştüğünde kısmi `.av1oxide_stage_*` dosyalarını temizlemeli ve mevcut
/// kurulumu bayt-birebir korumalıdır.
#[cfg(unix)]
#[test]
fn capi_install_prefix_rlimit_fsize_cleans_partial_stage_files() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let scratch_dir = workspace_root.join(".cache/capi-prefix-rlimit-test");
    let _ = std::fs::remove_dir_all(&scratch_dir);
    let prefix_dir = scratch_dir.join("prefix");
    let target_dir = scratch_dir.join("target");

    // 1. Önce kurucuyu ve C-ABI kütüphanelerini normal sınırlarla ısıtıp geçerli bir kurulum yap.
    let warmup = Command::new("cargo")
        .current_dir(workspace_root)
        .args([
            "run",
            "--locked",
            "--release",
            "-p",
            "av1oxide",
            "--example",
            "install_capi_prefix",
            "--",
        ])
        .arg(&prefix_dir)
        .arg("--target-dir")
        .arg(&target_dir)
        .output()
        .expect("ısınma kurulumu çalıştırılmalı");
    assert!(
        warmup.status.success(),
        "ısınma kurulumu başarılı olmalı: {}",
        String::from_utf8_lossy(&warmup.stderr)
    );

    let inc_dav1d = prefix_dir.join("include").join("dav1d");
    let lib_dir = prefix_dir.join("lib");
    let pc_dir = lib_dir.join("pkgconfig");
    let sentinel_hdr = inc_dav1d.join("dav1d.h");
    let sentinel_pc = pc_dir.join("dav1d.pc");
    let sentinel_keep = prefix_dir.join("keep_unrelated.txt");
    #[cfg(target_os = "macos")]
    let sentinel_shared = lib_dir.join("libdav1d.7.dylib");
    #[cfg(not(target_os = "macos"))]
    let sentinel_shared = lib_dir.join("libdav1d.so.7.0.0");

    std::fs::write(&sentinel_hdr, b"SENTINEL_RLIMIT_DAV1D_H").unwrap();
    std::fs::write(&sentinel_pc, b"SENTINEL_RLIMIT_DAV1D_PC").unwrap();
    std::fs::write(&sentinel_shared, b"SENTINEL_RLIMIT_SHARED").unwrap();
    std::fs::write(&sentinel_keep, b"SENTINEL_RLIMIT_KEEP").unwrap();

    // 2. Doğrudan derlenmiş `install_capi_prefix` ikilisini `SIGXFSZ` yoksayılı ve
    //    `RLIMIT_FSIZE = 16 KiB` (`ulimit -f 32` * 512 B) ile çalıştır.
    let cargo_target_base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|p| {
            if p.is_absolute() {
                p
            } else {
                workspace_root.join(p)
            }
        })
        .unwrap_or_else(|| workspace_root.join("target"));
    let bin_path = cargo_target_base
        .join("release")
        .join("examples")
        .join("install_capi_prefix");
    assert!(
        bin_path.is_file(),
        "derlenmiş install_capi_prefix ikilisi bulunmalı: {}",
        bin_path.display()
    );
    let limited = Command::new("sh")
        .current_dir(workspace_root)
        .args([
            "-c",
            "trap '' XFSZ; ulimit -f 32; exec \"$1\" \"$2\" --target-dir \"$3\"",
            "sh",
        ])
        .arg(&bin_path)
        .arg(&prefix_dir)
        .arg(&target_dir)
        .output()
        .expect("RLIMIT_FSIZE alt süreci çalıştırılmalı");

    let limited_stderr = String::from_utf8_lossy(&limited.stderr);
    assert!(
        !limited.status.success(),
        "16 KiB RLIMIT_FSIZE altında kurulum başarısız olmalı"
    );
    assert!(
        limited_stderr.contains("sahnelenemedi"),
        "hata sahneleme yazma sınırından (RLIMIT_FSIZE) gelmeli, stderr: {limited_stderr}"
    );
    assert_eq!(
        std::fs::read(&sentinel_hdr).unwrap(),
        b"SENTINEL_RLIMIT_DAV1D_H",
        "mevcut dav1d.h korunmalı"
    );
    assert_eq!(
        std::fs::read(&sentinel_pc).unwrap(),
        b"SENTINEL_RLIMIT_DAV1D_PC",
        "mevcut dav1d.pc korunmalı"
    );
    assert_eq!(
        std::fs::read(&sentinel_shared).unwrap(),
        b"SENTINEL_RLIMIT_SHARED",
        "mevcut paylaşımlı kütüphane korunmalı"
    );
    assert_eq!(
        std::fs::read(&sentinel_keep).unwrap(),
        b"SENTINEL_RLIMIT_KEEP",
        "ilgisiz dosya korunmalı"
    );

    // Önek altında hiçbir kısmi `.av1oxide_stage_*` veya `.av1oxide_bak_*` dosyası kalmamalı!
    for dir in [&prefix_dir, &inc_dav1d, &lib_dir, &pc_dir] {
        for entry in std::fs::read_dir(dir).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".av1oxide_"),
                "RLIMIT_FSIZE hatasında artakalan geçici dosya olmamalı: {name}"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&scratch_dir);
}

/// `6.G.1` (`P2`): `n_fc > 1` halka tamponunda gecikmeli arka plan çözme hatası emekliye
/// ayrılırken yeni gelen kurtarma `KEY` karesini ve çoklu-OBU (superframe) paketin kalan
/// alt-karelerini düşürmemeli, hata `dav1d_get_picture` üzerinden eski bozuk paketin
/// `cached_error_props` bilgileriyle bildirilmelidir.
#[test]
fn capi_delayed_decode_error_preserves_superframe_and_recovery_keyframe() {
    let ivf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf",
    );
    let raw = std::fs::read(&ivf_path).expect("IVF fikstürü okunmalı");
    let ivf = parse_ivf(&raw);
    assert!(ivf.len() >= 5, "en az 5 karelik IVF fikstürü gerekir");

    let pkt_key = ivf[0].1;
    let pkt_inter1 = ivf[1].1;
    let pkt_inter2 = ivf[2].1;

    // `pkt_inter1` içindeki `OBU_FRAME` başlığını (`FrameHeader`) koruyup her karoyu 1 bayt
    // uzunluğa indirge (`tile_size_minus_1 = 0` + 1 bayt `0x00`): ayrıştırıcı ve
    // `split_tile_groups` hatasız geçer, arka plan işçi çözümü ise kesinlikle
    // `MSAC veri sonu aşıldı` (gecikmeli asenkron hata) üretir.
    let seq = av1oxide::bitstream::parse_sequence_header_obus(pkt_key)
        .unwrap()
        .expect("SequenceHeader bulunmalı");
    let mut fh0_opt = None;
    {
        let mut off0 = 0usize;
        while off0 < pkt_key.len() {
            let Some(obu) = av1oxide::bitstream::split_next_with(pkt_key, &mut off0, true).unwrap()
            else {
                break;
            };
            if obu.header.obu_type == av1oxide::bitstream::ObuType::Frame {
                let mut br = av1oxide::bitstream::BitReader::new(obu.payload);
                let refs0: [Option<av1oxide::bitstream::FrameHeader>; 8] = Default::default();
                let ctx = av1oxide::bitstream::FrameHeaderContext {
                    sequence_header: &seq,
                    refs: &refs0,
                    temporal_id: 0,
                    spatial_id: 0,
                    strict_std_compliance: false,
                };
                fh0_opt = Some(av1oxide::bitstream::FrameHeader::parse(&mut br, &ctx).unwrap());
                break;
            }
        }
    }
    let fh0 = fh0_opt.expect("KEY FrameHeader bulunmalı");
    let mut corrupt_inter1 = Vec::new();
    let mut off = 0usize;
    while off < pkt_inter1.len() {
        let Some(obu) = av1oxide::bitstream::split_next_with(pkt_inter1, &mut off, true).unwrap()
        else {
            break;
        };
        if obu.header.obu_type == av1oxide::bitstream::ObuType::Frame {
            let mut br = av1oxide::bitstream::BitReader::new(obu.payload);
            let refs1: [Option<av1oxide::bitstream::FrameHeader>; 8] =
                std::array::from_fn(|_| Some(fh0.clone()));
            let ctx = av1oxide::bitstream::FrameHeaderContext {
                sequence_header: &seq,
                refs: &refs1,
                temporal_id: 0,
                spatial_id: 0,
                strict_std_compliance: false,
            };
            let fh = av1oxide::bitstream::FrameHeader::parse(&mut br, &ctx).unwrap();
            br.align_to_byte();
            let fh_end = br.byte_position();
            let n_tiles = usize::from(fh.tiling.cols) * usize::from(fh.tiling.rows);
            let size_bytes = usize::from(fh.tiling.n_bytes);
            let mut body = obu.payload[..fh_end].to_vec();
            for t in 0..n_tiles {
                if t + 1 < n_tiles {
                    body.resize(body.len() + size_bytes, 0u8);
                }
                body.push(0x00);
            }
            corrupt_inter1.extend_from_slice(&[0x12, 0x00, 0x32]);
            let mut rem = body.len();
            loop {
                let mut byte = (rem & 0x7F) as u8;
                rem >>= 7;
                if rem != 0 {
                    byte |= 0x80;
                }
                corrupt_inter1.push(byte);
                if rem == 0 {
                    break;
                }
            }
            corrupt_inter1.extend_from_slice(&body);
            break;
        }
    }
    assert!(!corrupt_inter1.is_empty());

    // Süper-kare paketi: 2 bağımsız kare (`KEY` + `INTER1`) tek bir `Dav1dData` paketinde.
    let mut superframe_pkt = Vec::with_capacity(pkt_key.len() + pkt_inter1.len());
    superframe_pkt.extend_from_slice(pkt_key);
    superframe_pkt.extend_from_slice(pkt_inter1);

    // Referans pikseller (tek paket, temiz çözüm: KEY, INTER1, INTER2).
    let ref_planes = |pkts: &[&[u8]]| -> Vec<Vec<Vec<u8>>> {
        // SAFETY: Bağımsız tek-thread C-ABI referans çözücüsü.
        unsafe {
            let mut s = Dav1dSettings::default();
            dav1d_default_settings(&mut s);
            s.n_threads = 1;
            s.max_frame_delay = 1;
            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!(dav1d_open(&mut c, &s), 0);
            let mut out = Vec::new();
            let mut pic = Dav1dPicture::default();
            for &pkt in pkts {
                let mut d = Dav1dData::default();
                let dst = dav1d_data_create(&mut d, pkt.len());
                ptr::copy_nonoverlapping(pkt.as_ptr(), dst, pkt.len());
                loop {
                    let sr = dav1d_send_data(c, &mut d);
                    assert!(sr == 0 || sr == dav1d_err(EAGAIN));
                    loop {
                        let gr = dav1d_get_picture(c, &mut pic);
                        if gr == 0 {
                            out.push(copy_c_picture_planes(&pic));
                            dav1d_picture_unref(&mut pic);
                        } else {
                            assert_eq!(gr, dav1d_err(EAGAIN));
                            break;
                        }
                    }
                    if sr == 0 {
                        break;
                    }
                }
            }
            loop {
                let gr = dav1d_get_picture(c, &mut pic);
                if gr == dav1d_err(EAGAIN) {
                    break;
                }
                assert_eq!(gr, 0);
                out.push(copy_c_picture_planes(&pic));
                dav1d_picture_unref(&mut pic);
            }
            dav1d_close(&mut c);
            out
        }
    };

    let clean_refs = ref_planes(&[pkt_key, pkt_inter1, pkt_inter2]);
    assert_eq!(clean_refs.len(), 3);

    // Akış düzeni:
    //   0: KEY (sağlam)
    //   1: corrupt_inter1 (arka planda başarısız -> ts=1001, off=2001)
    //   2: INTER2 (bozuk 1'e bağlı -> arka planda başarısız -> ts=1002, off=2002)
    //   3: KEY (kurtarma karesi -> ts=1003)
    //   4: superframe(KEY + INTER1) (1. alt-kare gecikmeli hatayı emekliye ayırırken 2. alt-kare düşmemeli -> ts=1004)
    //   5: INTER2 (superframe içindeki INTER1'e bağlı -> ts=1005)
    let stream: &[(&[u8], i64, i64)] = &[
        (pkt_key, 1000, 2000),
        (&corrupt_inter1, 1001, 2001),
        (pkt_inter2, 1002, 2002),
        (pkt_key, 1003, 2003),
        (&superframe_pkt, 1004, 2004),
        (pkt_inter2, 1005, 2005),
    ];

    for (nt, mfd) in [(4, 4), (14, 4), (4, 2)] {
        // SAFETY: Her koşu bağımsız `Dav1dContext` açıp kapatır.
        unsafe {
            let mut s = Dav1dSettings::default();
            dav1d_default_settings(&mut s);
            s.n_threads = nt;
            s.max_frame_delay = mfd;
            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!(dav1d_open(&mut c, &s), 0);

            let mut decoded: Vec<(i64, Vec<Vec<u8>>)> = Vec::new();
            let mut err_props_seen: Vec<(i64, i64)> = Vec::new();
            let mut pic = Dav1dPicture::default();

            for &(pkt, ts, off) in stream {
                let mut d = Dav1dData::default();
                let dst = dav1d_data_create(&mut d, pkt.len());
                ptr::copy_nonoverlapping(pkt.as_ptr(), dst, pkt.len());
                d.m.timestamp = ts;
                d.m.offset = off;

                loop {
                    let sr = dav1d_send_data(c, &mut d);
                    assert!(
                        sr == 0 || sr == dav1d_err(EAGAIN),
                        "t={nt} mfd={mfd} ts={ts}: gecikmeli hata dav1d_send_data'yı kesmemeli (sr={sr})"
                    );
                    let gr = dav1d_get_picture(c, &mut pic);
                    if gr == 0 {
                        decoded.push((pic.m.timestamp, copy_c_picture_planes(&pic)));
                        dav1d_picture_unref(&mut pic);
                    } else if gr != dav1d_err(EAGAIN) {
                        let mut ep = Dav1dDataProps::default();
                        assert_eq!(dav1d_get_decode_error_data_props(c, &mut ep), 0);
                        err_props_seen.push((ep.timestamp, ep.offset));
                    }
                    if sr == 0 {
                        break;
                    }
                }
            }
            loop {
                let gr = dav1d_get_picture(c, &mut pic);
                if gr == dav1d_err(EAGAIN) {
                    break;
                }
                if gr == 0 {
                    decoded.push((pic.m.timestamp, copy_c_picture_planes(&pic)));
                    dav1d_picture_unref(&mut pic);
                } else {
                    let mut ep = Dav1dDataProps::default();
                    assert_eq!(dav1d_get_decode_error_data_props(c, &mut ep), 0);
                    err_props_seen.push((ep.timestamp, ep.offset));
                }
            }
            dav1d_close(&mut c);

            // Hata özellikleri yalnızca bozuk paketlere (`1001/2001` ve bağımlı `1002/2002`) ait olmalı.
            assert!(
                err_props_seen.contains(&(1001, 2001)),
                "t={nt} mfd={mfd} bozuk paket 1001/2001 hata metaverisi bildirilmeli: {err_props_seen:?}"
            );
            assert!(
                err_props_seen.iter().all(|&(t, _)| t == 1001 || t == 1002),
                "t={nt} mfd={mfd} kurtarma/superframe paketleri hata üretmemeli: {err_props_seen:?}"
            );

            // Beklenen 5 sağlam kare: ts=1000 (KEY), ts=1003 (kurtarma KEY), ts=1004 (superframe KEY),
            // ts=1004 (superframe INTER1), ts=1005 (INTER2).
            let got_ts: Vec<i64> = decoded.iter().map(|(t, _)| *t).collect();
            assert_eq!(
                got_ts,
                vec![1000, 1003, 1004, 1004, 1005],
                "t={nt} mfd={mfd} kurtarma KEY ve superframe alt-karelerinin tamamı korunmalı"
            );
            assert_eq!(decoded[0].1, clean_refs[0]);
            assert_eq!(decoded[1].1, clean_refs[0]);
            assert_eq!(decoded[2].1, clean_refs[0]);
            assert_eq!(decoded[3].1, clean_refs[1]);
            assert_eq!(decoded[4].1, clean_refs[2]);
        }
    }
}

/// Upstream `videolan/dav1d@f8ab763da5` (`frmhdr.w/h > seqhdr.max_w/h` -> `DAV1D_ERR(EINVAL)`)
/// ve `videolan/dav1d@e5fd231fe9` (`OBU_META_ITUT_T35` tahsis/sınır sonrası sızıntısız toparlanma)
/// diferansiyel C-ABI regresyon testi.
#[test]
fn capi_frame_size_exceeding_seq_max_and_itut_t35_parity() {
    use std::sync::Mutex;

    #[derive(Clone, Copy)]
    struct Dav1dApi {
        default_settings: unsafe extern "C" fn(*mut Dav1dSettings),
        open: unsafe extern "C" fn(*mut *mut Dav1dContext, *const Dav1dSettings) -> c_int,
        close: unsafe extern "C" fn(*mut *mut Dav1dContext),
        flush: unsafe extern "C" fn(*mut Dav1dContext),
        data_create: unsafe extern "C" fn(*mut Dav1dData, usize) -> *mut u8,
        data_unref: unsafe extern "C" fn(*mut Dav1dData),
        data_props_unref: unsafe extern "C" fn(*mut Dav1dDataProps),
        send_data: unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dData) -> c_int,
        get_picture: unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dPicture) -> c_int,
        picture_unref: unsafe extern "C" fn(*mut Dav1dPicture),
        get_event_flags: unsafe extern "C" fn(*mut Dav1dContext, *mut u32) -> c_int,
        get_frame_error_data_props:
            unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dDataProps) -> c_int,
    }

    let av1oxide_api = Dav1dApi {
        default_settings: dav1d_default_settings,
        open: dav1d_open,
        close: dav1d_close,
        flush: dav1d_flush,
        data_create: dav1d_data_create,
        data_unref: dav1d_data_unref,
        data_props_unref: dav1d_data_props_unref,
        send_data: dav1d_send_data,
        get_picture: dav1d_get_picture,
        picture_unref: dav1d_picture_unref,
        get_event_flags: dav1d_get_event_flags,
        get_frame_error_data_props: dav1d_get_decode_error_data_props,
    };

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    struct RefDav1dLib {
        handle: *mut c_void,
        api: Dav1dApi,
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    impl RefDav1dLib {
        fn try_load(so_path: &std::path::Path) -> Option<Self> {
            use core::ffi::c_char;
            use std::ffi::CString;
            unsafe extern "C" {
                fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
                fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            }
            const RTLD_NOW: c_int = 2;
            const RTLD_LOCAL: c_int = 0;
            if !so_path.is_file() {
                return None;
            }
            let cpath = CString::new(so_path.to_str()?).ok()?;
            // SAFETY: Referans C `libdav1d.so.7` paylaşımlı kütüphanesi salt test karşılaştırması için yüklenir.
            unsafe {
                let handle = dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_LOCAL);
                if handle.is_null() {
                    return None;
                }
                let sym = |name: &CStr| -> *mut c_void { dlsym(handle, name.as_ptr()) };
                let f_def = sym(c"dav1d_default_settings");
                let f_open = sym(c"dav1d_open");
                let f_close = sym(c"dav1d_close");
                let f_flush = sym(c"dav1d_flush");
                let f_create = sym(c"dav1d_data_create");
                let f_dunref = sym(c"dav1d_data_unref");
                let f_dpunref = sym(c"dav1d_data_props_unref");
                let f_send = sym(c"dav1d_send_data");
                let f_get = sym(c"dav1d_get_picture");
                let f_unref = sym(c"dav1d_picture_unref");
                let f_evflags = sym(c"dav1d_get_event_flags");
                let f_errprops = sym(c"dav1d_get_decode_error_data_props");
                if [
                    f_def, f_open, f_close, f_flush, f_create, f_dunref, f_dpunref, f_send, f_get,
                    f_unref, f_evflags, f_errprops,
                ]
                .iter()
                .any(|p| p.is_null())
                {
                    return None;
                }
                Some(Self {
                    handle,
                    api: Dav1dApi {
                        default_settings: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dSettings),
                        >(f_def),
                        open: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(
                                *mut *mut Dav1dContext,
                                *const Dav1dSettings,
                            ) -> c_int,
                        >(f_open),
                        close: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut *mut Dav1dContext),
                        >(f_close),
                        flush: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dContext),
                        >(f_flush),
                        data_create: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dData, usize) -> *mut u8,
                        >(f_create),
                        data_unref: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dData),
                        >(f_dunref),
                        data_props_unref: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dDataProps),
                        >(f_dpunref),
                        send_data: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dData) -> c_int,
                        >(f_send),
                        get_picture: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dPicture) -> c_int,
                        >(f_get),
                        picture_unref: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dPicture),
                        >(f_unref),
                        get_event_flags: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dContext, *mut u32) -> c_int,
                        >(f_evflags),
                        get_frame_error_data_props: core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn(*mut Dav1dContext, *mut Dav1dDataProps) -> c_int,
                        >(f_errprops),
                    },
                })
            }
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    impl Drop for RefDav1dLib {
        fn drop(&mut self) {
            unsafe extern "C" {
                fn dlclose(handle: *mut c_void) -> c_int;
            }
            // SAFETY: `self.handle`, `dlopen` ile açılmış geçerli tanıtıcıdır.
            unsafe {
                let _ = dlclose(self.handle);
            }
        }
    }

    struct MsbBits {
        bits: Vec<u8>,
    }
    impl MsbBits {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn bit(&mut self, v: u32) {
            self.bits.push((v & 1) as u8);
        }
        fn bits(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                self.bit((v >> i) & 1);
            }
        }
        fn bytes(self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, b) in self.bits.iter().enumerate() {
                if *b != 0 {
                    out[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            out
        }
    }

    fn wrap_obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + payload.len());
        out.push((obu_type << 3) | 0x02); // has_size_field = 1
        let mut rem = payload.len();
        loop {
            let mut b = (rem & 0x7F) as u8;
            rem >>= 7;
            if rem != 0 {
                b |= 0x80;
            }
            out.push(b);
            if rem == 0 {
                break;
            }
        }
        out.extend_from_slice(payload);
        out
    }

    // `SEQ_OBU`: width_n_bits=8, height_n_bits=8, max_width=1 (0), max_height=1 (0).
    let mut seq_bits = MsbBits::new();
    seq_bits.bits(0, 3); // seq_profile = 0
    seq_bits.bit(0); // still_picture = 0
    seq_bits.bit(0); // reduced_still_picture_header = 0
    seq_bits.bit(0); // timing_info_present_flag = 0
    seq_bits.bit(0); // initial_display_delay_present_flag = 0
    seq_bits.bits(0, 5); // operating_points_cnt_minus_1 = 0
    seq_bits.bits(0, 12); // operating_point_idc[0] = 0
    seq_bits.bits(0, 5); // seq_level_idx[0] = 0
    seq_bits.bits(7, 4); // frame_width_bits_minus_1 = 7 (8 bit)
    seq_bits.bits(7, 4); // frame_height_bits_minus_1 = 7 (8 bit)
    seq_bits.bits(0, 8); // max_frame_width_minus_1 = 0 -> max_width = 1
    seq_bits.bits(0, 8); // max_frame_height_minus_1 = 0 -> max_height = 1
    seq_bits.bit(0); // frame_id_numbers_present_flag = 0
    seq_bits.bit(1); // use_128x128_superblock = 1
    seq_bits.bit(1); // enable_filter_intra = 1
    seq_bits.bit(1); // enable_intra_edge_filter = 1
    seq_bits.bit(1); // enable_interintra_compound = 1
    seq_bits.bit(1); // enable_masked_compound = 1
    seq_bits.bit(1); // enable_warped_motion = 1
    seq_bits.bit(1); // enable_dual_filter = 1
    seq_bits.bit(1); // enable_order_hint = 1
    seq_bits.bit(1); // enable_jnt_comp = 1
    seq_bits.bit(1); // enable_ref_frame_mvs = 1
    seq_bits.bit(1); // seq_choose_screen_content_tools = 1
    seq_bits.bit(1); // seq_choose_integer_mv = 1
    seq_bits.bits(6, 3); // order_hint_bits_minus_1 = 6 (7 bit)
    seq_bits.bit(0); // enable_superres = 0
    seq_bits.bit(1); // enable_cdef = 1
    seq_bits.bit(1); // enable_restoration = 1
    seq_bits.bit(0); // high_bitdepth = 0
    seq_bits.bit(0); // mono_chrome = 0
    seq_bits.bit(0); // color_description_present_flag = 0
    seq_bits.bit(0); // color_range = 0
    seq_bits.bits(0, 2); // chroma_sample_position = 0
    seq_bits.bit(0); // separate_uv_delta_q = 0
    seq_bits.bit(0); // film_grain_params_present = 0
    seq_bits.bit(1); // trailing_one_bit = 1
    let seq_obu = wrap_obu(1, &seq_bits.bytes());

    // 1. `KEY_FRAME` (`frame_size_override = 1`) içinde `width = 2 > max_width = 1` ve `height = 2 > max_height = 1`:
    let build_bad_key_pkt = |w_m1: u32, h_m1: u32| -> Vec<u8> {
        let mut fb = MsbBits::new();
        fb.bit(0); // show_existing_frame = 0
        fb.bits(0, 2); // frame_type = KEY_FRAME (0)
        fb.bit(1); // show_frame = 1 (error_resilient_mode örtük olarak 1, bit okunmaz)
        fb.bit(0); // disable_cdf_update = 0
        fb.bit(0); // allow_screen_content_tools = 0
        fb.bit(1); // frame_size_override_flag = 1
        fb.bits(0, 7); // order_hint = 0
        fb.bits(w_m1, 8); // frame_width_minus_1
        fb.bits(h_m1, 8); // frame_height_minus_1
        fb.bits(0, 32); // padding bits
        fb.bits(0, 32);
        let frame_obu = wrap_obu(6, &fb.bytes());
        let mut pkt = vec![0x12, 0x00];
        pkt.extend_from_slice(&seq_obu);
        pkt.extend_from_slice(&frame_obu);
        pkt
    };

    // 2. Geçerli 1x1 `KEY_FRAME` (`MINIMAL_1X1_OBUS`) ardından aynı sequence parametreleriyle
    //    `max_width = 1, max_height = 1, width_n_bits = 8, height_n_bits = 8` altında
    //    `INTER_FRAME` (`use_ref = true`, `found_ref = [0; 7]`, `frame_size_override = 1`)
    //    içinde `width = 2 > max_width = 1`:
    let build_bad_inter_pkt = |w_m1: u32, h_m1: u32| -> Vec<u8> {
        let mut fb = MsbBits::new();
        fb.bit(0); // show_existing_frame = 0
        fb.bits(1, 2); // frame_type = INTER_FRAME (1)
        fb.bit(1); // show_frame = 1
        fb.bit(0); // error_resilient_mode = 0
        fb.bit(0); // disable_cdf_update = 0
        fb.bit(0); // allow_screen_content_tools = 0
        fb.bit(1); // frame_size_override_flag = 1
        fb.bits(1, 7); // order_hint = 1
        fb.bits(0, 3); // primary_ref_frame = 0
        fb.bits(0, 8); // refresh_frame_flags = 0
        fb.bit(0); // frame_refs_short_signaling = 0
        for _ in 0..7 {
            fb.bits(0, 3); // ref_frame_idx[i] = 0
        }
        for _ in 0..7 {
            fb.bit(0); // found_ref[i] = 0 (use_ref döngüsü)
        }
        fb.bits(w_m1, 8); // frame_width_minus_1
        fb.bits(h_m1, 8); // frame_height_minus_1
        fb.bits(0, 32); // padding bits
        fb.bits(0, 32);
        let frame_obu = wrap_obu(6, &fb.bytes());
        let mut pkt = vec![0x12, 0x00];
        pkt.extend_from_slice(&seq_obu);
        pkt.extend_from_slice(&frame_obu);
        pkt
    };

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let ref_dav1d =
        RefDav1dLib::try_load(&workspace_root.join(".cache/dav1d-build-asm/src/libdav1d.so.7"));

    let run_seq = |api: Dav1dApi, pkts: &[&[u8]]| -> Vec<(c_int, c_int)> {
        // SAFETY: Her koşu bağımsız `Dav1dContext` açıp kapatır.
        unsafe {
            let mut s = Dav1dSettings::default();
            (api.default_settings)(&mut s);
            s.n_threads = 1;
            s.max_frame_delay = 1;
            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!((api.open)(&mut c, &s), 0);
            let mut results = Vec::new();
            let mut pic = Dav1dPicture::default();
            for &pkt in pkts {
                let mut d = Dav1dData::default();
                let dst = (api.data_create)(&mut d, pkt.len());
                ptr::copy_nonoverlapping(pkt.as_ptr(), dst, pkt.len());
                let sr = (api.send_data)(c, &mut d);
                if d.sz > 0 {
                    (api.data_unref)(&mut d);
                }
                let gr = (api.get_picture)(c, &mut pic);
                if gr == 0 {
                    (api.picture_unref)(&mut pic);
                }
                results.push((sr, gr));
            }
            (api.close)(&mut c);
            results
        }
    };

    for (w_m1, h_m1) in [(1u32, 0u32), (0u32, 1u32)] {
        let bad_key = build_bad_key_pkt(w_m1, h_m1);
        let bad_inter = build_bad_inter_pkt(w_m1, h_m1);

        let got_key = run_seq(av1oxide_api, &[&bad_key]);
        assert_eq!(
            got_key[0],
            (dav1d_err(EINVAL), dav1d_err(EAGAIN)),
            "KEY frame_size_override max_width/height aşımı send_data'da -EINVAL döndürmeli (-ERANGE değil)"
        );

        let got_inter = run_seq(av1oxide_api, &[SAMPLE_OBU_PACKETS[0], &bad_inter]);
        assert_eq!(got_inter[0], (0, 0));
        assert_eq!(
            got_inter[1],
            (dav1d_err(EINVAL), dav1d_err(EAGAIN)),
            "INTER frame_size_override max_width/height aşımı send_data'da -EINVAL döndürmeli"
        );

        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        if let Some(ref_lib) = ref_dav1d.as_ref() {
            let exp_key = run_seq(ref_lib.api, &[&bad_key]);
            let exp_inter = run_seq(ref_lib.api, &[SAMPLE_OBU_PACKETS[0], &bad_inter]);
            assert_eq!(
                got_key, exp_key,
                "KEY frame_size_override reddi referans C libdav1d ile birebir aynı olmalı"
            );
            assert_eq!(
                got_inter, exp_inter,
                "INTER frame_size_override reddi referans C libdav1d ile birebir aynı olmalı"
            );
        }
    }

    // 3. `OBU_META_ITUT_T35` çoklu girdi + kare teslimi ve referans C `libdav1d` eşliği:
    let mut t35_pkt = Vec::new();
    t35_pkt.extend_from_slice(&wrap_obu(5, &[0x04, 0xB5, 0x11, 0x22, 0x33, 0x80]));
    t35_pkt.extend_from_slice(&wrap_obu(5, &[0x04, 0xFF, 0x07, 0xAA, 0xBB, 0x80]));
    t35_pkt.extend_from_slice(SAMPLE_OBU_PACKETS[0]);

    let inspect_t35 = |api: Dav1dApi| -> Vec<(u8, u8, Vec<u8>)> {
        // SAFETY: Bağımsız `Dav1dContext` üzerinde T.35 metaverisi okunup serbest bırakılır.
        unsafe {
            let mut s = Dav1dSettings::default();
            (api.default_settings)(&mut s);
            s.n_threads = 1;
            s.max_frame_delay = 1;
            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!((api.open)(&mut c, &s), 0);
            let mut d = Dav1dData::default();
            let dst = (api.data_create)(&mut d, t35_pkt.len());
            ptr::copy_nonoverlapping(t35_pkt.as_ptr(), dst, t35_pkt.len());
            assert_eq!((api.send_data)(c, &mut d), 0);
            let mut pic = Dav1dPicture::default();
            assert_eq!((api.get_picture)(c, &mut pic), 0);
            let mut entries = Vec::new();
            for i in 0..pic.n_itut_t35 {
                let entry = &*pic.itut_t35.add(i);
                let payload =
                    std::slice::from_raw_parts(entry.payload, entry.payload_size).to_vec();
                entries.push((
                    entry.country_code,
                    entry.country_code_extension_byte,
                    payload,
                ));
            }
            (api.picture_unref)(&mut pic);
            (api.close)(&mut c);
            entries
        }
    };

    let got_t35 = inspect_t35(av1oxide_api);
    assert_eq!(
        got_t35,
        vec![
            (0xB5, 0x00, vec![0x11, 0x22, 0x33]),
            (0xFF, 0x07, vec![0xAA, 0xBB]),
        ]
    );
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if let Some(ref_lib) = ref_dav1d.as_ref() {
        let exp_t35 = inspect_t35(ref_lib.api);
        assert_eq!(got_t35, exp_t35);
    }

    // 4. `strict_std_compliance` karo sonu MSAC trailing bit + sıfır dolgu denetimi
    //    (`decode.c:2620-2635, 2786-2791`):
    //    Geçerli ilk paketin (`SAMPLE_OBU_PACKETS[0]`) sonuna `0x01` sıfır-dışı kuyruk baytı
    //    eklendiğinde `strict_std_compliance = 0` altında çözülür (`0`), `strict_std_compliance = 1`
    //    altında `-EINVAL` ile reddedilir.
    let mut bad_msac_tail_pkt = Vec::new();
    {
        let pkt0 = SAMPLE_OBU_PACKETS[0];
        let mut off = 0usize;
        while off < pkt0.len() {
            let start = off;
            let Some(obu) = av1oxide::bitstream::split_next_with(pkt0, &mut off, true).unwrap()
            else {
                break;
            };
            if obu.header.obu_type == av1oxide::bitstream::ObuType::Frame {
                let mut payload = obu.payload.to_vec();
                payload.push(0x01); // sıfır olmayan MSAC kuyruk baytı
                bad_msac_tail_pkt.extend_from_slice(&wrap_obu(6, &payload));
            } else {
                bad_msac_tail_pkt.extend_from_slice(&pkt0[start..off]);
            }
        }
    }
    let run_strict_msac = |api: Dav1dApi, strict: c_int| -> (c_int, c_int) {
        // SAFETY: Bağımsız `Dav1dContext` üzerinde `strict_std_compliance` sınanır.
        unsafe {
            let mut s = Dav1dSettings::default();
            (api.default_settings)(&mut s);
            s.n_threads = 1;
            s.max_frame_delay = 1;
            s.strict_std_compliance = strict;
            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!((api.open)(&mut c, &s), 0);
            let mut d = Dav1dData::default();
            let dst = (api.data_create)(&mut d, bad_msac_tail_pkt.len());
            ptr::copy_nonoverlapping(bad_msac_tail_pkt.as_ptr(), dst, bad_msac_tail_pkt.len());
            let sr = (api.send_data)(c, &mut d);
            if d.sz > 0 {
                (api.data_unref)(&mut d);
            }
            let mut pic = Dav1dPicture::default();
            let gr = (api.get_picture)(c, &mut pic);
            if gr == 0 {
                (api.picture_unref)(&mut pic);
            }
            (api.close)(&mut c);
            (sr, gr)
        }
    };
    let got_non_strict = run_strict_msac(av1oxide_api, 0);
    let got_strict = run_strict_msac(av1oxide_api, 1);
    assert_eq!(got_non_strict, (0, 0));
    assert_eq!(got_strict, (dav1d_err(EINVAL), dav1d_err(EAGAIN)));
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if let Some(ref_lib) = ref_dav1d.as_ref() {
        assert_eq!(got_non_strict, run_strict_msac(ref_lib.api, 0));
        assert_eq!(got_strict, run_strict_msac(ref_lib.api, 1));
    }

    // 5. `max_frame_delay > 1` kipinde bozuk `tile_size_minus_1` (`split_tile_groups`) hatasının
    //    `dav1d_send_data` yerine işçi görevine (`dav1d_get_picture`) ertelenmesi
    //    (`decode.c:3191-3228`, `thread_task.c:802-814`):
    //    2 karolu (128x64, `sb128 = 0`, 2 sütun x 1 satır) bir `KEY_FRAME` paketinde ilk karonun
    //    `tile_size_minus_1` alanı kalan yükten büyük (`0xFF = 256 bayt`) verilir.
    let bad_tile_size_pkt = {
        let mut sb = MsbBits::new();
        sb.bits(0, 3); // seq_profile = 0
        sb.bit(0); // still_picture = 0
        sb.bit(0); // reduced_still_picture_header = 0
        sb.bit(0); // timing_info_present_flag = 0
        sb.bit(0); // initial_display_delay_present_flag = 0
        sb.bits(0, 5); // operating_points_cnt_minus_1 = 0
        sb.bits(0, 12); // operating_point_idc[0] = 0
        sb.bits(0, 5); // seq_level_idx[0] = 0
        sb.bits(7, 4); // frame_width_bits_minus_1 = 7 (8 bit)
        sb.bits(7, 4); // frame_height_bits_minus_1 = 7 (8 bit)
        sb.bits(127, 8); // max_frame_width_minus_1 = 127 (128 px)
        sb.bits(63, 8); // max_frame_height_minus_1 = 63 (64 px)
        sb.bit(0); // frame_id_numbers_present_flag = 0
        sb.bit(0); // use_128x128_superblock = 0 (64x64 SB -> 2x1 SB)
        sb.bit(0); // enable_filter_intra = 0
        sb.bit(0); // enable_intra_edge_filter = 0
        sb.bit(0); // enable_interintra_compound = 0
        sb.bit(0); // enable_masked_compound = 0
        sb.bit(0); // enable_warped_motion = 0
        sb.bit(0); // enable_dual_filter = 0
        sb.bit(0); // enable_order_hint = 0
        sb.bit(1); // seq_choose_screen_content_tools = 1
        sb.bit(1); // seq_choose_integer_mv = 1
        sb.bit(0); // enable_superres = 0
        sb.bit(0); // enable_cdef = 0
        sb.bit(0); // enable_restoration = 0
        sb.bit(0); // high_bitdepth = 0
        sb.bit(0); // mono_chrome = 0
        sb.bit(0); // color_description_present_flag = 0
        sb.bit(0); // color_range = 0
        sb.bits(0, 2); // chroma_sample_position = 0
        sb.bit(0); // separate_uv_delta_q = 0
        sb.bit(0); // film_grain_params_present = 0
        sb.bit(1); // trailing_one_bit = 1
        let seq_2tile = wrap_obu(1, &sb.bytes());

        let mut fb = MsbBits::new();
        fb.bit(0); // show_existing_frame = 0
        fb.bits(0, 2); // frame_type = KEY_FRAME
        fb.bit(1); // show_frame = 1
        fb.bit(0); // disable_cdf_update = 0
        fb.bit(0); // allow_screen_content_tools = 0
        fb.bit(0); // frame_size_override_flag = 0
        fb.bit(0); // render_and_frame_size_different = 0
        // tile_info: 128x64 (2x1 64x64 SB)
        fb.bit(1); // uniform_tile_spacing_flag = 1
        fb.bit(1); // increment_tile_cols_log2: 1 -> tile_cols_log2 = 1 (2 tile col)
        // max_tile_Rows_log2 = 0 -> satır biti okunmaz
        fb.bits(0, 1); // context_update_tile_id = 0 (1 bit)
        fb.bits(0, 2); // tile_size_bytes_minus_1 = 0 (1 bayt)
        // quantization_params:
        fb.bits(64, 8); // base_q_idx = 64
        fb.bit(0); // diff_uv_delta (ydc_delta_present = 0)
        fb.bit(0); // udc_delta_present = 0
        fb.bit(0); // uac_delta_present = 0
        fb.bit(0); // using_qmatrix = 0
        // segmentation_params:
        fb.bit(0); // segmentation_enabled = 0
        // delta_q_params:
        fb.bit(0); // delta_q_present = 0
        // loop_filter_params:
        fb.bits(0, 6); // loop_filter_level[0] = 0
        fb.bits(0, 6); // loop_filter_level[1] = 0
        fb.bits(0, 3); // loop_filter_sharpness = 0
        fb.bit(0); // loop_filter_delta_enabled = 0
        // tx_mode:
        fb.bit(0); // tx_mode_select = 0
        // reduced_tx_set:
        fb.bit(0);
        let mut frame_body = fb.bytes();
        // 2 karo: ilk karonun 1 baytlık `tile_size_minus_1` değeri `0xFF` (256 bayt),
        // ancak toplam kalan veri yalnız 2 bayt -> `split_tile_groups` sınır aşımı!
        frame_body.extend_from_slice(&[0xFF, 0x80, 0x80]);
        let mut pkt = vec![0x12, 0x00];
        pkt.extend_from_slice(&seq_2tile);
        pkt.extend_from_slice(&wrap_obu(6, &frame_body));
        pkt
    };
    let run_bad_tile_size = |api: Dav1dApi, mfd: c_int| -> (c_int, c_int, c_int) {
        // SAFETY: Bağımsız `Dav1dContext` üzerinde `max_frame_delay` davranışı sınanır.
        unsafe {
            let mut s = Dav1dSettings::default();
            (api.default_settings)(&mut s);
            s.n_threads = 2;
            s.max_frame_delay = mfd;
            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!((api.open)(&mut c, &s), 0);
            let mut d = Dav1dData::default();
            let dst = (api.data_create)(&mut d, bad_tile_size_pkt.len());
            ptr::copy_nonoverlapping(bad_tile_size_pkt.as_ptr(), dst, bad_tile_size_pkt.len());
            let sr = (api.send_data)(c, &mut d);
            if d.sz > 0 {
                (api.data_unref)(&mut d);
            }
            let mut pic = Dav1dPicture::default();
            let gr1 = (api.get_picture)(c, &mut pic);
            if gr1 == 0 {
                (api.picture_unref)(&mut pic);
            }
            let gr2 = (api.get_picture)(c, &mut pic);
            if gr2 == 0 {
                (api.picture_unref)(&mut pic);
            }
            (api.close)(&mut c);
            (sr, gr1, gr2)
        }
    };
    let got_mfd1 = run_bad_tile_size(av1oxide_api, 1);
    let got_mfd2 = run_bad_tile_size(av1oxide_api, 2);
    assert_eq!(
        got_mfd1,
        (dav1d_err(EINVAL), dav1d_err(EAGAIN), dav1d_err(EAGAIN)),
        "max_frame_delay=1 iken bozuk tile_size_minus_1 send_data'da -EINVAL döndürmeli"
    );
    assert_eq!(
        got_mfd2,
        (0, dav1d_err(EAGAIN), dav1d_err(EINVAL)),
        "max_frame_delay=2 iken bozuk tile_size_minus_1 send_data'da 0 dönüp drain get_picture'da -EINVAL döndürmeli"
    );
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if let Some(ref_lib) = ref_dav1d.as_ref() {
        assert_eq!(got_mfd1, run_bad_tile_size(ref_lib.api, 1));
        assert_eq!(got_mfd2, run_bad_tile_size(ref_lib.api, 2));
    }

    // 6. C-ABI `dav1d_flush` `event_flags` korumu (`lib.c:389, 517`) ve
    //    `frame_size_limit` `ERANGE` hatasında `cached_error_props` korunumu + günlük formatı
    //    (`obu.c:1310-1316`) + ölümcül olmayan T.35/Metadata/OBU uyarı günlükleri (`obu.c:1471, 1533, 1547`):
    unsafe extern "C" {
        fn vsnprintf(
            s: *mut core::ffi::c_char,
            n: usize,
            format: *const core::ffi::c_char,
            arg: core::ffi::VaList<'_>,
        ) -> c_int;
    }
    unsafe extern "C" fn edge_logger_cb(
        cookie: *mut c_void,
        fmt: *const core::ffi::c_char,
        ap: core::ffi::VaList<'_>,
    ) {
        if !cookie.is_null() && !fmt.is_null() {
            let mut buf = [0u8; 512];
            // SAFETY: `buf` 512 bayt yazılabilir yerel tampondur; `fmt` ve `ap` C variadic
            // `Dav1dLogger.callback` çağrısından geçerli olarak gelir; `cookie` canlı `Mutex<Vec<String>>` adresidir.
            unsafe {
                let n = vsnprintf(
                    buf.as_mut_ptr() as *mut core::ffi::c_char,
                    buf.len(),
                    fmt,
                    ap,
                );
                if n > 0 {
                    let text = CStr::from_ptr(buf.as_ptr() as *const core::ffi::c_char)
                        .to_string_lossy()
                        .into_owned();
                    let logs = &*(cookie as *const Mutex<Vec<String>>);
                    logs.lock().unwrap().push(text);
                }
            }
        }
    }
    let run_edge_cases = |api: Dav1dApi| -> (u32, c_int, i64, Vec<String>) {
        let logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
        // SAFETY: Bağımsız `Dav1dContext` üzerinde `event_flags`, `ERANGE` ve `logger` sınanır.
        unsafe {
            let mut s = Dav1dSettings::default();
            (api.default_settings)(&mut s);
            s.n_threads = 1;
            s.max_frame_delay = 1;
            s.frame_size_limit = 100; // 432x240 = 103680 > 100 -> ERANGE
            s.logger.cookie = (&logs as *const Mutex<Vec<String>>) as *mut c_void;
            s.logger.callback = edge_logger_cb as *const c_void;
            let mut c: *mut Dav1dContext = ptr::null_mut();
            assert_eq!((api.open)(&mut c, &s), 0);

            // (a) İlk pakette geçerli kare çözülüp `ready_pic` oluşsun, ardından `get_picture`
            //     çağrılmadan `dav1d_flush` çağrılsın -> `DAV1D_EVENT_FLAG_NEW_SEQUENCE` korunmalı!
            let mut s_no_lim = s;
            s_no_lim.frame_size_limit = 0;
            let mut c_fl: *mut Dav1dContext = ptr::null_mut();
            assert_eq!((api.open)(&mut c_fl, &s_no_lim), 0);
            let pkt0 = SAMPLE_OBU_PACKETS[0];
            let mut d0 = Dav1dData::default();
            let p0 = (api.data_create)(&mut d0, pkt0.len());
            ptr::copy_nonoverlapping(pkt0.as_ptr(), p0, pkt0.len());
            assert_eq!((api.send_data)(c_fl, &mut d0), 0);
            (api.flush)(c_fl);
            let mut flags_after_flush: u32 = 0;
            assert_eq!((api.get_event_flags)(c_fl, &mut flags_after_flush), 0);
            (api.close)(&mut c_fl);

            // (b) `c` üzerinde önce `EINVAL` hatası (`timestamp = 111`) üretip `cached_error_props`
            //     dolduralım, ardından `frame_size_limit` (`ERANGE`, `timestamp = 222`) paketi
            //     gönderelim -> `cached_error_props.timestamp` `111` olarak kalmalı ve
            //     `"Frame size 128x64 exceeds limit 100\n"` günlüklenmeli.
            let bad_obu = [0x0Au8, 0x7F, 0xFF];
            let mut d_einval = Dav1dData::default();
            let p_einval = (api.data_create)(&mut d_einval, bad_obu.len());
            ptr::copy_nonoverlapping(bad_obu.as_ptr(), p_einval, bad_obu.len());
            d_einval.m.timestamp = 111;
            assert_eq!((api.send_data)(c, &mut d_einval), dav1d_err(EINVAL));
            (api.data_unref)(&mut d_einval);

            // Uyarı OBU'ları + `frame_size_limit` aşan 128x64 paketi:
            // - Bozuk T.35 (`[0x04, 0xB5, 0x00]` -> `0x80` yok)
            // - Bilinmeyen Metadata OBU (`meta_type = 32` -> `[0x20, 0x80]`)
            // - Yok sayılan Metadata OBU (`meta_type = 6` -> `[0x06, 0x80]`, günlük yazmamalı)
            // - Bilinmeyen OBU tipi `9` (`Reserved(9)`, 3 bayt yük)
            // - `OBU_PADDING` (`15`, günlük yazmamalı)
            let mut combo_pkt = Vec::new();
            combo_pkt.extend_from_slice(&wrap_obu(5, &[0x04, 0xB5, 0x00]));
            combo_pkt.extend_from_slice(&wrap_obu(5, &[0x20, 0x80]));
            combo_pkt.extend_from_slice(&wrap_obu(5, &[0x06, 0x80]));
            combo_pkt.extend_from_slice(&wrap_obu(9, &[0x01, 0x02, 0x03]));
            combo_pkt.extend_from_slice(&wrap_obu(15, &[0x00, 0x00]));
            combo_pkt.extend_from_slice(&bad_tile_size_pkt);

            let mut d_erange = Dav1dData::default();
            let p_erange = (api.data_create)(&mut d_erange, combo_pkt.len());
            ptr::copy_nonoverlapping(combo_pkt.as_ptr(), p_erange, combo_pkt.len());
            d_erange.m.timestamp = 222;
            let erange_rc = (api.send_data)(c, &mut d_erange);
            (api.data_unref)(&mut d_erange);

            let mut err_props = Dav1dDataProps::default();
            assert_eq!((api.get_frame_error_data_props)(c, &mut err_props), 0);
            let ts_after_erange = err_props.timestamp;
            (api.data_props_unref)(&mut err_props);
            (api.close)(&mut c);

            let collected = logs.lock().unwrap().clone();
            (flags_after_flush, erange_rc, ts_after_erange, collected)
        }
    };

    let (got_flags, got_erange_rc, got_ts, got_logs) = run_edge_cases(av1oxide_api);
    assert_eq!(
        got_flags, DAV1D_EVENT_FLAG_NEW_SEQUENCE,
        "dav1d_flush event_flags alanını sıfırlamamalı"
    );
    assert_eq!(got_erange_rc, dav1d_err(ERANGE));
    assert_eq!(
        got_ts, 111,
        "ERANGE hatası cached_error_props değerini ezmemeli"
    );
    assert_eq!(
        got_logs,
        vec![
            "Error parsing OBU data\n".to_string(),
            "Malformed ITU-T T.35 metadata message format\n".to_string(),
            "Unknown Metadata OBU type 32\n".to_string(),
            "Unknown OBU type 9 of size 3\n".to_string(),
            "Frame size 128x64 exceeds limit 100\n".to_string(),
        ]
    );
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if let Some(ref_lib) = ref_dav1d.as_ref() {
        let (exp_flags, exp_erange_rc, exp_ts, exp_logs) = run_edge_cases(ref_lib.api);
        assert_eq!(got_flags, exp_flags);
        assert_eq!(got_erange_rc, exp_erange_rc);
        assert_eq!(got_ts, exp_ts);
        assert_eq!(got_logs, exp_logs);
    }
}
