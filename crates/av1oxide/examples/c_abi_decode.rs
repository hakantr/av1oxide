//! `av1oxide` `libdav1d` (`7.0.0`) uyumlu C-ABI kullanım örneği (`v0.3.2`).
//!
//! Çalıştırma:
//!   cargo run --release -p av1oxide --example c_abi_decode

use core::ffi::CStr;
use core::ptr;

use av1oxide::capi::{
    DAV1D_EVENT_FLAG_NEW_SEQUENCE, Dav1dContext, Dav1dData, Dav1dEventFlags, Dav1dPicture,
    Dav1dSequenceHeader, Dav1dSettings, EAGAIN, dav1d_close, dav1d_data_create,
    dav1d_default_settings, dav1d_err, dav1d_get_event_flags, dav1d_get_picture, dav1d_open,
    dav1d_parse_sequence_header, dav1d_picture_unref, dav1d_send_data, dav1d_version,
    dav1d_version_api,
};

/// Gömülü 2 paketlik (1×1, 8-bit, 4:2:0) örnek AV1 OBU akışı.
const EMBEDDED_OBU_PACKETS: &[&[u8]] = &[
    // Paket 0: Temporal Delimiter + Sequence Header + Keyframe (38 bayt)
    &[
        0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
        0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
        0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ],
    // Paket 1: Temporal Delimiter + Inter Frame (31 bayt)
    &[
        0x12, 0x00, 0x32, 0x1B, 0x30, 0x02, 0x01, 0x00, 0xA7, 0x2E, 0x47, 0xB0, 0x80, 0x00, 0x02,
        0x40, 0x00, 0x00, 0x80, 0x00, 0x00, 0x03, 0x5A, 0xF0, 0x1B, 0x0B, 0xA1, 0xEF, 0xD4, 0x62,
        0xC0,
    ],
];

fn main() {
    // SAFETY: C-ABI örnek çağrılarındaki tüm işaretçiler yerel yığın değişkenlerinden
    // veya statik OBU dilimlerinden üretilir ve yaşam süreleri bu blok içinde geçerlidir.
    unsafe {
        let ver_str = CStr::from_ptr(dav1d_version()).to_string_lossy();
        let api_ver = dav1d_version_api();
        println!(
            "av1oxide C-ABI: dav1d_version={} (API {}.{}.{})",
            ver_str,
            (api_ver >> 16) & 0xFF,
            (api_ver >> 8) & 0xFF,
            api_ver & 0xFF
        );

        // 1. Sequence Header OBU'sunu bağımsız olarak ayrıştır.
        let mut seq_hdr = core::mem::MaybeUninit::<Dav1dSequenceHeader>::zeroed();
        let rc = dav1d_parse_sequence_header(
            seq_hdr.as_mut_ptr(),
            EMBEDDED_OBU_PACKETS[0].as_ptr(),
            EMBEDDED_OBU_PACKETS[0].len(),
        );
        assert_eq!(rc, 0, "dav1d_parse_sequence_header başarılı olmalı");
        let seq_hdr = seq_hdr.assume_init();
        println!(
            "Sequence Header: profile={}, max={}x{}, layout={}, hbd={}",
            seq_hdr.profile, seq_hdr.max_width, seq_hdr.max_height, seq_hdr.layout, seq_hdr.hbd
        );

        // 2. Varsayılan ayarları başlat ve çözücüyü aç.
        let mut settings = Dav1dSettings::default();
        dav1d_default_settings(&mut settings);
        settings.n_threads = 2;
        settings.max_frame_delay = 1;

        let mut ctx: *mut Dav1dContext = ptr::null_mut();
        assert_eq!(dav1d_open(&mut ctx, &settings), 0);
        assert!(!ctx.is_null());

        let mut decoded = 0usize;
        let mut pic = Dav1dPicture::default();

        // 3. Paketleri `dav1d_data_create` ile oluşturup `dav1d_send_data` / `dav1d_get_picture` ile çöz.
        for (pts, payload) in EMBEDDED_OBU_PACKETS.iter().enumerate() {
            let mut data = Dav1dData::default();
            let dst = dav1d_data_create(&mut data, payload.len());
            assert!(!dst.is_null());
            ptr::copy_nonoverlapping(payload.as_ptr(), dst, payload.len());
            data.m.timestamp = pts as i64;

            assert_eq!(dav1d_send_data(ctx, &mut data), 0);
            assert!(data.data.is_null(), "tüketilen Dav1dData sıfırlanmalı");

            loop {
                let res = dav1d_get_picture(ctx, &mut pic);
                if res == dav1d_err(EAGAIN) {
                    break;
                }
                assert_eq!(res, 0);
                let y_ptr = pic.data[0] as *const u8;
                assert!(!y_ptr.is_null());
                assert_eq!((y_ptr as usize) % 64, 0, "Y düzlemi 64-bayt hizalı olmalı");
                let y0 = *y_ptr;
                let mut flags: Dav1dEventFlags = 0;
                assert_eq!(dav1d_get_event_flags(ctx, &mut flags), 0);
                println!(
                    "C-ABI kare #{decoded}: {}x{}, bpc={}, pts={}, Y[0,0]={y0}, new_seq={}",
                    pic.p.w,
                    pic.p.h,
                    pic.p.bpc,
                    pic.m.timestamp,
                    (flags & DAV1D_EVENT_FLAG_NEW_SEQUENCE) != 0
                );
                dav1d_picture_unref(&mut pic);
                decoded += 1;
            }
        }

        // 4. Akış sonu (EOS) boşaltma döngüsü.
        while dav1d_get_picture(ctx, &mut pic) == 0 {
            dav1d_picture_unref(&mut pic);
            decoded += 1;
        }

        dav1d_close(&mut ctx);
        assert!(ctx.is_null(), "dav1d_close *c_out = NULL yapmalı");
        assert_eq!(decoded, 2, "toplam 2 kare çözülmeli");
    }
}
