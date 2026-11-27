//! `libdav1d` (`1.5.4`, API `7.0.0`) 19 C-ABI giriş noktası ve FFI bellek yönetimi.
//!
//! Ham işaretçi işlemleri yalnızca bu sınır dosyasında toplanmıştır; tip tanımları
//! (`capi::types`) ve başlık/ayar dönüşümleri (`capi::convert`) `%100 #![forbid(unsafe_code)]`
//! altındadır.

#![allow(unsafe_code)]

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::api::{Decoder, OutputCapiMeta, Packet};
use crate::capi::convert::{
    bpc_to_bit_depth, chroma_to_pixel_layout, error_to_dav1d, film_grain_from_c,
    frame_delay_from_settings, frame_header_to_c, has_grain_c, parse_sequence_header_bytes,
    pixel_layout_to_chroma, seq_header_to_c, settings_to_config,
};
use crate::capi::types::{
    DAV1D_API_VERSION_MAJOR, DAV1D_API_VERSION_MINOR, DAV1D_API_VERSION_PATCH,
    DAV1D_PICTURE_ALIGNMENT, Dav1dContentLightLevel, Dav1dContext, Dav1dData, Dav1dDataProps,
    Dav1dEventFlags, Dav1dFrameHeader, Dav1dITUTT35, Dav1dMasteringDisplay, Dav1dPicAllocator,
    Dav1dPicture, Dav1dPictureParameters, Dav1dRef, Dav1dSequenceHeader, Dav1dSettings, EAGAIN,
    EINVAL, ENOMEM, dav1d_err,
};
use crate::frame::{Frame, PlaneType};

/// 64-bayt hizalı bellek bloğu (`DAV1D_PICTURE_ALIGNMENT = 64`).
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct Aligned64([u8; 64]);

impl Default for Aligned64 {
    fn default() -> Self {
        Self([0u8; 64])
    }
}

/// `Dav1dRef` tarafından yönetilen referans sayımlı kaynak türleri.
#[allow(dead_code, clippy::large_enum_variant)]
pub(crate) enum RefPayload {
    /// `dav1d_data_create` tarafından ayrılan 64-bayt hizalı ve 64-bayt dolgulu veri tamponu.
    DataBuffer(Vec<Aligned64>),
    /// `dav1d_data_wrap` veya `dav1d_data_wrap_user_data` tarafından sarılan dış bellek.
    WrappedData {
        ptr: *const u8,
        free_cb: extern "C" fn(*const u8, *mut c_void),
        cookie: *mut c_void,
    },
    /// Varsayılan resim ayırıcının 64-bayt hizalı Y/U/V düzlem tamponları.
    DefaultPicturePlanes([Vec<Aligned64>; 3]),
    /// Özel `Dav1dPicAllocator` tarafından ayrılan resim tamponu.
    CustomPicture {
        pic_snapshot: Dav1dPicture,
        release_cb: extern "C" fn(*mut Dav1dPicture, *mut c_void),
        cookie: *mut c_void,
    },
    /// `Dav1dPicture.seq_hdr` sahibi.
    SeqHdr(Box<Dav1dSequenceHeader>),
    /// `Dav1dPicture.frame_hdr` sahibi.
    FrameHdr(Box<Dav1dFrameHeader>),
    /// `Dav1dPicture.content_light` sahibi.
    ContentLight(Box<Dav1dContentLightLevel>),
    /// `Dav1dPicture.mastering_display` sahibi.
    MasteringDisplay(Box<Dav1dMasteringDisplay>),
    /// `Dav1dPicture.itut_t35` dizisi ve yük tamponlarının sahibi.
    ItutT35 {
        entries: Vec<Dav1dITUTT35>,
        _payloads: Vec<Vec<u8>>,
    },
}

/// `Dav1dRef` iç yapısı; son referans düştüğünde (`Drop`) kaynağı serbest bırakır.
pub(crate) struct RefInner(pub(crate) RefPayload);

impl Drop for RefInner {
    fn drop(&mut self) {
        match &mut self.0 {
            RefPayload::WrappedData {
                ptr,
                free_cb,
                cookie,
            } => {
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    free_cb(*ptr, *cookie);
                }));
            }
            RefPayload::CustomPicture {
                pic_snapshot,
                release_cb,
                cookie,
            } => {
                let pic_ptr = pic_snapshot as *mut Dav1dPicture;
                let cookie_val = *cookie;
                let cb = *release_cb;
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    cb(pic_ptr, cookie_val);
                }));
            }
            RefPayload::DataBuffer(_)
            | RefPayload::DefaultPicturePlanes(_)
            | RefPayload::SeqHdr(_)
            | RefPayload::FrameHdr(_)
            | RefPayload::ContentLight(_)
            | RefPayload::MasteringDisplay(_)
            | RefPayload::ItutT35 { .. } => {}
        }
    }
}

/// `Dav1dContext` iç durumu.
pub(crate) struct ContextInner {
    pub(crate) decoder: Decoder,
    pub(crate) allocator: Dav1dPicAllocator,
    pub(crate) all_layers: bool,
    pub(crate) cached_pic: Option<(Frame, OutputCapiMeta)>,
    pub(crate) drain: bool,
    pub(crate) event_flags: Dav1dEventFlags,
    pub(crate) cached_error_props: Dav1dDataProps,
    pub(crate) next_packet_tag: u64,
    pub(crate) packet_props: VecDeque<(u64, Dav1dDataProps)>,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        data_props_unref_value(&mut self.cached_error_props);
        while let Some((_, mut props)) = self.packet_props.pop_front() {
            data_props_unref_value(&mut props);
        }
    }
}

#[allow(clippy::arc_with_non_send_sync)]
fn ref_into_raw(payload: RefPayload) -> *mut Dav1dRef {
    let arc = Arc::new(Dav1dRef {
        inner: RefInner(payload),
    });
    Arc::into_raw(arc) as *mut Dav1dRef
}

fn ref_inc(ptr: *mut Dav1dRef) -> *mut Dav1dRef {
    if !ptr.is_null() {
        // SAFETY: (1) ISA koşulu yoktur (platformdan bağımsız C-ABI referans sayımı).
        // (2) sınır: `ptr` null değildir ve `ref_into_raw` (`Arc::into_raw`) tarafından üretilmiştir.
        // (3) hizalama/takma ad: `Arc::increment_strong_count` atomik referans sayacını güvenle artırır.
        unsafe {
            Arc::increment_strong_count(ptr as *const Dav1dRef);
        }
    }
    ptr
}

fn ref_dec(slot: &mut *mut Dav1dRef) {
    let ptr = *slot;
    *slot = core::ptr::null_mut();
    if !ptr.is_null() {
        // SAFETY: (1) ISA koşulu yoktur (platformdan bağımsız C-ABI referans sayımı).
        // (2) sınır: `ptr` null değildir ve `ref_into_raw` / `ref_inc` ile eşleşen canlı bir `Arc` referansıdır.
        // (3) hizalama/takma ad: `*slot` sıfırlanarak çift serbest bırakma önlenmiştir.
        unsafe {
            drop(Arc::from_raw(ptr as *const Dav1dRef));
        }
    }
}

fn clone_data_props(src: &Dav1dDataProps) -> Dav1dDataProps {
    let mut copy = *src;
    copy.user_data.r#ref = ref_inc(src.user_data.r#ref);
    copy
}

fn data_props_unref_value(props: &mut Dav1dDataProps) {
    let mut user_ref = props.user_data.r#ref;
    *props = Dav1dDataProps::default();
    ref_dec(&mut user_ref);
}

fn data_unref_value(data: &mut Dav1dData) {
    let mut data_ref = data.r#ref;
    let mut user_ref = data.m.user_data.r#ref;
    *data = Dav1dData::default();
    ref_dec(&mut data_ref);
    ref_dec(&mut user_ref);
}

fn picture_unref_value(pic: &mut Dav1dPicture) {
    let mut pic_ref = pic.r#ref;
    let mut seq_ref = pic.seq_hdr_ref;
    let mut frame_ref = pic.frame_hdr_ref;
    let mut cll_ref = pic.content_light_ref;
    let mut mdcv_ref = pic.mastering_display_ref;
    let mut itut_ref = pic.itut_t35_ref;
    let mut user_ref = pic.m.user_data.r#ref;
    *pic = Dav1dPicture::default();
    ref_dec(&mut pic_ref);
    ref_dec(&mut seq_ref);
    ref_dec(&mut frame_ref);
    ref_dec(&mut cll_ref);
    ref_dec(&mut mdcv_ref);
    ref_dec(&mut itut_ref);
    ref_dec(&mut user_ref);
}

fn aligned64_vec(byte_len: usize) -> Option<Vec<Aligned64>> {
    let total_bytes = byte_len.checked_add(DAV1D_PICTURE_ALIGNMENT)?;
    let blocks = total_bytes.div_ceil(64).max(1);
    let mut v = Vec::new();
    v.try_reserve_exact(blocks).ok()?;
    v.resize(blocks, Aligned64::default());
    Some(v)
}

fn copy_frame_to_custom_pic(frame: &Frame, pic: &mut Dav1dPicture) -> Result<(), c_int> {
    let plane_types = [PlaneType::Y, PlaneType::U, PlaneType::V];
    for (idx, plane_type) in plane_types.iter().copied().enumerate() {
        let Some(src_plane) = frame.plane(plane_type) else {
            continue;
        };
        let dst_base = pic.data[idx] as *mut u8;
        if dst_base.is_null() {
            return Err(dav1d_err(EINVAL));
        }
        let dst_stride = pic.stride[usize::from(idx > 0)];
        let row_bytes = src_plane.layout().row_bytes();
        let h = src_plane.height();
        if dst_stride.unsigned_abs() < row_bytes {
            return Err(dav1d_err(EINVAL));
        }
        for y in 0..h {
            let Some(src_row) = src_plane.row(y) else {
                return Err(dav1d_err(EINVAL));
            };
            let offset = (y as isize)
                .checked_mul(dst_stride)
                .ok_or(dav1d_err(EINVAL))?;
            // SAFETY: (1) ISA koşulu yoktur (düzlem satır kopyası).
            // (2) sınır: özel `alloc_picture_callback` sözleşmesiyle `dst_base` en az `h` satır ve `|dst_stride| >= row_bytes` kapasitelidir.
            // (3) hizalama/takma ad: `u8` kopyası hizalama gerektirmez ve kaynak/hedef düzlemler ayrıktır.
            unsafe {
                let dst_row = dst_base.offset(offset);
                core::ptr::copy_nonoverlapping(src_row.as_ptr(), dst_row, row_bytes);
            }
        }
    }
    Ok(())
}

fn allocate_default_picture_planes(
    frame: &Frame,
    pic: &mut Dav1dPicture,
) -> Result<[Vec<Aligned64>; 3], c_int> {
    let mut buffers: [Vec<Aligned64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let plane_types = [PlaneType::Y, PlaneType::U, PlaneType::V];
    for (idx, plane_type) in plane_types.iter().copied().enumerate() {
        let Some(src_plane) = frame.plane(plane_type) else {
            pic.data[idx] = core::ptr::null_mut();
            continue;
        };
        let row_bytes = src_plane.layout().row_bytes();
        let stride = (row_bytes + (DAV1D_PICTURE_ALIGNMENT - 1)) & !(DAV1D_PICTURE_ALIGNMENT - 1);
        let h = src_plane.height();
        // `dav1d` sözleşmesi: düzlem yüksekliği 128 piksel katına tamamlanabilecek veya en az
        // `stride * h + 64` bayt dolgulu olacak şekilde 64-bayt hizalı ayrılır.
        let total = stride.checked_mul(h).ok_or(dav1d_err(ENOMEM))?;
        let mut buf = aligned64_vec(total).ok_or(dav1d_err(ENOMEM))?;
        let base_ptr = buf.as_mut_ptr() as *mut u8;
        let total_slice_len = buf.len() * 64;
        // SAFETY: (1) ISA koşulu yoktur (varsayılan düzlem tamponu doldurma).
        // (2) sınır: `buf` `total + 64` bayttan büyük veya eşit olacak şekilde `Aligned64` bloklarıyla ayrılmıştır.
        // (3) hizalama/takma ad: `Aligned64` `#[repr(C, align(64))]` olduğundan `base_ptr` 64-bayt hizalıdır ve tek sahibi `buf`'tır.
        let dst_bytes = unsafe { core::slice::from_raw_parts_mut(base_ptr, total_slice_len) };
        for y in 0..h {
            if let Some(src_row) = src_plane.row(y) {
                let start = y * stride;
                dst_bytes[start..start + row_bytes].copy_from_slice(src_row);
            }
        }
        pic.data[idx] = base_ptr as *mut c_void;
        if idx == 0 {
            pic.stride[0] = stride as isize;
        } else if idx == 1 {
            pic.stride[1] = stride as isize;
        }
        buffers[idx] = buf;
    }
    Ok(buffers)
}

fn attach_picture_metadata(pic: &mut Dav1dPicture, meta: &OutputCapiMeta) {
    if let Some(seq) = &meta.seq_hdr {
        let mut boxed = Box::new(seq_header_to_c(seq));
        let ptr = &mut *boxed as *mut Dav1dSequenceHeader;
        let r = ref_into_raw(RefPayload::SeqHdr(boxed));
        pic.seq_hdr = ptr;
        pic.seq_hdr_ref = r;
    }
    if let Some(fh) = &meta.frame_hdr {
        let mut boxed = Box::new(frame_header_to_c(fh));
        let ptr = &mut *boxed as *mut Dav1dFrameHeader;
        let r = ref_into_raw(RefPayload::FrameHdr(boxed));
        pic.frame_hdr = ptr;
        pic.frame_hdr_ref = r;
    }
    if let Some((max_cll, max_fall)) = meta.content_light {
        let mut boxed = Box::new(Dav1dContentLightLevel {
            max_content_light_level: max_cll,
            max_frame_average_light_level: max_fall,
        });
        let ptr = &mut *boxed as *mut Dav1dContentLightLevel;
        let r = ref_into_raw(RefPayload::ContentLight(boxed));
        pic.content_light = ptr;
        pic.content_light_ref = r;
    }
    if let Some(md) = &meta.mastering_display {
        let mut boxed = Box::new(Dav1dMasteringDisplay {
            primaries: md.primaries,
            white_point: md.white_point,
            max_luminance: md.max_luminance,
            min_luminance: md.min_luminance,
        });
        let ptr = &mut *boxed as *mut Dav1dMasteringDisplay;
        let r = ref_into_raw(RefPayload::MasteringDisplay(boxed));
        pic.mastering_display = ptr;
        pic.mastering_display_ref = r;
    }
    if !meta.itut_t35.is_empty() {
        let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(meta.itut_t35.len());
        let mut entries: Vec<Dav1dITUTT35> = Vec::with_capacity(meta.itut_t35.len());
        for item in &meta.itut_t35 {
            let mut p = item.payload.clone();
            let ptr = if p.is_empty() {
                core::ptr::null_mut()
            } else {
                p.as_mut_ptr()
            };
            let len = p.len();
            payloads.push(p);
            entries.push(Dav1dITUTT35 {
                country_code: item.country_code,
                country_code_extension_byte: item.country_code_extension_byte,
                payload_size: len,
                payload: ptr,
            });
        }
        let n = entries.len();
        let entries_ptr = entries.as_mut_ptr();
        let r = ref_into_raw(RefPayload::ItutT35 {
            entries,
            _payloads: payloads,
        });
        pic.itut_t35 = entries_ptr;
        pic.n_itut_t35 = n;
        pic.itut_t35_ref = r;
    }
}

fn fill_output_picture(
    ctx: &ContextInner,
    out: &mut Dav1dPicture,
    frame: &Frame,
    meta: &OutputCapiMeta,
    props: Dav1dDataProps,
) -> Result<(), c_int> {
    picture_unref_value(out);
    out.p = Dav1dPictureParameters {
        w: frame.width() as c_int,
        h: frame.height() as c_int,
        layout: chroma_to_pixel_layout(frame.chroma_format()),
        bpc: frame.bit_depth().bits() as c_int,
    };

    if let (Some(alloc_cb), Some(release_cb)) = (
        ctx.allocator.alloc_picture_callback,
        ctx.allocator.release_picture_callback,
    ) {
        let rc = alloc_cb(out as *mut Dav1dPicture, ctx.allocator.cookie);
        if rc < 0 {
            let mut p = props;
            data_props_unref_value(&mut p);
            *out = Dav1dPicture::default();
            return Err(rc);
        }
        let snapshot = *out;
        let pic_ref = ref_into_raw(RefPayload::CustomPicture {
            pic_snapshot: snapshot,
            release_cb,
            cookie: ctx.allocator.cookie,
        });
        out.r#ref = pic_ref;
        if let Err(e) = copy_frame_to_custom_pic(frame, out) {
            let mut p = props;
            data_props_unref_value(&mut p);
            picture_unref_value(out);
            return Err(e);
        }
    } else {
        match allocate_default_picture_planes(frame, out) {
            Ok(planes) => {
                out.r#ref = ref_into_raw(RefPayload::DefaultPicturePlanes(planes));
            }
            Err(e) => {
                let mut p = props;
                data_props_unref_value(&mut p);
                *out = Dav1dPicture::default();
                return Err(e);
            }
        }
    }

    out.m = props;
    attach_picture_metadata(out, meta);
    Ok(())
}

fn frame_from_c_picture(pic: &Dav1dPicture) -> Result<Frame, c_int> {
    if pic.p.w <= 0 || pic.p.h <= 0 {
        return Err(dav1d_err(EINVAL));
    }
    let chroma = pixel_layout_to_chroma(pic.p.layout).ok_or(dav1d_err(EINVAL))?;
    let bd = bpc_to_bit_depth(pic.p.bpc).ok_or(dav1d_err(EINVAL))?;
    let mut frame = Frame::new(pic.p.w as usize, pic.p.h as usize, chroma, bd)
        .map_err(|_| dav1d_err(ENOMEM))?;
    let plane_types = [PlaneType::Y, PlaneType::U, PlaneType::V];
    for (idx, plane_type) in plane_types.iter().copied().enumerate() {
        let Some(dst_plane) = frame.plane_mut(plane_type) else {
            continue;
        };
        let src_base = pic.data[idx] as *const u8;
        if src_base.is_null() {
            return Err(dav1d_err(EINVAL));
        }
        let src_stride = pic.stride[usize::from(idx > 0)];
        let row_bytes = dst_plane.layout().row_bytes();
        let h = dst_plane.height();
        if src_stride.unsigned_abs() < row_bytes {
            return Err(dav1d_err(EINVAL));
        }
        for y in 0..h {
            let offset = (y as isize)
                .checked_mul(src_stride)
                .ok_or(dav1d_err(EINVAL))?;
            // SAFETY: (1) ISA koşulu yoktur (`Dav1dPicture` düzlem satırı okuma).
            // (2) sınır: `pic` geçerli bir `Dav1dPicture` olduğundan her satırda `row_bytes` bayt okunabilir.
            // (3) hizalama/takma ad: `u8` dilimi hizalama gerektirmez ve `dst_plane` ayrı bir tahsistir.
            unsafe {
                let src_row = core::slice::from_raw_parts(src_base.offset(offset), row_bytes);
                if let Some(dst_row) = dst_plane.row_mut(y) {
                    dst_row.copy_from_slice(src_row);
                }
            }
        }
    }
    Ok(frame)
}

/// Kütüphane sürüm dizesini döndürür (`"0.5.1"`).
///
/// # Safety
///
/// Bu fonksiyon argüman almaz ve her zaman statik ömürlü, sonu sıfır (`NUL`) ile biten
/// geçerli bir C karakter dizesi işaretçisi döndürür.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_version() -> *const c_char {
    c"0.5.1".as_ptr()
}

/// `libdav1d` C-ABI sürüm numarasını (`0x0007_0000`) döndürür.
///
/// # Safety
///
/// Bu fonksiyon argüman almaz ve her zaman güvenle çağrılabilir.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_version_api() -> c_uint {
    (DAV1D_API_VERSION_MAJOR << 16) | (DAV1D_API_VERSION_MINOR << 8) | DAV1D_API_VERSION_PATCH
}

/// `Dav1dSettings` yapısını varsayılan değerlerle başlatır.
///
/// # Safety
///
/// `s` geçerli ve yazılabilir bir `Dav1dSettings` işaretçisi (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_default_settings(s: *mut Dav1dSettings) {
    if s.is_null() {
        return;
    }
    // SAFETY: (1) ISA koşulu yoktur (C yapı başlatma).
    // (2) sınır: `s` null kontrolünden geçmiştir ve tek bir `Dav1dSettings` nesnesini gösterir.
    // (3) hizalama/takma ad: `Dav1dSettings` `#[repr(C)]` düzenindedir ve çağrı süresince münhasıran yazılır.
    unsafe {
        *s = Dav1dSettings::default();
    }
}

/// Verilen `Dav1dSettings` için etkin kare gecikmesini (`1..=max_frame_delay`) döndürür.
///
/// # Safety
///
/// `s` geçerli bir `Dav1dSettings` işaretçisi (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_get_frame_delay(s: *const Dav1dSettings) -> c_int {
    if s.is_null() {
        return dav1d_err(EINVAL);
    }
    // SAFETY: (1) ISA koşulu yoktur (`Dav1dSettings` okuma).
    // (2) sınır: `s` null değildir ve çağıran tarafından geçerli bir `Dav1dSettings` olarak sağlanmıştır.
    // (3) hizalama/takma ad: salt-okunur referans yalnız bu kapsamda kullanılır.
    let settings = unsafe { &*s };
    match frame_delay_from_settings(settings) {
        Ok(delay) => delay,
        Err(code) => code,
    }
}

/// Yeni bir `Dav1dContext` çözücü örneği oluşturur ve `*c_out` içine yazar.
///
/// # Safety
///
/// `c_out` ve `s` geçerli işaretçiler (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_open(
    c_out: *mut *mut Dav1dContext,
    s: *const Dav1dSettings,
) -> c_int {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if c_out.is_null() || s.is_null() {
            if !c_out.is_null() {
                // SAFETY: (1) ISA koşulu yoktur (`c_out` sıfırlama).
                // (2) sınır: `c_out` null değildir.
                // (3) hizalama/takma ad: `*c_out` işaretçi boyutunda hizalıdır.
                unsafe {
                    *c_out = core::ptr::null_mut();
                }
            }
            return dav1d_err(EINVAL);
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dSettings` okuma ve `c_out` sıfırlama).
        // (2) sınır: `c_out` ve `s` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `s` salt-okunur, `c_out` yazılabilir ayrık işaretçilerdir.
        let settings = unsafe {
            *c_out = core::ptr::null_mut();
            &*s
        };
        let config = match settings_to_config(settings) {
            Ok(cfg) => cfg,
            Err(code) => return code,
        };
        let decoder = match Decoder::new(config) {
            Ok(dec) => dec,
            Err(err) => return error_to_dav1d(&err),
        };
        let ctx = Box::new(Dav1dContext {
            inner: ContextInner {
                decoder,
                allocator: settings.allocator,
                all_layers: settings.all_layers != 0,
                cached_pic: None,
                drain: false,
                event_flags: 0,
                cached_error_props: Dav1dDataProps::default(),
                next_packet_tag: 1,
                packet_props: VecDeque::new(),
            },
        });
        // SAFETY: (1) ISA koşulu yoktur (`*c_out` ataması).
        // (2) sınır: `c_out` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `Box::into_raw(ctx)` geçerli bir `Dav1dContext` işaretçisidir.
        unsafe {
            *c_out = Box::into_raw(ctx);
        }
        0
    }));
    res.unwrap_or(dav1d_err(ENOMEM))
}

/// Ham OBU tamponundan Sequence Header ayrıştırır ve `*out` içine yazar.
///
/// # Safety
///
/// `out` geçerli bir `Dav1dSequenceHeader` işaretçisi ve `buf` en az `sz` baytlık
/// okunabilir bir tampon (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_parse_sequence_header(
    out: *mut Dav1dSequenceHeader,
    buf: *const u8,
    sz: usize,
) -> c_int {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if out.is_null() || buf.is_null() || sz == 0 || sz > usize::MAX / 2 {
            return dav1d_err(EINVAL);
        }
        // SAFETY: (1) ISA koşulu yoktur (OBU bayt dilimi oluşturma).
        // (2) sınır: `buf` null değildir ve `0 < sz <= usize::MAX / 2` doğrulanmıştır.
        // (3) hizalama/takma ad: `u8` dilimi 1-bayt hizalıdır ve ayrıştırma boyunca salt-okunurdur.
        let slice = unsafe { core::slice::from_raw_parts(buf, sz) };
        match parse_sequence_header_bytes(slice) {
            Ok(seq_c) => {
                // SAFETY: (1) ISA koşulu yoktur (`*out` yazma).
                // (2) sınır: `out` null değildir ve tek bir `Dav1dSequenceHeader` nesnesini gösterir.
                // (3) hizalama/takma ad: `Dav1dSequenceHeader` `#[repr(C)]` düzenindedir ve `slice` ile örtüşmez.
                unsafe {
                    *out = seq_c;
                }
                0
            }
            Err(code) => code,
        }
    }));
    res.unwrap_or(dav1d_err(EINVAL))
}

/// Çözücüye bir `Dav1dData` OBU paketi gönderir (`0` başarıda `in_data` referansını tüketir).
///
/// # Safety
///
/// `c` ve `in_data` geçerli işaretçiler (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_send_data(c: *mut Dav1dContext, in_data: *mut Dav1dData) -> c_int {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if c.is_null() || in_data.is_null() {
            return dav1d_err(EINVAL);
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dContext` ve `Dav1dData` erişimi).
        // (2) sınır: `c` ve `in_data` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `Dav1dContext` ve `Dav1dData` ayrık tiplerdir.
        let (ctx, data_ref) = unsafe { (&mut (*c).inner, &mut *in_data) };
        if !data_ref.data.is_null() {
            if data_ref.sz == 0 || data_ref.sz > usize::MAX / 2 {
                return dav1d_err(EINVAL);
            }
            ctx.drain = false;
        } else if data_ref.sz != 0 {
            return dav1d_err(EINVAL);
        } else {
            data_unref_value(data_ref);
            return 0;
        }

        // SAFETY: (1) ISA koşulu yoktur (girdi paketi bayt dilimi).
        // (2) sınır: `data_ref.data` null değildir ve `0 < data_ref.sz <= usize::MAX / 2` doğrulanmıştır.
        // (3) hizalama/takma ad: `u8` dilimi hizalama gerektirmez ve `send_packet_capi` süresince salt-okunurdur.
        let payload = unsafe { core::slice::from_raw_parts(data_ref.data, data_ref.sz) };
        let packet_tag = ctx.next_packet_tag;
        ctx.next_packet_tag = ctx.next_packet_tag.wrapping_add(1).max(1);

        let retained_props = clone_data_props(&data_ref.m);
        if ctx.packet_props.len() >= 1024
            && let Some((_, mut old)) = ctx.packet_props.pop_front()
        {
            data_props_unref_value(&mut old);
        }
        ctx.packet_props.push_back((packet_tag, retained_props));

        let packet = Packet::with_timestamp(payload, data_ref.m.timestamp);
        match ctx.decoder.send_packet_capi(&packet, packet_tag) {
            Ok(()) => {
                if !ctx.decoder.has_pending_packet_tag(packet_tag)
                    && let Some(pos) = ctx.packet_props.iter().position(|(t, _)| *t == packet_tag)
                    && let Some((_, mut unused_props)) = ctx.packet_props.remove(pos)
                {
                    data_props_unref_value(&mut unused_props);
                }
                data_unref_value(data_ref);
                0
            }
            Err(err) => {
                data_props_unref_value(&mut ctx.cached_error_props);
                if !ctx.decoder.has_pending_packet_tag(packet_tag)
                    && let Some(pos) = ctx.packet_props.iter().position(|(t, _)| *t == packet_tag)
                    && let Some((_, props)) = ctx.packet_props.remove(pos)
                {
                    ctx.cached_error_props = props;
                } else {
                    ctx.cached_error_props = clone_data_props(&data_ref.m);
                }
                error_to_dav1d(&err)
            }
        }
    }));
    res.unwrap_or(dav1d_err(EINVAL))
}

/// `dav1d` `all_layers == 0` (`src/lib.c:332-351`) uzamsal katman önbelleğini uygulayarak
/// sıradaki C-ABI çıktı karesini döndürür.
fn receive_next_capi_frame(
    ctx: &mut ContextInner,
    drain: bool,
) -> crate::error::Result<Option<(Frame, OutputCapiMeta)>> {
    let max_spatial_id = ctx.decoder.max_spatial_id();
    if ctx.all_layers || max_spatial_id == 0 {
        if let Some(cached) = ctx.cached_pic.take() {
            return Ok(Some(cached));
        }
        return ctx.decoder.receive_frame_capi(drain);
    }

    loop {
        if let Some((_, ref cached_meta)) = ctx.cached_pic {
            let cached_sid = cached_meta.frame_hdr.as_ref().map_or(0, |h| h.spatial_id);
            if cached_sid >= max_spatial_id {
                return Ok(ctx.cached_pic.take());
            }
            match ctx.decoder.receive_frame_capi(drain)? {
                Some((next_frame, next_meta)) => {
                    let next_sid = next_meta.frame_hdr.as_ref().map_or(0, |h| h.spatial_id);
                    let is_new_tu =
                        next_sid <= cached_sid || next_meta.packet_tag != cached_meta.packet_tag;
                    if is_new_tu {
                        let out = ctx.cached_pic.replace((next_frame, next_meta));
                        return Ok(out);
                    }
                    // Aynı zamansal birimde daha yüksek uzamsal katman: eski katmanı at.
                    if let Some((_, dropped_meta)) = ctx.cached_pic.replace((next_frame, next_meta))
                    {
                        ctx.event_flags |= dropped_meta.event_flags;
                        let tag = dropped_meta.packet_tag;
                        let still_held = ctx.decoder.has_pending_packet_tag(tag)
                            || ctx
                                .cached_pic
                                .as_ref()
                                .is_some_and(|(_, m)| m.packet_tag == tag);
                        if !still_held
                            && let Some(pos) = ctx.packet_props.iter().position(|(t, _)| *t == tag)
                            && let Some((_, mut props)) = ctx.packet_props.remove(pos)
                        {
                            data_props_unref_value(&mut props);
                        }
                    }
                }
                None => {
                    // dav1d `src/lib.c:476`: `output_picture_ready(c, c->n_fc == 1)`
                    if drain || ctx.decoder.config().effective_frame_delay() == 1 {
                        return Ok(ctx.cached_pic.take());
                    }
                    return Ok(None);
                }
            }
        } else {
            match ctx.decoder.receive_frame_capi(drain)? {
                Some((frame, meta)) => {
                    let sid = meta.frame_hdr.as_ref().map_or(0, |h| h.spatial_id);
                    if sid >= max_spatial_id {
                        return Ok(Some((frame, meta)));
                    }
                    ctx.cached_pic = Some((frame, meta));
                }
                None => return Ok(None),
            }
        }
    }
}

/// Çözülmüş sıradaki resmi `*out` içine yazar (`0` = kare hazır, `-EAGAIN` = yeni veri gerekli).
///
/// # Safety
///
/// `c` ve `out` geçerli işaretçiler (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_get_picture(c: *mut Dav1dContext, out: *mut Dav1dPicture) -> c_int {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if c.is_null() || out.is_null() {
            return dav1d_err(EINVAL);
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dContext` ve `Dav1dPicture` erişimi).
        // (2) sınır: `c` ve `out` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `Dav1dContext` ve `Dav1dPicture` ayrık nesnelerdir.
        let (ctx, out_ref) = unsafe { (&mut (*c).inner, &mut *out) };
        let drain = ctx.drain;
        ctx.drain = true;

        match receive_next_capi_frame(ctx, drain) {
            Ok(Some((frame, meta))) => {
                ctx.event_flags |= meta.event_flags;
                let still_pending = ctx.decoder.has_pending_packet_tag(meta.packet_tag)
                    || ctx
                        .cached_pic
                        .as_ref()
                        .is_some_and(|(_, m)| m.packet_tag == meta.packet_tag);
                let props = if let Some(pos) = ctx
                    .packet_props
                    .iter()
                    .position(|(tag, _)| *tag == meta.packet_tag)
                {
                    if still_pending {
                        clone_data_props(&ctx.packet_props[pos].1)
                    } else {
                        ctx.packet_props.remove(pos).unwrap().1
                    }
                } else {
                    Dav1dDataProps {
                        timestamp: frame.metadata().timestamp.unwrap_or(i64::MIN),
                        ..Dav1dDataProps::default()
                    }
                };
                match fill_output_picture(ctx, out_ref, &frame, &meta, props) {
                    Ok(()) => 0,
                    Err(code) => code,
                }
            }
            Ok(None) => dav1d_err(EAGAIN),
            Err(err) => {
                if let Some(tag) = ctx.decoder.last_error_packet_tag()
                    && let Some(pos) = ctx.packet_props.iter().position(|(t, _)| *t == tag)
                {
                    let still_pending = ctx.decoder.has_pending_packet_tag(tag);
                    let props = if still_pending {
                        clone_data_props(&ctx.packet_props[pos].1)
                    } else {
                        ctx.packet_props.remove(pos).unwrap().1
                    };
                    data_props_unref_value(&mut ctx.cached_error_props);
                    ctx.cached_error_props = props;
                }
                error_to_dav1d(&err)
            }
        }
    }));
    res.unwrap_or(dav1d_err(EINVAL))
}

/// Daha önce çözülmüş bir resme (`in_pic`) film grain sentezini uygular ve `*out` içine yazar.
///
/// # Safety
///
/// `c`, `out` ve `in_pic` geçerli işaretçiler (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_apply_grain(
    c: *mut Dav1dContext,
    out: *mut Dav1dPicture,
    in_pic: *const Dav1dPicture,
) -> c_int {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if c.is_null() || out.is_null() || in_pic.is_null() {
            return dav1d_err(EINVAL);
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dContext` ve `Dav1dPicture` erişimi).
        // (2) sınır: `c`, `out` ve `in_pic` null değildir.
        // (3) hizalama/takma ad: `in_pic` önce kopyalanarak `out == in_pic` örtüşmesine karşı korunur.
        let (ctx, in_snapshot, out_ref) = unsafe { (&mut (*c).inner, *in_pic, &mut *out) };
        if in_snapshot.frame_hdr.is_null() || in_snapshot.data[0].is_null() {
            return dav1d_err(EINVAL);
        }
        // SAFETY: (1) ISA koşulu yoktur (`frame_hdr` ve `seq_hdr` okuma).
        // (2) sınır: `in_snapshot.frame_hdr` null değildir.
        // (3) hizalama/takma ad: başlık yapıları `#[repr(C)]` düzenindedir ve salt-okunur erişilir.
        let (fh_c, mc_identity) = unsafe {
            let fh = &*in_snapshot.frame_hdr;
            let mc_id = !in_snapshot.seq_hdr.is_null() && (*in_snapshot.seq_hdr).mtrx == 0;
            (*fh, mc_id)
        };

        if !has_grain_c(&fh_c.film_grain.data) {
            picture_unref_value(out_ref);
            *out_ref = in_snapshot;
            out_ref.r#ref = ref_inc(in_snapshot.r#ref);
            out_ref.seq_hdr_ref = ref_inc(in_snapshot.seq_hdr_ref);
            out_ref.frame_hdr_ref = ref_inc(in_snapshot.frame_hdr_ref);
            out_ref.content_light_ref = ref_inc(in_snapshot.content_light_ref);
            out_ref.mastering_display_ref = ref_inc(in_snapshot.mastering_display_ref);
            out_ref.itut_t35_ref = ref_inc(in_snapshot.itut_t35_ref);
            out_ref.m.user_data.r#ref = ref_inc(in_snapshot.m.user_data.r#ref);
            return 0;
        }

        let src_frame = match frame_from_c_picture(&in_snapshot) {
            Ok(f) => f,
            Err(code) => return code,
        };
        let fg_rust = film_grain_from_c(&fh_c.film_grain);
        let grained_frame = match ctx
            .decoder
            .apply_grain_capi(&src_frame, &fg_rust, mc_identity)
        {
            Ok(f) => f,
            Err(err) => return error_to_dav1d(&err),
        };

        let props = clone_data_props(&in_snapshot.m);
        let empty_meta = OutputCapiMeta::default();
        if let Err(code) = fill_output_picture(ctx, out_ref, &grained_frame, &empty_meta, props) {
            return code;
        }
        out_ref.seq_hdr = in_snapshot.seq_hdr;
        out_ref.seq_hdr_ref = ref_inc(in_snapshot.seq_hdr_ref);
        out_ref.frame_hdr = in_snapshot.frame_hdr;
        out_ref.frame_hdr_ref = ref_inc(in_snapshot.frame_hdr_ref);
        out_ref.content_light = in_snapshot.content_light;
        out_ref.content_light_ref = ref_inc(in_snapshot.content_light_ref);
        out_ref.mastering_display = in_snapshot.mastering_display;
        out_ref.mastering_display_ref = ref_inc(in_snapshot.mastering_display_ref);
        out_ref.itut_t35 = in_snapshot.itut_t35;
        out_ref.n_itut_t35 = in_snapshot.n_itut_t35;
        out_ref.itut_t35_ref = ref_inc(in_snapshot.itut_t35_ref);
        0
    }));
    res.unwrap_or(dav1d_err(EINVAL))
}

/// Çözücü durumunu ve bekleyen tüm kareleri temizler (`Decoder::reset` karşılığı).
///
/// # Safety
///
/// `c` geçerli bir `Dav1dContext` işaretçisi (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_flush(c: *mut Dav1dContext) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if c.is_null() {
            return;
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dContext` sıfırlama).
        // (2) sınır: `c` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `c` münhasıran bu çağrıda değiştirilir.
        let ctx = unsafe { &mut (*c).inner };
        ctx.decoder.reset();
        ctx.cached_pic = None;
        ctx.drain = false;
        ctx.event_flags = 0;
        data_props_unref_value(&mut ctx.cached_error_props);
        while let Some((_, mut props)) = ctx.packet_props.pop_front() {
            data_props_unref_value(&mut props);
        }
    }));
}

/// Çözücü örneğini kapatır, tüm belleği serbest bırakır ve `*c_out = NULL` yazar.
///
/// # Safety
///
/// `c_out` geçerli bir `*mut Dav1dContext` işaretçisi (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_close(c_out: *mut *mut Dav1dContext) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if c_out.is_null() {
            return;
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dContext` kapatma ve `*c_out` sıfırlama).
        // (2) sınır: `c_out` null değildir; `*c_out` null ise işlem yapılmaz.
        // (3) hizalama/takma ad: `*c_out` `dav1d_open` içinde `Box::into_raw` ile üretilmiştir ve sıfırlanır.
        unsafe {
            let ptr = *c_out;
            *c_out = core::ptr::null_mut();
            if !ptr.is_null() {
                drop(Box::from_raw(ptr));
            }
        }
    }));
}

/// Biriken `DAV1D_EVENT_FLAG_*` olay bayraklarını `*flags` içine yazar ve bağlamda sıfırlar.
///
/// # Safety
///
/// `c` ve `flags` geçerli işaretçiler (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_get_event_flags(
    c: *mut Dav1dContext,
    flags: *mut Dav1dEventFlags,
) -> c_int {
    if c.is_null() || flags.is_null() {
        return dav1d_err(EINVAL);
    }
    // SAFETY: (1) ISA koşulu yoktur (olay bayraklarını okuma ve temizleme).
    // (2) sınır: `c` ve `flags` null kontrolünden geçmiştir.
    // (3) hizalama/takma ad: `c` ve `flags` ayrık işaretçilerdir.
    unsafe {
        let ctx = &mut (*c).inner;
        *flags = ctx.event_flags;
        ctx.event_flags = 0;
    }
    0
}

/// Son çözme hatasına ait girdi paketinin `Dav1dDataProps` metaverisini `*out` içine taşır.
///
/// # Safety
///
/// `c` ve `out` geçerli işaretçiler (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_get_decode_error_data_props(
    c: *mut Dav1dContext,
    out: *mut Dav1dDataProps,
) -> c_int {
    if c.is_null() || out.is_null() {
        return dav1d_err(EINVAL);
    }
    // SAFETY: (1) ISA koşulu yoktur (`cached_error_props` taşıma).
    // (2) sınır: `c` ve `out` null kontrolünden geçmiştir.
    // (3) hizalama/takma ad: `*out` üzerindeki eski referans düşürülüp yeni değer taşınır.
    unsafe {
        let ctx = &mut (*c).inner;
        let out_ref = &mut *out;
        data_props_unref_value(out_ref);
        *out_ref = ctx.cached_error_props;
        ctx.cached_error_props = Dav1dDataProps::default();
    }
    0
}

/// `sz` baytlık yeni bir `Dav1dData` tamponu ayırır ve yazılabilir tampon adresini döndürür.
///
/// # Safety
///
/// `data` geçerli bir `Dav1dData` işaretçisi (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_data_create(data: *mut Dav1dData, sz: usize) -> *mut u8 {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() || sz > usize::MAX / 2 {
            return core::ptr::null_mut();
        }
        let Some(mut buf) = aligned64_vec(sz) else {
            return core::ptr::null_mut();
        };
        let ptr = buf.as_mut_ptr() as *mut u8;
        let r = ref_into_raw(RefPayload::DataBuffer(buf));
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dData` başlatma).
        // (2) sınır: `data` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `Dav1dData` `#[repr(C)]` düzenindedir ve `ptr` 64-bayt hizalıdır.
        unsafe {
            let d = &mut *data;
            *d = Dav1dData {
                data: ptr,
                sz,
                r#ref: r,
                m: Dav1dDataProps {
                    size: sz,
                    ..Dav1dDataProps::default()
                },
            };
        }
        ptr
    }));
    res.unwrap_or(core::ptr::null_mut())
}

/// Mevcut bir bellek bölgesini (`buf`, `sz`) özel serbest bırakma geri çağrısıyla `Dav1dData` içine sarar.
///
/// # Safety
///
/// `data`, `buf` ve `free_callback` geçerli işaretçiler (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_data_wrap(
    data: *mut Dav1dData,
    buf: *const u8,
    sz: usize,
    free_callback: Option<extern "C" fn(data: *const u8, cookie: *mut c_void)>,
    cookie: *mut c_void,
) -> c_int {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() || buf.is_null() || sz > usize::MAX / 2 {
            return dav1d_err(EINVAL);
        }
        let Some(free_cb) = free_callback else {
            return dav1d_err(EINVAL);
        };
        let r = ref_into_raw(RefPayload::WrappedData {
            ptr: buf,
            free_cb,
            cookie,
        });
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dData` sarma).
        // (2) sınır: `data` ve `buf` null değildir.
        // (3) hizalama/takma ad: `Dav1dData` `#[repr(C)]` düzenindedir.
        unsafe {
            let d = &mut *data;
            *d = Dav1dData {
                data: buf,
                sz,
                r#ref: r,
                m: Dav1dDataProps {
                    size: sz,
                    ..Dav1dDataProps::default()
                },
            };
        }
        0
    }));
    res.unwrap_or(dav1d_err(ENOMEM))
}

/// Kullanıcı tanımlı metaveri işaretçisini (`user_data`) referans sayımlı olarak `data->m.user_data` içine sarar.
///
/// # Safety
///
/// `data` ve `free_callback` geçerli işaretçiler (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_data_wrap_user_data(
    data: *mut Dav1dData,
    user_data: *const u8,
    free_callback: Option<extern "C" fn(user_data: *const u8, cookie: *mut c_void)>,
    cookie: *mut c_void,
) -> c_int {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return dav1d_err(EINVAL);
        }
        let Some(free_cb) = free_callback else {
            return dav1d_err(EINVAL);
        };
        let r = ref_into_raw(RefPayload::WrappedData {
            ptr: user_data,
            free_cb,
            cookie,
        });
        // SAFETY: (1) ISA koşulu yoktur (`user_data` sarma).
        // (2) sınır: `data` null değildir.
        // (3) hizalama/takma ad: önceki `user_data.ref` varsa düşürülüp yeni referans atanır.
        unsafe {
            let d = &mut *data;
            ref_dec(&mut d.m.user_data.r#ref);
            d.m.user_data.data = user_data;
            d.m.user_data.r#ref = r;
        }
        0
    }));
    res.unwrap_or(dav1d_err(ENOMEM))
}

/// `Dav1dData` referansını serbest bırakır ve yapıyı sıfırlar.
///
/// # Safety
///
/// `data` geçerli bir `Dav1dData` işaretçisi (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_data_unref(data: *mut Dav1dData) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() {
            return;
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dData` serbest bırakma).
        // (2) sınır: `data` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `data_unref_value` referansları sıfırlayıp düşürür.
        unsafe {
            data_unref_value(&mut *data);
        }
    }));
}

/// `Dav1dDataProps` referansını serbest bırakır ve varsayılan değerlere sıfırlar.
///
/// # Safety
///
/// `props` geçerli bir `Dav1dDataProps` işaretçisi (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_data_props_unref(props: *mut Dav1dDataProps) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if props.is_null() {
            return;
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dDataProps` serbest bırakma).
        // (2) sınır: `props` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `data_props_unref_value` `user_data.ref` referansını düşürür.
        unsafe {
            data_props_unref_value(&mut *props);
        }
    }));
}

/// `Dav1dPicture` referansını serbest bırakır ve yapıyı sıfırlar.
///
/// # Safety
///
/// `p` geçerli bir `Dav1dPicture` işaretçisi (veya `NULL`) olmalıdır.
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn dav1d_picture_unref(p: *mut Dav1dPicture) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if p.is_null() {
            return;
        }
        // SAFETY: (1) ISA koşulu yoktur (`Dav1dPicture` serbest bırakma).
        // (2) sınır: `p` null kontrolünden geçmiştir.
        // (3) hizalama/takma ad: `picture_unref_value` tüm alt `Dav1dRef` referanslarını düşürür.
        unsafe {
            picture_unref_value(&mut *p);
        }
    }));
}
