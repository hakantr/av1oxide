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

use crate::api::{
    Decoder, ItutT35Meta, MasteringDisplayMeta, OutputCapiMeta, PreallocCb, PreallocRequest,
};
use crate::capi::convert::{
    bpc_to_bit_depth, chroma_to_pixel_layout, decode_frame_type_from_c, error_to_dav1d,
    film_grain_from_c, frame_delay_from_settings, frame_header_to_c, has_grain_c,
    inloop_filters_from_c, parse_sequence_header_bytes, pixel_layout_to_chroma, seq_header_to_c,
    settings_to_config,
};
use crate::capi::types::{
    DAV1D_API_VERSION_MAJOR, DAV1D_API_VERSION_MINOR, DAV1D_API_VERSION_PATCH,
    DAV1D_PICTURE_ALIGNMENT, Dav1dContentLightLevel, Dav1dContext, Dav1dData, Dav1dDataProps,
    Dav1dEventFlags, Dav1dFrameHeader, Dav1dITUTT35, Dav1dLogger, Dav1dMasteringDisplay,
    Dav1dPicAllocator, Dav1dPicture, Dav1dPictureParameters, Dav1dRef, Dav1dSequenceHeader,
    Dav1dSettings, EAGAIN, EINVAL, ENOMEM, dav1d_err,
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

/// Özel `Dav1dPicAllocator` ile gönderim aşamasında ayrılan ve referans yuvası
/// boyunca (`show_existing_frame`) yeniden kullanılabilen resim girişi.
pub(crate) struct PreallocatedPicture {
    pub(crate) pic: Dav1dPicture,
    pub(crate) populated: bool,
}

/// `Dav1dContext` iç durumu (`dav1d` `src/lib.c` `c->in` / `c->out` / `c->cache` uyumlu).
pub(crate) struct ContextInner {
    pub(crate) decoder: Decoder,
    pub(crate) allocator: Dav1dPicAllocator,
    pub(crate) logger: Dav1dLogger,
    pub(crate) all_layers: bool,
    pub(crate) ready_pic: Option<(Frame, OutputCapiMeta)>,
    pub(crate) cached_pic: Option<(Frame, OutputCapiMeta)>,
    pub(crate) pending_in: Dav1dData,
    pub(crate) pending_offset: usize,
    pub(crate) pending_tag: u64,
    pub(crate) cached_error: c_int,
    pub(crate) drain: bool,
    pub(crate) event_flags: Dav1dEventFlags,
    pub(crate) cached_error_props: Dav1dDataProps,
    pub(crate) next_packet_tag: u64,
    pub(crate) packet_props: VecDeque<(u64, Dav1dDataProps)>,
    pub(crate) next_prealloc_id: u64,
    pub(crate) preallocated_pics: VecDeque<(u64, PreallocatedPicture)>,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        data_unref_value(&mut self.pending_in);
        data_props_unref_value(&mut self.cached_error_props);
        while let Some((_, mut props)) = self.packet_props.pop_front() {
            data_props_unref_value(&mut props);
        }
        while let Some((_, mut entry)) = self.preallocated_pics.pop_front() {
            picture_unref_value(&mut entry.pic);
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

/// `payload`'ı yeni bir `Dav1dRef` içine yerleştirir ve C'ye verilecek alan işaretçisini
/// (`Dav1dPicture.seq_hdr` gibi) yük son yerine konduktan **sonra** türetir.
///
/// `Box` taşındığında Rust'ın takma ad kuralları o kutudan daha önce türetilmiş işaretçileri
/// geçersiz kılar. İşaretçi burada `Arc` tek sahipken (`Arc::get_mut`) alınır; kutu bundan
/// sonra hiç taşınmaz ve son referans düşene kadar aynı yerde kalır.
#[allow(clippy::arc_with_non_send_sync)]
fn ref_into_raw_with_field<T>(
    payload: RefPayload,
    field: fn(&mut RefPayload) -> Option<&mut T>,
) -> (*mut Dav1dRef, *mut T) {
    let mut arc = Arc::new(Dav1dRef {
        inner: RefInner(payload),
    });
    let ptr = Arc::get_mut(&mut arc)
        .and_then(|r| field(&mut r.inner.0))
        .map_or(core::ptr::null_mut(), |v| v as *mut T);
    (Arc::into_raw(arc) as *mut Dav1dRef, ptr)
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

type Dav1dLoggerCallback = unsafe extern "C" fn(*mut c_void, *const c_char, core::ffi::VaList<'_>);

unsafe extern "C" fn invoke_dav1d_logger_va(
    cb: Dav1dLoggerCallback,
    cookie: *mut c_void,
    fmt: *const c_char,
    args: ...
) {
    // SAFETY: (1) ISA koşulu yoktur (C `va_list` günlük geri çağrısı).
    // (2) sınır: `cb` `log_capi_message` içinde null kontrolünden geçmiş geçerli bir `Dav1dLogger.callback` işlev işaretçisidir ve `fmt` ile `args` geçerli C dizeleridir.
    // (3) hizalama/takma ad: `args` C variadic ABI tarafından üretilen geçerli bir `va_list` değeridir ve çağrı süresince salt-okunur tüketilir.
    unsafe {
        cb(cookie, fmt, args);
    }
}

fn log_capi_message(logger: &Dav1dLogger, msg: &str) {
    if logger.callback.is_null() {
        return;
    }
    let Ok(cmsg) = std::ffi::CString::new(msg) else {
        return;
    };
    let cookie = logger.cookie;
    let cb_ptr = logger.callback;
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: (1) ISA koşulu yoktur (C günlük geri çağrısı çağrımı).
        // (2) sınır: `cb_ptr` null değildir ve `Dav1dLogger` sözleşmesi gereği `void (*callback)(void *cookie, const char *format, va_list ap)` imzasına sahip geçerli bir C fonksiyon işaretçisidir; `c"%s\n"` ve `cmsg` NUL sonludur.
        // (3) hizalama/takma ad: format dizesi ile `cmsg` salt-okunur C dizeleridir ve çağrı süresince canlıdır.
        unsafe {
            let cb: Dav1dLoggerCallback = core::mem::transmute(cb_ptr);
            invoke_dav1d_logger_va(cb, cookie, c"%s\n".as_ptr(), cmsg.as_ptr());
        }
    }));
}

/// C `dav1d` `src/obu.c` (`dav1d_parse_obus`) ile birebir uyumlu OBU/başlık ayrıştırma
/// günlük iletilerini üretir; karo/MSAC/referans/tahsis çözme hatalarında (`src/decode.c`,
/// `src/thread_task.c`, `src/lib.c`) C `dav1d` gibi günlük yazmaz.
fn log_capi_obu_error(
    logger: &Dav1dLogger,
    err: &crate::Error,
    size_limit_exceeded: Option<(i32, i32, u64)>,
) {
    use crate::error::BitstreamErrorKind;
    let crate::Error::Bitstream(b) = err else {
        return;
    };
    match b.kind {
        BitstreamErrorKind::InvalidFrameHeader(msg) if msg.contains("frame_size_limit") => {
            if let Some((w, h, lim)) = size_limit_exceeded {
                log_capi_message(logger, &format!("Frame size {w}x{h} exceeds limit {lim}"));
            } else {
                log_capi_message(logger, "Frame size exceeds limit");
            }
        }
        BitstreamErrorKind::InvalidSequenceHeader(_) => {
            log_capi_message(logger, "Error parsing sequence header");
            log_capi_message(logger, "Error parsing OBU data");
        }
        BitstreamErrorKind::InvalidFrameHeader(msg)
            if !msg.contains("sequence header yok")
                && !msg.contains("show_existing_frame taşıyamaz")
                && !msg.contains("ITU-T T.35") =>
        {
            log_capi_message(logger, "Error parsing frame header");
            log_capi_message(logger, "Error parsing OBU data");
        }
        BitstreamErrorKind::UnexpectedEnd => {
            log_capi_message(logger, "Overrun in OBU bit buffer");
        }
        _ => {
            log_capi_message(logger, "Error parsing OBU data");
        }
    }
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

fn clone_data_ref(src: &Dav1dData) -> Dav1dData {
    let mut copy = *src;
    copy.r#ref = ref_inc(src.r#ref);
    copy.m.user_data.r#ref = ref_inc(src.m.user_data.r#ref);
    copy
}

/// `dav1d_apply_grain` içinde `out == in_pic` veya paylaşımlı `Dav1dRef` örtüşmesine karşı
/// giriş resminin 7 referansını işlem boyunca canlı tutan RAII koruyucusu.
struct OwnedPictureRefs(Dav1dPicture);

impl OwnedPictureRefs {
    fn retain(src: &Dav1dPicture) -> Self {
        let mut retained = *src;
        retained.r#ref = ref_inc(src.r#ref);
        retained.seq_hdr_ref = ref_inc(src.seq_hdr_ref);
        retained.frame_hdr_ref = ref_inc(src.frame_hdr_ref);
        retained.content_light_ref = ref_inc(src.content_light_ref);
        retained.mastering_display_ref = ref_inc(src.mastering_display_ref);
        retained.itut_t35_ref = ref_inc(src.itut_t35_ref);
        retained.m.user_data.r#ref = ref_inc(src.m.user_data.r#ref);
        Self(retained)
    }

    fn into_picture(mut self) -> Dav1dPicture {
        let pic = self.0;
        self.0 = Dav1dPicture::default();
        pic
    }
}

impl Drop for OwnedPictureRefs {
    fn drop(&mut self) {
        picture_unref_value(&mut self.0);
    }
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

fn attach_picture_metadata_parts(
    pic: &mut Dav1dPicture,
    seq_hdr: Option<&crate::bitstream::SequenceHeader>,
    frame_hdr: Option<&crate::bitstream::frame_header::FrameHeader>,
    content_light: Option<(u16, u16)>,
    mastering_display: Option<MasteringDisplayMeta>,
    itut_t35: &[ItutT35Meta],
) -> Result<(), c_int> {
    if let Some(seq) = seq_hdr {
        let payload = RefPayload::SeqHdr(Box::new(seq_header_to_c(seq)));
        (pic.seq_hdr_ref, pic.seq_hdr) = ref_into_raw_with_field(payload, |p| match p {
            RefPayload::SeqHdr(b) => Some(&mut **b),
            _ => None,
        });
    }
    if let Some(fh) = frame_hdr {
        let payload = RefPayload::FrameHdr(Box::new(frame_header_to_c(fh)));
        (pic.frame_hdr_ref, pic.frame_hdr) = ref_into_raw_with_field(payload, |p| match p {
            RefPayload::FrameHdr(b) => Some(&mut **b),
            _ => None,
        });
    }
    if let Some((max_cll, max_fall)) = content_light {
        let payload = RefPayload::ContentLight(Box::new(Dav1dContentLightLevel {
            max_content_light_level: max_cll,
            max_frame_average_light_level: max_fall,
        }));
        (pic.content_light_ref, pic.content_light) =
            ref_into_raw_with_field(payload, |p| match p {
                RefPayload::ContentLight(b) => Some(&mut **b),
                _ => None,
            });
    }
    if let Some(md) = mastering_display {
        let payload = RefPayload::MasteringDisplay(Box::new(Dav1dMasteringDisplay {
            primaries: md.primaries,
            white_point: md.white_point,
            max_luminance: md.max_luminance,
            min_luminance: md.min_luminance,
        }));
        (pic.mastering_display_ref, pic.mastering_display) =
            ref_into_raw_with_field(payload, |p| match p {
                RefPayload::MasteringDisplay(b) => Some(&mut **b),
                _ => None,
            });
    }
    if !itut_t35.is_empty() {
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        let mut entries: Vec<Dav1dITUTT35> = Vec::new();
        if payloads.try_reserve_exact(itut_t35.len()).is_err()
            || entries.try_reserve_exact(itut_t35.len()).is_err()
        {
            clear_picture_metadata_refs(pic);
            return Err(dav1d_err(ENOMEM));
        }
        for item in itut_t35 {
            let mut p: Vec<u8> = Vec::new();
            if p.try_reserve_exact(item.payload.len()).is_err() {
                clear_picture_metadata_refs(pic);
                return Err(dav1d_err(ENOMEM));
            }
            p.extend_from_slice(&item.payload);
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
    Ok(())
}

fn attach_picture_metadata(pic: &mut Dav1dPicture, meta: &OutputCapiMeta) -> Result<(), c_int> {
    attach_picture_metadata_parts(
        pic,
        meta.seq_hdr.as_deref(),
        meta.frame_hdr.as_deref(),
        meta.content_light,
        meta.mastering_display,
        &meta.itut_t35,
    )
}

fn clear_picture_metadata_refs(pic: &mut Dav1dPicture) {
    let mut seq_ref = pic.seq_hdr_ref;
    let mut frame_ref = pic.frame_hdr_ref;
    let mut cll_ref = pic.content_light_ref;
    let mut mdcv_ref = pic.mastering_display_ref;
    let mut itut_ref = pic.itut_t35_ref;
    pic.seq_hdr = core::ptr::null_mut();
    pic.seq_hdr_ref = core::ptr::null_mut();
    pic.frame_hdr = core::ptr::null_mut();
    pic.frame_hdr_ref = core::ptr::null_mut();
    pic.content_light = core::ptr::null_mut();
    pic.content_light_ref = core::ptr::null_mut();
    pic.mastering_display = core::ptr::null_mut();
    pic.mastering_display_ref = core::ptr::null_mut();
    pic.itut_t35 = core::ptr::null_mut();
    pic.n_itut_t35 = 0;
    pic.itut_t35_ref = core::ptr::null_mut();
    data_props_unref_value(&mut pic.m);
    ref_dec(&mut seq_ref);
    ref_dec(&mut frame_ref);
    ref_dec(&mut cll_ref);
    ref_dec(&mut mdcv_ref);
    ref_dec(&mut itut_ref);
}

fn custom_picture_release_snapshot(pic: &Dav1dPicture) -> Dav1dPicture {
    Dav1dPicture {
        p: pic.p,
        data: pic.data,
        stride: pic.stride,
        allocator_data: pic.allocator_data,
        ..Dav1dPicture::default()
    }
}

fn preallocate_custom_picture(
    allocator: &Dav1dPicAllocator,
    props: Dav1dDataProps,
    req: &PreallocRequest<'_>,
) -> Result<Dav1dPicture, c_int> {
    let (Some(alloc_cb), Some(release_cb)) = (
        allocator.alloc_picture_callback,
        allocator.release_picture_callback,
    ) else {
        return Err(dav1d_err(EINVAL));
    };
    let mut pic = Dav1dPicture {
        p: Dav1dPictureParameters {
            w: req.width as c_int,
            h: req.height as c_int,
            layout: chroma_to_pixel_layout(req.chroma),
            bpc: req.bit_depth.bits() as c_int,
        },
        m: props,
        ..Dav1dPicture::default()
    };
    if let Err(e) = attach_picture_metadata_parts(
        &mut pic,
        req.seq_hdr,
        Some(req.frame_hdr),
        req.content_light,
        req.mastering_display,
        req.itut_t35,
    ) {
        picture_unref_value(&mut pic);
        return Err(e);
    }
    let rc = alloc_cb(&mut pic as *mut Dav1dPicture, allocator.cookie);
    if rc < 0 {
        picture_unref_value(&mut pic);
        return Err(rc);
    }
    let snapshot = custom_picture_release_snapshot(&pic);
    pic.r#ref = ref_into_raw(RefPayload::CustomPicture {
        pic_snapshot: snapshot,
        release_cb,
        cookie: allocator.cookie,
    });
    Ok(pic)
}

enum PictureMetaSource<'a> {
    Owned(&'a OutputCapiMeta),
    RetainFrom(&'a Dav1dPicture),
}

fn retain_picture_metadata(dst: &mut Dav1dPicture, src: &Dav1dPicture) {
    dst.seq_hdr = src.seq_hdr;
    dst.seq_hdr_ref = ref_inc(src.seq_hdr_ref);
    dst.frame_hdr = src.frame_hdr;
    dst.frame_hdr_ref = ref_inc(src.frame_hdr_ref);
    dst.content_light = src.content_light;
    dst.content_light_ref = ref_inc(src.content_light_ref);
    dst.mastering_display = src.mastering_display;
    dst.mastering_display_ref = ref_inc(src.mastering_display_ref);
    dst.itut_t35 = src.itut_t35;
    dst.n_itut_t35 = src.n_itut_t35;
    dst.itut_t35_ref = ref_inc(src.itut_t35_ref);
}

fn fill_output_picture(
    ctx: &ContextInner,
    out: &mut Dav1dPicture,
    frame: &Frame,
    meta_src: PictureMetaSource<'_>,
    props: Dav1dDataProps,
) -> Result<(), c_int> {
    picture_unref_value(out);
    out.p = Dav1dPictureParameters {
        w: frame.width() as c_int,
        h: frame.height() as c_int,
        layout: chroma_to_pixel_layout(frame.chroma_format()),
        bpc: frame.bit_depth().bits() as c_int,
    };
    // `dav1d` (`src/picture.c`) sözleşmesi gereği özel `alloc_picture_callback` çağrılmadan
    // önce `p`, `m`, `seq_hdr`, `frame_hdr` ve HDR/T.35 başlıkları eksiksiz doldurulmalıdır
    // (ör. FFmpeg `libdav1d_picture_allocator` `p->seq_hdr->hbd` okur).
    out.m = props;
    match meta_src {
        PictureMetaSource::Owned(meta) => {
            if let Err(e) = attach_picture_metadata(out, meta) {
                picture_unref_value(out);
                return Err(e);
            }
        }
        PictureMetaSource::RetainFrom(src) => retain_picture_metadata(out, src),
    }

    if let (Some(alloc_cb), Some(release_cb)) = (
        ctx.allocator.alloc_picture_callback,
        ctx.allocator.release_picture_callback,
    ) {
        let rc = alloc_cb(out as *mut Dav1dPicture, ctx.allocator.cookie);
        if rc < 0 {
            picture_unref_value(out);
            return Err(rc);
        }
        let snapshot = custom_picture_release_snapshot(out);
        let pic_ref = ref_into_raw(RefPayload::CustomPicture {
            pic_snapshot: snapshot,
            release_cb,
            cookie: ctx.allocator.cookie,
        });
        out.r#ref = pic_ref;
        if let Err(e) = copy_frame_to_custom_pic(frame, out) {
            picture_unref_value(out);
            return Err(e);
        }
    } else {
        match allocate_default_picture_planes(frame, out) {
            Ok(planes) => {
                out.r#ref = ref_into_raw(RefPayload::DefaultPicturePlanes(planes));
            }
            Err(e) => {
                picture_unref_value(out);
                return Err(e);
            }
        }
    }

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
            Err(code) => {
                log_capi_message(&settings.logger, "dav1d_open: invalid settings");
                return code;
            }
        };
        let mut decoder = match Decoder::new(config) {
            Ok(dec) => dec,
            Err(err) => {
                log_capi_message(&settings.logger, "dav1d_open: invalid settings");
                return error_to_dav1d(&err);
            }
        };
        decoder.set_inloop_filters(inloop_filters_from_c(settings.inloop_filters));
        decoder.set_capi_strict_frame_delay(true);
        if let Some(ft) = decode_frame_type_from_c(settings.decode_frame_type) {
            decoder.set_decode_frame_type(ft);
        }
        let ctx = Box::new(Dav1dContext {
            inner: ContextInner {
                decoder,
                allocator: settings.allocator,
                logger: settings.logger,
                all_layers: settings.all_layers != 0,
                ready_pic: None,
                cached_pic: None,
                pending_in: Dav1dData::default(),
                pending_offset: 0,
                pending_tag: 0,
                cached_error: 0,
                drain: false,
                event_flags: 0,
                cached_error_props: Dav1dDataProps::default(),
                next_packet_tag: 1,
                packet_props: VecDeque::new(),
                next_prealloc_id: 1,
                preallocated_pics: VecDeque::new(),
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

fn is_packet_tag_referenced(ctx: &ContextInner, tag: u64) -> bool {
    if tag == 0 {
        return false;
    }
    ctx.pending_tag == tag
        || ctx.decoder.has_pending_packet_tag(tag)
        || ctx
            .ready_pic
            .as_ref()
            .is_some_and(|(_, m)| m.packet_tag == tag)
        || ctx
            .cached_pic
            .as_ref()
            .is_some_and(|(_, m)| m.packet_tag == tag)
}

fn cleanup_all_unused_packet_props(ctx: &mut ContextInner) {
    let mut i = 0;
    while i < ctx.packet_props.len() {
        let tag = ctx.packet_props[i].0;
        if is_packet_tag_referenced(ctx, tag) {
            i += 1;
        } else if let Some((_, mut unused_props)) = ctx.packet_props.remove(i) {
            data_props_unref_value(&mut unused_props);
        }
    }
}

fn record_error_props_for_tag(ctx: &mut ContextInner, tag: u64, fallback: Option<&Dav1dDataProps>) {
    if tag != 0
        && let Some(pos) = ctx.packet_props.iter().position(|(t, _)| *t == tag)
    {
        let props = if is_packet_tag_referenced(ctx, tag) {
            clone_data_props(&ctx.packet_props[pos].1)
        } else {
            ctx.packet_props.remove(pos).unwrap().1
        };
        data_props_unref_value(&mut ctx.cached_error_props);
        ctx.cached_error_props = props;
    } else if let Some(fb) = fallback {
        data_props_unref_value(&mut ctx.cached_error_props);
        ctx.cached_error_props = clone_data_props(fb);
    }
}

fn is_prealloc_id_referenced(ctx: &ContextInner, id: u64) -> bool {
    if id == 0 {
        return false;
    }
    ctx.decoder.has_pending_prealloc_id(id)
        || ctx.decoder.has_ref_slot_prealloc_id(id)
        || ctx
            .ready_pic
            .as_ref()
            .is_some_and(|(_, m)| m.prealloc_id == id)
        || ctx
            .cached_pic
            .as_ref()
            .is_some_and(|(_, m)| m.prealloc_id == id)
}

fn cleanup_orphaned_preallocs(ctx: &mut ContextInner) {
    let mut i = 0;
    while i < ctx.preallocated_pics.len() {
        let id = ctx.preallocated_pics[i].0;
        if is_prealloc_id_referenced(ctx, id) {
            i += 1;
        } else if let Some((_, mut entry)) = ctx.preallocated_pics.remove(i) {
            picture_unref_value(&mut entry.pic);
        }
    }
}

fn pull_ready_nonblocking(ctx: &mut ContextInner) {
    if ctx.ready_pic.is_some() || ctx.cached_error != 0 {
        return;
    }
    match ctx.decoder.receive_frame_capi(false) {
        Ok(Some(mut item)) => {
            ctx.event_flags |= item.1.event_flags;
            item.1.event_flags = 0;
            ctx.ready_pic = Some(item);
        }
        Ok(None) => {}
        Err(err) => {
            if let Some(tag) = ctx.decoder.last_error_packet_tag() {
                record_error_props_for_tag(ctx, tag, None);
            }
            cleanup_orphaned_preallocs(ctx);
            cleanup_all_unused_packet_props(ctx);
            ctx.cached_error = error_to_dav1d(&err);
        }
    }
}

/// `dav1d` `output_picture_ready(c, drain)` (`src/lib.c:332-351`) karşılığı.
fn output_picture_ready(ctx: &mut ContextInner, drain: bool) -> bool {
    loop {
        if ctx.cached_error != 0 {
            return true;
        }
        pull_ready_nonblocking(ctx);
        if ctx.cached_error != 0 {
            return true;
        }
        let max_spatial_id = ctx.decoder.max_spatial_id();
        if !ctx.all_layers && max_spatial_id > 0 {
            if let (Some((_, ready_meta)), Some((_, cached_meta))) =
                (ctx.ready_pic.as_ref(), ctx.cached_pic.as_ref())
            {
                let cached_sid = cached_meta.frame_hdr.as_ref().map_or(0, |h| h.spatial_id);
                let ready_sid = ready_meta.frame_hdr.as_ref().map_or(0, |h| h.spatial_id);
                // `dav1d` SVC `all_layers == 0`: zamansal birim sınırı paket sınırıyla
                // değil `OBU_TEMPORAL_DELIMITER` (`new_temporal_unit`) veya azalan/eşit
                // `spatial_id` ile belirlenir.
                let is_new_tu = ready_sid <= cached_sid || ready_meta.new_temporal_unit;
                if cached_sid >= max_spatial_id || is_new_tu {
                    return true;
                }
                // Aynı zamansal birimde daha yüksek uzamsal katman: eski katmanı at.
                let next = ctx.ready_pic.take();
                if let Some((_, dropped_meta)) = ctx.cached_pic.take() {
                    ctx.event_flags |= dropped_meta.event_flags;
                    ctx.cached_pic = next;
                    cleanup_orphaned_preallocs(ctx);
                    cleanup_all_unused_packet_props(ctx);
                } else {
                    ctx.cached_pic = next;
                }
                continue;
            }
            if ctx.cached_pic.is_some() && drain {
                return true;
            }
            if ctx.ready_pic.is_some() {
                ctx.cached_pic = ctx.ready_pic.take();
                continue;
            }
            return false;
        }
        return ctx.cached_pic.is_some() || ctx.ready_pic.is_some();
    }
}

/// `dav1d` `gen_picture(c)` (`src/lib.c:412-436`) karşılığı: bekleyen girdi (`pending_in`)
/// üzerinden çıktı hazır olana kadar OBU bazında ilerler.
fn gen_picture(ctx: &mut ContextInner) -> c_int {
    if output_picture_ready(ctx, false) {
        return 0;
    }

    while !ctx.pending_in.data.is_null() && ctx.pending_offset < ctx.pending_in.sz {
        let data_ptr = ctx.pending_in.data;
        let data_sz = ctx.pending_in.sz;
        let ts = ctx.pending_in.m.timestamp;
        let tag = ctx.pending_tag;
        let mut offset = ctx.pending_offset;

        // SAFETY: (1) ISA koşulu yoktur (bekleyen girdi paketi bayt dilimi).
        // (2) sınır: `data_ptr` null değildir ve `dav1d_send_data` girişinde `0 < data_sz <= usize::MAX / 2` doğrulanmıştır; `ctx.pending_in` tamponu bu çağrı boyunca canlı tutar.
        // (3) hizalama/takma ad: `u8` dilimi hizalama gerektirmez ve `step_one_obu_capi` süresince salt-okunurdur.
        let payload = unsafe { core::slice::from_raw_parts(data_ptr, data_sz) };
        let has_custom_alloc = ctx.allocator.alloc_picture_callback.is_some()
            && ctx.allocator.release_picture_callback.is_some();
        let mut prealloc_err: Option<c_int> = None;
        let step_res = {
            let ContextInner {
                decoder,
                allocator,
                pending_in,
                packet_props,
                preallocated_pics,
                next_prealloc_id,
                ..
            } = ctx;
            let mut prealloc_cb = |req: &PreallocRequest<'_>| -> crate::Result<u64> {
                let props = if req.packet_tag != 0
                    && let Some((_, p)) = packet_props.iter().find(|(t, _)| *t == req.packet_tag)
                {
                    clone_data_props(p)
                } else {
                    clone_data_props(&pending_in.m)
                };
                match preallocate_custom_picture(allocator, props, req) {
                    Ok(pic) => {
                        let id = *next_prealloc_id;
                        *next_prealloc_id = next_prealloc_id.wrapping_add(1).max(1);
                        preallocated_pics.push_back((
                            id,
                            PreallocatedPicture {
                                pic,
                                populated: false,
                            },
                        ));
                        Ok(id)
                    }
                    Err(code) => {
                        prealloc_err = Some(code);
                        Err(crate::Error::InvalidState(
                            "özel resim ayırıcısı hata döndürdü",
                        ))
                    }
                }
            };
            let prealloc_opt: Option<&mut PreallocCb<'_>> = if has_custom_alloc {
                Some(&mut prealloc_cb)
            } else {
                None
            };
            decoder.step_one_obu_capi(payload, &mut offset, Some(ts), tag, prealloc_opt)
        };

        for warning in ctx.decoder.take_capi_warnings() {
            log_capi_message(&ctx.logger, &warning);
        }

        match step_res {
            Ok(true) => {
                ctx.pending_offset = offset;
                if ctx.pending_offset >= ctx.pending_in.sz {
                    data_unref_value(&mut ctx.pending_in);
                    ctx.pending_offset = 0;
                    ctx.pending_tag = 0;
                }
                cleanup_orphaned_preallocs(ctx);
                cleanup_all_unused_packet_props(ctx);
                if output_picture_ready(ctx, false) {
                    break;
                }
            }
            Ok(false) => {
                data_unref_value(&mut ctx.pending_in);
                ctx.pending_offset = 0;
                ctx.pending_tag = 0;
                cleanup_orphaned_preallocs(ctx);
                cleanup_all_unused_packet_props(ctx);
                if output_picture_ready(ctx, false) {
                    break;
                }
            }
            Err(err) => {
                let size_limit_exceeded = ctx.decoder.take_last_frame_size_limit_exceeded();
                log_capi_obu_error(&ctx.logger, &err, size_limit_exceeded);
                let code = prealloc_err.unwrap_or_else(|| error_to_dav1d(&err));
                let err_tag = ctx.decoder.last_error_packet_tag().unwrap_or(tag);
                let fallback_props = clone_data_props(&ctx.pending_in.m);
                data_unref_value(&mut ctx.pending_in);
                ctx.pending_offset = 0;
                ctx.pending_tag = 0;
                cleanup_orphaned_preallocs(ctx);
                // C `dav1d` `src/obu.c:1310-1316`: `frame_size_limit` (`ERANGE`) hatasında
                // `c->cached_error_props` güncellenmeden doğrudan `DAV1D_ERR(ERANGE)` dönülür.
                if !(prealloc_err.is_none() && code == dav1d_err(crate::capi::ERANGE)) {
                    record_error_props_for_tag(ctx, err_tag, Some(&fallback_props));
                }
                let mut fb_drop = fallback_props;
                data_props_unref_value(&mut fb_drop);
                cleanup_all_unused_packet_props(ctx);
                if output_picture_ready(ctx, false) {
                    if ctx.cached_error == 0 {
                        ctx.cached_error = code;
                    }
                    break;
                }
                return code;
            }
        }
    }

    0
}

fn take_or_clone_packet_props(
    ctx: &mut ContextInner,
    packet_tag: u64,
    fallback_ts: Option<i64>,
) -> Dav1dDataProps {
    let still_pending = is_packet_tag_referenced(ctx, packet_tag);
    if let Some(pos) = ctx
        .packet_props
        .iter()
        .position(|(tag, _)| *tag == packet_tag)
    {
        if still_pending {
            clone_data_props(&ctx.packet_props[pos].1)
        } else {
            ctx.packet_props.remove(pos).unwrap().1
        }
    } else {
        Dav1dDataProps {
            timestamp: fallback_ts.unwrap_or(i64::MIN),
            ..Dav1dDataProps::default()
        }
    }
}

fn deliver_ready_picture(ctx: &mut ContextInner, out_ref: &mut Dav1dPicture) -> c_int {
    let max_spatial_id = ctx.decoder.max_spatial_id();
    let item = ctx.cached_pic.take().or_else(|| ctx.ready_pic.take());
    if !ctx.all_layers && max_spatial_id > 0 && ctx.cached_pic.is_none() {
        ctx.cached_pic = ctx.ready_pic.take();
    }
    let Some((frame, meta)) = item else {
        return dav1d_err(EAGAIN);
    };
    ctx.event_flags |= meta.event_flags;

    let has_applied_grain = ctx.decoder.config().apply_grain
        && meta
            .frame_hdr
            .as_ref()
            .is_some_and(|fh| crate::decode::filmgrain::has_grain(&fh.film_grain.data));

    if meta.prealloc_id != 0
        && !has_applied_grain
        && let Some(pos) = ctx
            .preallocated_pics
            .iter()
            .position(|(pid, _)| *pid == meta.prealloc_id)
    {
        if !ctx.preallocated_pics[pos].1.populated {
            if let Err(code) =
                copy_frame_to_custom_pic(&frame, &mut ctx.preallocated_pics[pos].1.pic)
            {
                let err_props = if meta.is_show_existing {
                    take_or_clone_packet_props(ctx, meta.packet_tag, frame.metadata().timestamp)
                } else {
                    clone_data_props(&ctx.preallocated_pics[pos].1.pic.m)
                };
                data_props_unref_value(&mut ctx.cached_error_props);
                ctx.cached_error_props = err_props;
                cleanup_orphaned_preallocs(ctx);
                cleanup_all_unused_packet_props(ctx);
                return code;
            }
            ctx.preallocated_pics[pos].1.populated = true;
        }
        let still_referenced = is_prealloc_id_referenced(ctx, meta.prealloc_id);
        let mut delivered = if still_referenced {
            OwnedPictureRefs::retain(&ctx.preallocated_pics[pos].1.pic).into_picture()
        } else {
            ctx.preallocated_pics.remove(pos).unwrap().1.pic
        };
        if meta.is_show_existing {
            let props =
                take_or_clone_packet_props(ctx, meta.packet_tag, frame.metadata().timestamp);
            let err_props_backup = clone_data_props(&props);
            clear_picture_metadata_refs(&mut delivered);
            delivered.m = props;
            if let Err(code) = attach_picture_metadata(&mut delivered, &meta) {
                picture_unref_value(&mut delivered);
                data_props_unref_value(&mut ctx.cached_error_props);
                ctx.cached_error_props = err_props_backup;
                cleanup_orphaned_preallocs(ctx);
                cleanup_all_unused_packet_props(ctx);
                return code;
            }
            let mut drop_backup = err_props_backup;
            data_props_unref_value(&mut drop_backup);
        }
        picture_unref_value(out_ref);
        *out_ref = delivered;
        cleanup_orphaned_preallocs(ctx);
        cleanup_all_unused_packet_props(ctx);
        return 0;
    }

    if meta.prealloc_id != 0 {
        cleanup_orphaned_preallocs(ctx);
    }
    let props = take_or_clone_packet_props(ctx, meta.packet_tag, frame.metadata().timestamp);
    let err_props_backup = clone_data_props(&props);
    match fill_output_picture(ctx, out_ref, &frame, PictureMetaSource::Owned(&meta), props) {
        Ok(()) => {
            let mut drop_backup = err_props_backup;
            data_props_unref_value(&mut drop_backup);
            cleanup_orphaned_preallocs(ctx);
            cleanup_all_unused_packet_props(ctx);
            0
        }
        Err(code) => {
            data_props_unref_value(&mut ctx.cached_error_props);
            ctx.cached_error_props = err_props_backup;
            cleanup_orphaned_preallocs(ctx);
            cleanup_all_unused_packet_props(ctx);
            code
        }
    }
}

/// Çözücüye bir `Dav1dData` OBU paketi gönderir (`0` başarıda `in_data` referansını tüketir;
/// bekleyen giriş yuvası doluysa `in_data`ya dokunmadan `-EAGAIN` döndürür).
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
        }

        if !ctx.pending_in.data.is_null() {
            return dav1d_err(EAGAIN);
        }
        if data_ref.data.is_null() {
            data_unref_value(data_ref);
            return 0;
        }

        let packet_tag = ctx.next_packet_tag;
        ctx.next_packet_tag = ctx.next_packet_tag.wrapping_add(1).max(1);

        let retained_props = clone_data_props(&data_ref.m);
        if ctx.packet_props.len() >= 1024
            && let Some((_, mut old)) = ctx.packet_props.pop_front()
        {
            data_props_unref_value(&mut old);
        }
        ctx.packet_props.push_back((packet_tag, retained_props));

        ctx.pending_in = clone_data_ref(data_ref);
        ctx.pending_offset = 0;
        ctx.pending_tag = packet_tag;

        let rc = gen_picture(ctx);
        if rc == 0 {
            data_unref_value(data_ref);
        }
        rc
    }));
    res.unwrap_or(dav1d_err(EINVAL))
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

        let res = gen_picture(ctx);
        if res < 0 {
            return res;
        }

        if ctx.cached_error != 0 {
            let err_code = ctx.cached_error;
            ctx.cached_error = 0;
            return err_code;
        }

        let single_fc = ctx.decoder.config().effective_frame_delay() == 1;
        if output_picture_ready(ctx, single_fc) {
            if ctx.cached_error != 0 {
                let err_code = ctx.cached_error;
                ctx.cached_error = 0;
                return err_code;
            }
            return deliver_ready_picture(ctx, out_ref);
        }

        if !single_fc && drain {
            loop {
                match ctx.decoder.receive_frame_capi(true) {
                    Ok(Some(mut item)) => {
                        ctx.event_flags |= item.1.event_flags;
                        item.1.event_flags = 0;
                        ctx.ready_pic = Some(item);
                        if output_picture_ready(ctx, false) {
                            if ctx.cached_error != 0 {
                                let err_code = ctx.cached_error;
                                ctx.cached_error = 0;
                                return err_code;
                            }
                            return deliver_ready_picture(ctx, out_ref);
                        }
                    }
                    Ok(None) => {
                        if output_picture_ready(ctx, true) {
                            if ctx.cached_error != 0 {
                                let err_code = ctx.cached_error;
                                ctx.cached_error = 0;
                                return err_code;
                            }
                            return deliver_ready_picture(ctx, out_ref);
                        }
                        return dav1d_err(EAGAIN);
                    }
                    Err(err) => {
                        if let Some(tag) = ctx.decoder.last_error_packet_tag() {
                            record_error_props_for_tag(ctx, tag, None);
                        }
                        cleanup_orphaned_preallocs(ctx);
                        return error_to_dav1d(&err);
                    }
                }
            }
        }

        dav1d_err(EAGAIN)
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
        // `out == in_pic` veya paylaşımlı `Dav1dRef` durumunda `picture_unref_value(out_ref)`
        // çağrısının son referansı düşürüp UAF üretmesini önlemek için tüm 7 referansı önceden tut.
        let in_guard = OwnedPictureRefs::retain(&in_snapshot);
        // SAFETY: (1) ISA koşulu yoktur (`frame_hdr` ve `seq_hdr` okuma).
        // (2) sınır: `in_snapshot.frame_hdr` null değildir ve `in_guard` tarafından canlı tutulur.
        // (3) hizalama/takma ad: başlık yapıları `#[repr(C)]` düzenindedir ve salt-okunur erişilir.
        let (fh_c, mc_identity) = unsafe {
            let fh = &*in_guard.0.frame_hdr;
            let mc_id = !in_guard.0.seq_hdr.is_null() && (*in_guard.0.seq_hdr).mtrx == 0;
            (*fh, mc_id)
        };

        if !has_grain_c(&fh_c.film_grain.data) {
            picture_unref_value(out_ref);
            *out_ref = in_guard.into_picture();
            return 0;
        }

        let src_frame = match frame_from_c_picture(&in_guard.0) {
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

        let props = clone_data_props(&in_guard.0.m);
        if let Err(code) = fill_output_picture(
            ctx,
            out_ref,
            &grained_frame,
            PictureMetaSource::RetainFrom(&in_guard.0),
            props,
        ) {
            return code;
        }
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
        ctx.ready_pic = None;
        ctx.cached_pic = None;
        data_unref_value(&mut ctx.pending_in);
        ctx.pending_offset = 0;
        ctx.pending_tag = 0;
        ctx.cached_error = 0;
        ctx.drain = false;
        data_props_unref_value(&mut ctx.cached_error_props);
        while let Some((_, mut props)) = ctx.packet_props.pop_front() {
            data_props_unref_value(&mut props);
        }
        while let Some((_, mut entry)) = ctx.preallocated_pics.pop_front() {
            picture_unref_value(&mut entry.pic);
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
