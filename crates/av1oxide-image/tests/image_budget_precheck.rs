//! `av1oxide-image` bellek bütçesi sıkı kapasite (`max_single_alloc <= limit`, `peak_live <= limit`)
//! doğrulama testi (`6.H.2`).

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Cursor;

use av1oxide::avif::AvifDecodeConfig;
use av1oxide_image::AvifDecoder;
use image::error::{ImageError, LimitErrorKind};
use image::{ImageDecoder, Limits};

struct TrackingAlloc;

thread_local! {
    static TRACK_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static LIVE_BYTES: Cell<usize> = const { Cell::new(0) };
    static PEAK_LIVE: Cell<usize> = const { Cell::new(0) };
    static MAX_SINGLE: Cell<usize> = const { Cell::new(0) };
}

fn record_alloc(size: usize) {
    let _ = TRACK_ACTIVE.try_with(|active| {
        if active.get() {
            MAX_SINGLE.with(|m| m.set(m.get().max(size)));
            LIVE_BYTES.with(|live| {
                let next = live.get().saturating_add(size);
                live.set(next);
                PEAK_LIVE.with(|p| p.set(p.get().max(next)));
            });
        }
    });
}

fn record_dealloc(size: usize) {
    let _ = TRACK_ACTIVE.try_with(|active| {
        if active.get() {
            LIVE_BYTES.with(|live| live.set(live.get().saturating_sub(size)));
        }
    });
}

// SAFETY: Tüm bellek işlemleri standart `System` ayırıcısına yönlendirilir; izleme yalnızca
// thread-local sayaçları günceller.
unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `System::alloc` geçerli layout sözleşmesiyle çağrılır.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_dealloc(layout.size());
        // SAFETY: `ptr` bu ayırıcıdan `layout` ile ayrılmıştır.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` bu ayırıcıdan `layout` ile ayrılmıştır.
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            let _ = TRACK_ACTIVE.try_with(|active| {
                if active.get() {
                    MAX_SINGLE.with(|m| m.set(m.get().max(new_size)));
                    LIVE_BYTES.with(|live| {
                        let next = live
                            .get()
                            .saturating_sub(layout.size())
                            .saturating_add(new_size);
                        live.set(next);
                        PEAK_LIVE.with(|p| p.set(p.get().max(next)));
                    });
                }
            });
        }
        out
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

fn measure_allocs<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    LIVE_BYTES.with(|v| v.set(0));
    PEAK_LIVE.with(|v| v.set(0));
    MAX_SINGLE.with(|v| v.set(0));
    TRACK_ACTIVE.with(|a| a.set(true));
    let out = f();
    TRACK_ACTIVE.with(|a| a.set(false));
    let peak = PEAK_LIVE.with(Cell::get);
    let max_single = MAX_SINGLE.with(Cell::get);
    (out, peak, max_single)
}

const AV1_1X1_8BIT: &[u8] = &[
    0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17, 0x10,
    0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16, 0x8C, 0x64,
    0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
];

fn make_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
    out
}

fn make_full_box(fourcc: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + payload.len());
    body.push(version);
    body.push(((flags >> 16) & 0xff) as u8);
    body.push(((flags >> 8) & 0xff) as u8);
    body.push((flags & 0xff) as u8);
    body.extend_from_slice(payload);
    make_box(fourcc, &body)
}

fn build_padded_avif(target_len: usize, icc: Option<&[u8]>) -> Vec<u8> {
    let mut ftyp_payload = Vec::new();
    ftyp_payload.extend_from_slice(b"avif");
    ftyp_payload.extend_from_slice(&0u32.to_be_bytes());
    ftyp_payload.extend_from_slice(b"avifmif1miaf");
    let ftyp = make_box(b"ftyp", &ftyp_payload);

    let mut hdlr_body = vec![0u8; 4];
    hdlr_body.extend_from_slice(b"pict");
    hdlr_body.extend_from_slice(&[0u8; 12]);
    hdlr_body.push(0);
    let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());

    let mut infe_b = Vec::new();
    infe_b.extend_from_slice(&1u16.to_be_bytes());
    infe_b.extend_from_slice(&0u16.to_be_bytes());
    infe_b.extend_from_slice(b"av01");
    infe_b.push(0);
    let mut iinf_b = 1u16.to_be_bytes().to_vec();
    iinf_b.extend_from_slice(&make_full_box(b"infe", 2, 0, &infe_b));
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_b);

    let mut ipco_b = Vec::new();
    let mut ispe_b = Vec::new();
    ispe_b.extend_from_slice(&1u32.to_be_bytes());
    ispe_b.extend_from_slice(&1u32.to_be_bytes());
    ipco_b.extend_from_slice(&make_full_box(b"ispe", 0, 0, &ispe_b));
    ipco_b.extend_from_slice(&make_full_box(b"pixi", 0, 0, &[3, 8, 8, 8]));
    ipco_b.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0c, 0x00]));
    if let Some(profile) = icc {
        let mut colr_b = Vec::with_capacity(4 + profile.len());
        colr_b.extend_from_slice(b"prof");
        colr_b.extend_from_slice(profile);
        ipco_b.extend_from_slice(&make_box(b"colr", &colr_b));
    }
    let ipco = make_box(b"ipco", &ipco_b);

    let mut ipma_b = Vec::new();
    ipma_b.extend_from_slice(&1u32.to_be_bytes());
    ipma_b.extend_from_slice(&1u16.to_be_bytes());
    let mut props = vec![1u8, 2u8, 0x80 | 3u8];
    if icc.is_some() {
        props.push(4u8);
    }
    ipma_b.push(props.len() as u8);
    ipma_b.extend_from_slice(&props);
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_b);
    let iprp = make_box(b"iprp", &[ipco.as_slice(), ipma.as_slice()].concat());

    let mut iloc_b = Vec::new();
    iloc_b.push(0x44);
    iloc_b.push(0x00);
    iloc_b.extend_from_slice(&1u16.to_be_bytes());
    iloc_b.extend_from_slice(&1u16.to_be_bytes());
    iloc_b.extend_from_slice(&1u16.to_be_bytes());
    iloc_b.extend_from_slice(&0u16.to_be_bytes());
    iloc_b.extend_from_slice(&1u16.to_be_bytes());
    iloc_b.extend_from_slice(&0u32.to_be_bytes());
    iloc_b.extend_from_slice(&(AV1_1X1_8BIT.len() as u32).to_be_bytes());
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_b);
    let idat = make_box(b"idat", AV1_1X1_8BIT);

    let meta = make_full_box(
        b"meta",
        0,
        0,
        &[
            hdlr.as_slice(),
            pitm.as_slice(),
            iinf.as_slice(),
            iprp.as_slice(),
            iloc.as_slice(),
            idat.as_slice(),
        ]
        .concat(),
    );

    let mut file = Vec::new();
    file.extend_from_slice(&ftyp);
    file.extend_from_slice(&meta);
    if file.len() + 8 < target_len {
        let free_payload = vec![0u8; target_len - file.len() - 8];
        file.extend_from_slice(&make_box(b"free", &free_payload));
    }
    file
}

struct ChunkedReader<'a> {
    data: &'a [u8],
    chunk_size: usize,
}

impl std::io::Read for ChunkedReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.data.len().min(buf.len()).min(self.chunk_size);
        buf[..n].copy_from_slice(&self.data[..n]);
        self.data = &self.data[n..];
        Ok(n)
    }
}

#[test]
fn with_config_input_buffer_never_exceeds_allocation_limit_and_metadata_clones_respect_budget() {
    // 1. ~1 235 baytlık geçerli AVIF, `allocation_limit = Some(1747)` ile hem tek parça hem
    //    256 B parçalı akıştan açılırken `read_to_end` ikili katlamasıyla 2 048 B tahsis
    //    etmemeli: `max_single_alloc <= 1747` ve `peak_live <= 1747`.
    let avif = build_padded_avif(1_235, None);
    assert_eq!(avif.len(), 1_235);
    let limit = 1_747usize;

    for chunk_size in [256usize, 4096] {
        let (res, peak_live, max_single) = measure_allocs(|| {
            AvifDecoder::with_config(
                ChunkedReader {
                    data: &avif,
                    chunk_size,
                },
                AvifDecodeConfig {
                    allocation_limit: Some(limit),
                    ..AvifDecodeConfig::default()
                },
            )
        });
        assert!(
            max_single <= limit,
            "chunk={chunk_size}: tekil tahsis sınırı aşıldı: max_single={max_single} > limit={limit}"
        );
        assert!(
            peak_live <= limit,
            "chunk={chunk_size}: tepe canlı bellek sınırı aşıldı: peak_live={peak_live} > limit={limit}"
        );
        drop(res);
    }

    // 1b. Yeterli bütçede (`limit = 4 096 B`) 256 B parçalı akıştan başarıyla açılır ve
    //     hem tekil tahsis hem tepe canlı bellek `<= 4 096 B` kalır.
    let (res_ok, peak_ok, max_ok) = measure_allocs(|| {
        let dec = AvifDecoder::with_config(
            ChunkedReader {
                data: &avif,
                chunk_size: 256,
            },
            AvifDecodeConfig {
                allocation_limit: Some(4_096),
                ..AvifDecodeConfig::default()
            },
        )?;
        assert_eq!(dec.dimensions(), (1, 1));
        Ok::<_, ImageError>(dec)
    });
    let dec_ok = res_ok.expect("1 235 B AVIF 4 096 B bütçeyle açılmalı");
    assert!(max_ok <= 4_096, "max_ok={max_ok} > 4096");
    assert!(peak_ok <= 4_096, "peak_ok={peak_ok} > 4096");
    drop(dec_ok);

    // 2. `allocation_limit = Some(0)` iken sıfır yığın tahsisle anında `InsufficientMemory`.
    let (res0, peak0, max0) = measure_allocs(|| {
        AvifDecoder::with_config(
            Cursor::new(&avif),
            AvifDecodeConfig {
                allocation_limit: Some(0),
                ..AvifDecodeConfig::default()
            },
        )
    });
    assert!(matches!(
        res0.expect_err("limit=0 reddedilmeli"),
        ImageError::Limits(ref e) if e.kind() == LimitErrorKind::InsufficientMemory
    ));
    assert_eq!(peak0, 0);
    assert_eq!(max0, 0);

    // 3. `icc_profile()` kopyası da etkin `Limits::max_alloc` / `allocation_limit` tavanına uymalı.
    let icc = vec![0xABu8; 4_096];
    let avif_icc = build_padded_avif(0, Some(&icc));
    let held_exact =
        av1oxide::avif::AvifPrepared::from_vec(avif_icc.clone(), &AvifDecodeConfig::default())
            .expect("prepared")
            .held_bytes();
    let mut dec_icc = AvifDecoder::with_config(
        Cursor::new(&avif_icc),
        AvifDecodeConfig {
            allocation_limit: Some(held_exact + icc.len() + 4_096),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("ICC'li AVIF açılmalı");
    // `set_limits` tutulan belleğe izin verir ancak ikinci 4 096 B ICC kopyasını 1 baytla reddeder.
    let mut tight = Limits::no_limits();
    tight.max_alloc = Some((held_exact + icc.len() - 1) as u64);
    dec_icc
        .set_limits(tight)
        .expect("tutulan bellek sınıra sığmalı");
    let err_icc = dec_icc
        .icc_profile()
        .expect_err("ikinci ICC kopyası max_alloc sınırını aşmalı");
    assert!(matches!(
        err_icc,
        ImageError::Limits(ref e) if e.kind() == LimitErrorKind::InsufficientMemory
    ));
}
