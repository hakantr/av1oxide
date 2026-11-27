//! AVIF metaveri ayrıştırıcısında `Butce` sınırının yığın tahsisinden **önce** uygulandığını
//! gerçek `#[global_allocator]` izlemesiyle doğrulayan bütünleştirme testleri (`6.E.1` / `P1-2`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use av1oxide::Error;
use av1oxide::avif::{AvifDecodeConfig, decode_with, parse_info_with};

thread_local! {
    static TRACK_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static MAX_SINGLE_ALLOC: Cell<usize> = const { Cell::new(0) };
    static CURRENT_LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK_LIVE: Cell<usize> = const { Cell::new(0) };
}

struct TrackingAllocator;

// SAFETY: `TrackingAllocator`, tüm ham bellek işlemlerini doğrudan `std::alloc::System` ayırıcısına
// devreder ve yalnızca iş parçacığı-yerel (`thread_local!`) istatistik sayaçlarını günceller.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = TRACK_ACTIVE.try_with(|active| {
            if active.get() {
                let sz = layout.size();
                let _ = MAX_SINGLE_ALLOC.try_with(|m| {
                    if sz > m.get() {
                        m.set(sz);
                    }
                });
                let _ = CURRENT_LIVE.try_with(|cur| {
                    let next = cur.get().saturating_add(sz);
                    cur.set(next);
                    let _ = PEAK_LIVE.try_with(|p| {
                        if next > p.get() {
                            p.set(next);
                        }
                    });
                });
            }
        });
        // SAFETY: `layout` Rust `GlobalAlloc` sözleşmesi gereği geçerlidir.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = TRACK_ACTIVE.try_with(|active| {
            if active.get() {
                let sz = layout.size();
                let _ = CURRENT_LIVE.try_with(|cur| {
                    cur.set(cur.get().saturating_sub(sz));
                });
            }
        });
        // SAFETY: `ptr` ve `layout` önceki `System::alloc` çağrısıyla eşleşmektedir.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = TRACK_ACTIVE.try_with(|active| {
            if active.get() {
                let old_sz = layout.size();
                let _ = MAX_SINGLE_ALLOC.try_with(|m| {
                    if new_size > m.get() {
                        m.set(new_size);
                    }
                });
                let _ = CURRENT_LIVE.try_with(|cur| {
                    let next = cur.get().saturating_sub(old_sz).saturating_add(new_size);
                    cur.set(next);
                    let _ = PEAK_LIVE.try_with(|p| {
                        if next > p.get() {
                            p.set(next);
                        }
                    });
                });
            }
        });
        // SAFETY: `ptr`, `layout` ve `new_size` Rust `GlobalAlloc` sözleşmesine uygundur.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_TRACKER: TrackingAllocator = TrackingAllocator;

fn measure_allocs<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    MAX_SINGLE_ALLOC.with(|m| m.set(0));
    CURRENT_LIVE.with(|c| c.set(0));
    PEAK_LIVE.with(|p| p.set(0));
    TRACK_ACTIVE.with(|a| a.set(true));
    let out = f();
    TRACK_ACTIVE.with(|a| a.set(false));
    let max_single = MAX_SINGLE_ALLOC.with(Cell::get);
    let peak_live = PEAK_LIVE.with(Cell::get);
    (out, max_single, peak_live)
}

const SAMPLE_1X1_AV1_OBU: &[u8] = &[
    0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17, 0x10,
    0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16, 0x8C, 0x64,
    0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
];

fn make_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let total = (8 + payload.len()) as u32;
    let mut out = Vec::with_capacity(total as usize);
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
    out
}

fn make_full_box(box_type: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + body.len());
    payload.push(version);
    payload.push(((flags >> 16) & 0xFF) as u8);
    payload.push(((flags >> 8) & 0xFF) as u8);
    payload.push((flags & 0xFF) as u8);
    payload.extend_from_slice(body);
    make_box(box_type, &payload)
}

fn make_base_parts() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut ftyp_payload = Vec::new();
    ftyp_payload.extend_from_slice(b"avif");
    ftyp_payload.extend_from_slice(&0u32.to_be_bytes());
    ftyp_payload.extend_from_slice(b"mif1");
    let ftyp = make_box(b"ftyp", &ftyp_payload);

    let mut hdlr_body = Vec::new();
    hdlr_body.extend_from_slice(&0u32.to_be_bytes());
    hdlr_body.extend_from_slice(b"pict");
    hdlr_body.extend_from_slice(&[0u8; 13]);
    let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());
    let idat = make_box(b"idat", SAMPLE_1X1_AV1_OBU);
    (ftyp, hdlr, pitm, idat)
}

fn make_simple_iloc(extent_count: u16) -> Vec<u8> {
    let mut iloc_body = vec![0x44u8, 0x00];
    iloc_body.extend_from_slice(&1u16.to_be_bytes()); // item_count = 1
    iloc_body.extend_from_slice(&1u16.to_be_bytes()); // item_id = 1
    iloc_body.extend_from_slice(&1u16.to_be_bytes()); // construction_method = 1 (idat)
    iloc_body.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index = 0
    iloc_body.extend_from_slice(&extent_count.to_be_bytes());
    for i in 0..extent_count {
        let (off, len) = if i == 0 {
            (0u32, SAMPLE_1X1_AV1_OBU.len() as u32)
        } else {
            (0u32, 0u32)
        };
        iloc_body.extend_from_slice(&off.to_be_bytes());
        iloc_body.extend_from_slice(&len.to_be_bytes());
    }
    make_full_box(b"iloc", 1, 0, &iloc_body)
}

fn make_iprp(extra_prop: Option<&[u8]>) -> Vec<u8> {
    let mut ispe_body = Vec::new();
    ispe_body.extend_from_slice(&1u32.to_be_bytes());
    ispe_body.extend_from_slice(&1u32.to_be_bytes());
    let mut ipco_body = make_full_box(b"ispe", 0, 0, &ispe_body);
    ipco_body.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]));
    if let Some(ep) = extra_prop {
        ipco_body.extend_from_slice(ep);
    }
    let ipco = make_box(b"ipco", &ipco_body);
    let mut ipma_body = Vec::new();
    ipma_body.extend_from_slice(&1u32.to_be_bytes());
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    if extra_prop.is_some() {
        ipma_body.push(3);
        ipma_body.extend_from_slice(&[0x01, 0x82, 0x03]);
    } else {
        ipma_body.push(2);
        ipma_body.extend_from_slice(&[0x01, 0x82]);
    }
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);
    let mut iprp_body = ipco;
    iprp_body.extend_from_slice(&ipma);
    make_box(b"iprp", &iprp_body)
}

#[test]
fn avif_budget_precheck_never_allocates_oversized_metadata_before_rejecting() {
    let (ftyp, hdlr, pitm, idat) = make_base_parts();
    let cfg = AvifDecodeConfig {
        allocation_limit: Some(4096),
        ..AvifDecodeConfig::default()
    };

    // 1. 65 536 baytlık `infe` item_name: `allocation_limit = 4096` altında hem `parse_info_with`
    //    hem `decode_with` BudgetExceeded döndürmeli ve tekil yığın tahsisi 4096 B'ı aşmamalı.
    let mut infe_body = Vec::new();
    infe_body.extend_from_slice(&1u16.to_be_bytes());
    infe_body.extend_from_slice(&0u16.to_be_bytes());
    infe_body.extend_from_slice(b"av01");
    infe_body.resize(infe_body.len() + 65_536, b'A');
    infe_body.push(0);
    let mut iinf_body = 1u16.to_be_bytes().to_vec();
    iinf_body.extend_from_slice(&make_full_box(b"infe", 2, 0, &infe_body));
    let iinf_big_name = make_full_box(b"iinf", 0, 0, &iinf_body);

    let mut meta_infe = Vec::new();
    for b in [
        &hdlr,
        &pitm,
        &iinf_big_name,
        &make_iprp(None),
        &make_simple_iloc(1),
        &idat,
    ] {
        meta_infe.extend_from_slice(b);
    }
    let mut avif_big_infe = ftyp.clone();
    avif_big_infe.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_infe));

    let (res_info, max_info, _peak_info) = measure_allocs(|| parse_info_with(&avif_big_infe, &cfg));
    assert!(
        matches!(res_info, Err(Error::BudgetExceeded { .. })),
        "65536 B infe adı BudgetExceeded vermeli: {res_info:?}"
    );
    assert!(
        max_info <= 4096,
        "parse_info_with 65536 B infe adını reddetmeden önce {max_info} B ayırmamalı"
    );

    let (res_dec, max_dec, _peak_dec) = measure_allocs(|| decode_with(&avif_big_infe, &cfg));
    assert!(
        matches!(res_dec, Err(Error::BudgetExceeded { .. })),
        "decode_with 65536 B infe adında BudgetExceeded vermeli: {res_dec:?}"
    );
    assert!(
        max_dec <= 4096,
        "decode_with 65536 B infe adını reddetmeden önce {max_dec} B ayırmamalı"
    );

    // 2. 65 536 baytlık `colr` (`rICC`) profili: tekil yığın tahsisi 4096 B'ı aşmamalı.
    let mut normal_infe = Vec::new();
    normal_infe.extend_from_slice(&1u16.to_be_bytes());
    normal_infe.extend_from_slice(&0u16.to_be_bytes());
    normal_infe.extend_from_slice(b"av01");
    normal_infe.push(0);
    let mut iinf_norm_body = 1u16.to_be_bytes().to_vec();
    iinf_norm_body.extend_from_slice(&make_full_box(b"infe", 2, 0, &normal_infe));
    let iinf_norm = make_full_box(b"iinf", 0, 0, &iinf_norm_body);

    let mut icc_payload = b"rICC".to_vec();
    icc_payload.resize(4 + 65_536, 0x55);
    let colr_big = make_box(b"colr", &icc_payload);

    let mut meta_icc = Vec::new();
    for b in [
        &hdlr,
        &pitm,
        &iinf_norm,
        &make_iprp(Some(&colr_big)),
        &make_simple_iloc(1),
        &idat,
    ] {
        meta_icc.extend_from_slice(b);
    }
    let mut avif_big_icc = ftyp.clone();
    avif_big_icc.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_icc));

    let (res_icc, max_icc, _) = measure_allocs(|| parse_info_with(&avif_big_icc, &cfg));
    assert!(matches!(res_icc, Err(Error::BudgetExceeded { .. })));
    assert!(
        max_icc <= 4096,
        "colr rICC reddedilmeden önce {max_icc} B ayırmamalı"
    );

    // 3. 65 536 baytlık `auxC` alt tür yükü: tekil yığın tahsisi 4096 B'ı aşmamalı.
    let mut auxc_body = b"urn:test:aux\0".to_vec();
    auxc_body.resize(auxc_body.len() + 65_536, 0x33);
    let auxc_big = make_full_box(b"auxC", 0, 0, &auxc_body);
    let mut meta_auxc = Vec::new();
    for b in [
        &hdlr,
        &pitm,
        &iinf_norm,
        &make_iprp(Some(&auxc_big)),
        &make_simple_iloc(1),
        &idat,
    ] {
        meta_auxc.extend_from_slice(b);
    }
    let mut avif_big_auxc = ftyp.clone();
    avif_big_auxc.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_auxc));

    let (res_auxc, max_auxc, _) = measure_allocs(|| parse_info_with(&avif_big_auxc, &cfg));
    assert!(matches!(res_auxc, Err(Error::BudgetExceeded { .. })));
    assert!(
        max_auxc <= 4096,
        "auxC reddedilmeden önce {max_auxc} B ayırmamalı"
    );

    // 4. Çoklu `iloc` `extent` (1000 geçerli extent ≈ 24 KiB tablo): tekil yığın tahsisi 4096 B'ı aşmamalı.
    let mut meta_iloc = Vec::new();
    for b in [
        &hdlr,
        &pitm,
        &iinf_norm,
        &make_iprp(None),
        &make_simple_iloc(1000),
        &idat,
    ] {
        meta_iloc.extend_from_slice(b);
    }
    let mut avif_big_iloc = ftyp;
    avif_big_iloc.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_iloc));

    let (res_iloc, max_iloc, _) = measure_allocs(|| parse_info_with(&avif_big_iloc, &cfg));
    assert!(matches!(res_iloc, Err(Error::BudgetExceeded { .. })));
    assert!(
        max_iloc <= 4096,
        "büyük iloc extent dizisi reddedilmeden önce {max_iloc} B ayırmamalı"
    );
}
