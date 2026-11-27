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

/// `6.F.4`: GainMap ikinci `alt_icc_profile` kopyası + `Box<AvifImage>` ve Grid planlama
/// vektörlerinin (`!has_dups` sıfır-tahsis ve `has_dups` düz dizi) `Butce` tarafından
/// tahsis öncesinde eksiksiz sayıldığını doğrular.
#[test]
fn avif_budget_accounts_gainmap_second_icc_and_grid_planning() {
    let (ftyp, hdlr, pitm, _) = make_base_parts();

    // 1. Ana görüntüde 3 144 B ICC + `tmap` öğesinde 3 144 B `alt_icc_profile` içeren GainMap AVIF.
    let icc_size = 3144usize;
    let mut primary_icc_payload = b"rICC".to_vec();
    primary_icc_payload.resize(4 + icc_size, 0x11);
    let colr_primary = make_box(b"colr", &primary_icc_payload);

    let mut alt_icc_payload = b"rICC".to_vec();
    alt_icc_payload.resize(4 + icc_size, 0x22);
    let colr_tmap = make_box(b"colr", &alt_icc_payload);

    // ISO 21496-1 `tmap` yükü (version=0, minimum_version=0, writer_version=0, single-channel,
    // use_base_colour_space=0).
    let mut tmap_payload = Vec::new();
    tmap_payload.push(0u8); // version
    tmap_payload.extend_from_slice(&0u16.to_be_bytes()); // minimum_version
    tmap_payload.extend_from_slice(&0u16.to_be_bytes()); // writer_version
    tmap_payload.push(0x00u8); // is_multichannel = 0, use_base_colour_space = 0
    tmap_payload.extend_from_slice(&0u32.to_be_bytes()); // base_hdr_headroom_n
    tmap_payload.extend_from_slice(&1u32.to_be_bytes()); // base_hdr_headroom_d
    tmap_payload.extend_from_slice(&1u32.to_be_bytes()); // alternate_hdr_headroom_n
    tmap_payload.extend_from_slice(&1u32.to_be_bytes()); // alternate_hdr_headroom_d
    // 1 kanal: gain_map_min, gain_map_max, gamma, base_offset, alternate_offset
    tmap_payload.extend_from_slice(&0i32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1i32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&0i32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&0i32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());

    let make_infe = |id: u16, ty: &[u8; 4]| -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&id.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(ty);
        b.push(0);
        make_full_box(b"infe", 2, 0, &b)
    };
    let mut iinf_body = 3u16.to_be_bytes().to_vec();
    iinf_body.extend_from_slice(&make_infe(1, b"av01")); // primary
    iinf_body.extend_from_slice(&make_infe(2, b"av01")); // gain map image
    iinf_body.extend_from_slice(&make_infe(3, b"tmap")); // tmap derived item
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    let mut dimg_body = Vec::new();
    dimg_body.extend_from_slice(&3u16.to_be_bytes()); // from tmap (3)
    dimg_body.extend_from_slice(&2u16.to_be_bytes()); // count = 2
    dimg_body.extend_from_slice(&1u16.to_be_bytes()); // base = 1
    dimg_body.extend_from_slice(&2u16.to_be_bytes()); // gainmap = 2
    let iref = make_full_box(b"iref", 0, 0, &make_box(b"dimg", &dimg_body));

    let mut ispe_body = Vec::new();
    ispe_body.extend_from_slice(&1u32.to_be_bytes());
    ispe_body.extend_from_slice(&1u32.to_be_bytes());
    let mut ipco_body = make_full_box(b"ispe", 0, 0, &ispe_body); // prop 1
    ipco_body.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00])); // prop 2
    ipco_body.extend_from_slice(&colr_primary); // prop 3
    ipco_body.extend_from_slice(&colr_tmap); // prop 4
    let ipco = make_box(b"ipco", &ipco_body);

    let mut ipma_body = 3u32.to_be_bytes().to_vec();
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_body.push(3);
    ipma_body.extend_from_slice(&[0x01, 0x82, 0x03]);
    ipma_body.extend_from_slice(&2u16.to_be_bytes());
    ipma_body.push(2);
    ipma_body.extend_from_slice(&[0x01, 0x82]);
    ipma_body.extend_from_slice(&3u16.to_be_bytes());
    ipma_body.push(2);
    ipma_body.extend_from_slice(&[0x01, 0x04]);
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);
    let iprp = make_box(b"iprp", &[ipco.as_slice(), ipma.as_slice()].concat());

    let mut idat_payload = Vec::new();
    let off1 = 0u32;
    idat_payload.extend_from_slice(SAMPLE_1X1_AV1_OBU);
    let off2 = idat_payload.len() as u32;
    idat_payload.extend_from_slice(SAMPLE_1X1_AV1_OBU);
    let off3 = idat_payload.len() as u32;
    idat_payload.extend_from_slice(&tmap_payload);
    let idat = make_box(b"idat", &idat_payload);

    let mut iloc_body = vec![0x44u8, 0x00];
    iloc_body.extend_from_slice(&3u16.to_be_bytes());
    for (id, off, len) in [
        (1u16, off1, SAMPLE_1X1_AV1_OBU.len() as u32),
        (2u16, off2, SAMPLE_1X1_AV1_OBU.len() as u32),
        (3u16, off3, tmap_payload.len() as u32),
    ] {
        iloc_body.extend_from_slice(&id.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&off.to_be_bytes());
        iloc_body.extend_from_slice(&len.to_be_bytes());
    }
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

    let mut ftyp_gm_payload = Vec::new();
    ftyp_gm_payload.extend_from_slice(b"avif");
    ftyp_gm_payload.extend_from_slice(&0u32.to_be_bytes());
    ftyp_gm_payload.extend_from_slice(b"mif1");
    ftyp_gm_payload.extend_from_slice(b"tmap");
    let ftyp_gm = make_box(b"ftyp", &ftyp_gm_payload);

    let mut altr_body = Vec::new();
    altr_body.extend_from_slice(&10u32.to_be_bytes()); // group_id
    altr_body.extend_from_slice(&2u32.to_be_bytes()); // num_entities_in_group
    altr_body.extend_from_slice(&3u32.to_be_bytes()); // tmap (preferred)
    altr_body.extend_from_slice(&1u32.to_be_bytes()); // primary
    let grpl = make_box(b"grpl", &make_full_box(b"altr", 0, 0, &altr_body));

    let mut meta_gm = Vec::new();
    for b in [&hdlr, &pitm, &iinf, &iref, &grpl, &iprp, &iloc, &idat] {
        meta_gm.extend_from_slice(b);
    }
    let mut avif_gm = ftyp_gm;
    avif_gm.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_gm));

    // Tam çözümle: hem ana ICC hem iki `alt_icc_profile` mevcut ve `allocation_peak` ikisini de saymalı.
    let cfg_full = AvifDecodeConfig {
        decode_gain_map: true,
        n_threads: 1,
        ..AvifDecodeConfig::default()
    };
    let img_full = decode_with(&avif_gm, &cfg_full).expect("GainMap AVIF çözülmeli");
    assert_eq!(img_full.icc_profile().map(<[u8]>::len), Some(icc_size));
    assert_eq!(
        img_full
            .gain_map_metadata()
            .and_then(|m| m.alt_icc_profile.as_ref())
            .map(Vec::len),
        Some(icc_size)
    );
    assert_eq!(
        img_full
            .gain_map()
            .and_then(|g| g.metadata.alt_icc_profile.as_ref())
            .map(Vec::len),
        Some(icc_size)
    );

    // `decode_gain_map = false` ile başarı eşiği vs `decode_gain_map = true` ile eşik farkı:
    // ikinci `alt_icc_profile` (3 144 B) + `size_of::<AvifImage>()` bütçeye eklendiği için
    // `decode_gain_map = false`'un geçtiği en sıkı sınırdan +2048 B eklenmiş sınırda
    // `decode_gain_map = true` ikinci `3 144 B` kopyayı ayırmadan `BudgetExceeded` vermelidir!
    let mut lo = 1024usize;
    let mut hi = 4 * 1024 * 1024usize;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let c = AvifDecodeConfig {
            decode_gain_map: false,
            n_threads: 1,
            allocation_limit: Some(mid),
            ..AvifDecodeConfig::default()
        };
        if decode_with(&avif_gm, &c).is_ok() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let min_without_gm = lo;

    let mut lo2 = min_without_gm;
    let mut hi2 = 4 * 1024 * 1024usize;
    while lo2 < hi2 {
        let mid = lo2 + (hi2 - lo2) / 2;
        let c = AvifDecodeConfig {
            decode_gain_map: true,
            n_threads: 1,
            allocation_limit: Some(mid),
            ..AvifDecodeConfig::default()
        };
        if decode_with(&avif_gm, &c).is_ok() {
            hi2 = mid;
        } else {
            lo2 = mid + 1;
        }
    }
    let min_with_gm = lo2;
    assert!(
        min_with_gm >= min_without_gm + icc_size,
        "ikinci GainMap alt_icc_profile ({icc_size} B) bütçe eşiğine yansımalı: without={min_without_gm}, with={min_with_gm}"
    );

    // 2. 4×4 Grid AVIF (hem tekrarsız `!has_dups` renk ızgarası hem 16 renk karosunun
    //    ortak bir alfa karosuna `auxl` ile bağlandığı `has_dups` alfa ızgarası):
    //    bütçe altında hem başarı hem sıkı sınırda `BudgetExceeded` doğrulanır.
    let grid_desc: [u8; 8] = [0, 0, 3, 3, 0, 4, 0, 4]; // 4x4 grid, 4x4 çıktı
    let mut iinf_g_body = 18u16.to_be_bytes().to_vec();
    iinf_g_body.extend_from_slice(&make_infe(1, b"grid")); // primary color grid
    for id in 2u16..=17u16 {
        iinf_g_body.extend_from_slice(&make_infe(id, b"av01")); // 16 color tiles
    }
    iinf_g_body.extend_from_slice(&make_infe(18, b"av01")); // 1 shared alpha tile
    let iinf_g = make_full_box(b"iinf", 0, 0, &iinf_g_body);

    let mut dimg_color = Vec::new();
    dimg_color.extend_from_slice(&1u16.to_be_bytes());
    dimg_color.extend_from_slice(&16u16.to_be_bytes());
    for id in 2u16..=17u16 {
        dimg_color.extend_from_slice(&id.to_be_bytes());
    }
    let mut auxl_alpha = Vec::new();
    auxl_alpha.extend_from_slice(&18u16.to_be_bytes()); // from shared alpha tile 18
    auxl_alpha.extend_from_slice(&16u16.to_be_bytes()); // to all 16 color tiles
    for id in 2u16..=17u16 {
        auxl_alpha.extend_from_slice(&id.to_be_bytes());
    }
    let mut iref_g_body = make_box(b"dimg", &dimg_color);
    iref_g_body.extend_from_slice(&make_box(b"auxl", &auxl_alpha));
    let iref_g = make_full_box(b"iref", 0, 0, &iref_g_body);

    let mut ispe4 = Vec::new();
    ispe4.extend_from_slice(&4u32.to_be_bytes());
    ispe4.extend_from_slice(&4u32.to_be_bytes());
    let mut ispe1 = Vec::new();
    ispe1.extend_from_slice(&1u32.to_be_bytes());
    ispe1.extend_from_slice(&1u32.to_be_bytes());
    let auxc_alpha = make_full_box(
        b"auxC",
        0,
        0,
        b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0",
    );
    let mut ipco_g = make_full_box(b"ispe", 0, 0, &ispe4); // prop 1
    ipco_g.extend_from_slice(&make_full_box(b"ispe", 0, 0, &ispe1)); // prop 2
    ipco_g.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00])); // prop 3
    ipco_g.extend_from_slice(&auxc_alpha); // prop 4
    let ipco_g_box = make_box(b"ipco", &ipco_g);

    let mut ipma_g_body = 18u32.to_be_bytes().to_vec();
    ipma_g_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_g_body.push(1);
    ipma_g_body.push(0x01);
    for id in 2u16..=17u16 {
        ipma_g_body.extend_from_slice(&id.to_be_bytes());
        ipma_g_body.push(2);
        ipma_g_body.extend_from_slice(&[0x02, 0x83]);
    }
    ipma_g_body.extend_from_slice(&18u16.to_be_bytes());
    ipma_g_body.push(3);
    ipma_g_body.extend_from_slice(&[0x02, 0x83, 0x84]);
    let ipma_g = make_full_box(b"ipma", 0, 0, &ipma_g_body);
    let iprp_g = make_box(
        b"iprp",
        &[ipco_g_box.as_slice(), ipma_g.as_slice()].concat(),
    );

    let mut idat_g_payload = Vec::new();
    idat_g_payload.extend_from_slice(&grid_desc);
    let tile_off = idat_g_payload.len() as u32;
    idat_g_payload.extend_from_slice(SAMPLE_1X1_AV1_OBU);
    let idat_g = make_box(b"idat", &idat_g_payload);

    let mut iloc_g_body = vec![0x44u8, 0x00];
    iloc_g_body.extend_from_slice(&18u16.to_be_bytes());
    for id in 1u16..=18u16 {
        let (off, len) = if id == 1 {
            (0u32, grid_desc.len() as u32)
        } else {
            (tile_off, SAMPLE_1X1_AV1_OBU.len() as u32)
        };
        iloc_g_body.extend_from_slice(&id.to_be_bytes());
        iloc_g_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_g_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_g_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_g_body.extend_from_slice(&off.to_be_bytes());
        iloc_g_body.extend_from_slice(&len.to_be_bytes());
    }
    let iloc_g = make_full_box(b"iloc", 1, 0, &iloc_g_body);

    let mut meta_g = Vec::new();
    for b in [&hdlr, &pitm, &iinf_g, &iref_g, &iprp_g, &iloc_g, &idat_g] {
        meta_g.extend_from_slice(b);
    }
    let mut avif_grid = ftyp;
    avif_grid.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_g));

    let img_grid = decode_with(
        &avif_grid,
        &AvifDecodeConfig {
            n_threads: 1,
            ..AvifDecodeConfig::default()
        },
    )
    .expect("4x4 renk + tekrarlı alfa grid çözülmeli");
    assert_eq!((img_grid.width(), img_grid.height()), (4, 4));
    assert!(img_grid.alpha_plane().is_some());

    let (res_tight_grid, max_tight_grid, _) = measure_allocs(|| {
        decode_with(
            &avif_grid,
            &AvifDecodeConfig {
                n_threads: 1,
                allocation_limit: Some(1024),
                ..AvifDecodeConfig::default()
            },
        )
    });
    assert!(matches!(res_tight_grid, Err(Error::BudgetExceeded { .. })));
    assert!(
        max_tight_grid <= 4096,
        "sıkı bütçeli grid reddi aşırı tekil tahsis yapmamalı: {max_tight_grid}"
    );
}
