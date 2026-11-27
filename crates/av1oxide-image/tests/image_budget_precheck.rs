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

fn build_rich_avif_with_iref_grpl_tmap_and_props() -> Vec<u8> {
    let mut ftyp_payload = Vec::new();
    ftyp_payload.extend_from_slice(b"avif");
    ftyp_payload.extend_from_slice(&0u32.to_be_bytes());
    ftyp_payload.extend_from_slice(b"avifmif1miaftmap");
    let ftyp = make_box(b"ftyp", &ftyp_payload);

    let mut hdlr_body = vec![0u8; 4];
    hdlr_body.extend_from_slice(b"pict");
    hdlr_body.extend_from_slice(&[0u8; 13]);
    let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());

    let make_infe = |id: u16, ty: &[u8; 4]| -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&id.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(ty);
        b.push(0);
        make_full_box(b"infe", 2, 0, &b)
    };
    let mut iinf_b = 3u16.to_be_bytes().to_vec();
    iinf_b.extend_from_slice(&make_infe(1, b"av01"));
    iinf_b.extend_from_slice(&make_infe(2, b"av01"));
    iinf_b.extend_from_slice(&make_infe(3, b"tmap"));
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_b);

    let mut dimg_b = Vec::new();
    dimg_b.extend_from_slice(&3u16.to_be_bytes());
    dimg_b.extend_from_slice(&2u16.to_be_bytes());
    dimg_b.extend_from_slice(&1u16.to_be_bytes());
    dimg_b.extend_from_slice(&2u16.to_be_bytes());
    let iref = make_full_box(b"iref", 0, 0, &make_box(b"dimg", &dimg_b));

    let mut altr_b = Vec::new();
    altr_b.extend_from_slice(&1u32.to_be_bytes());
    altr_b.extend_from_slice(&2u32.to_be_bytes());
    altr_b.extend_from_slice(&3u32.to_be_bytes());
    altr_b.extend_from_slice(&1u32.to_be_bytes());
    let grpl = make_box(b"grpl", &make_full_box(b"altr", 0, 0, &altr_b));

    let mut ispe_b = Vec::new();
    ispe_b.extend_from_slice(&1u32.to_be_bytes());
    ispe_b.extend_from_slice(&1u32.to_be_bytes());
    let mut colr_nclx = Vec::new();
    colr_nclx.extend_from_slice(b"nclx");
    colr_nclx.extend_from_slice(&1u16.to_be_bytes());
    colr_nclx.extend_from_slice(&13u16.to_be_bytes());
    colr_nclx.extend_from_slice(&6u16.to_be_bytes());
    colr_nclx.push(0x80);
    let mut pasp_b = Vec::new();
    pasp_b.extend_from_slice(&1u32.to_be_bytes());
    pasp_b.extend_from_slice(&1u32.to_be_bytes());
    let mut ipco_b = Vec::new();
    ipco_b.extend_from_slice(&make_full_box(b"ispe", 0, 0, &ispe_b));
    ipco_b.extend_from_slice(&make_full_box(b"pixi", 0, 0, &[3, 8, 8, 8]));
    ipco_b.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0c, 0x00]));
    ipco_b.extend_from_slice(&make_box(b"colr", &colr_nclx));
    ipco_b.extend_from_slice(&make_box(b"pasp", &pasp_b));
    let ipco = make_box(b"ipco", &ipco_b);

    let mut ipma_b = 3u32.to_be_bytes().to_vec();
    ipma_b.extend_from_slice(&1u16.to_be_bytes());
    ipma_b.push(5);
    ipma_b.extend_from_slice(&[1, 2, 0x80 | 3, 4, 5]);
    ipma_b.extend_from_slice(&2u16.to_be_bytes());
    ipma_b.push(4);
    ipma_b.extend_from_slice(&[1, 2, 0x80 | 3, 4]);
    ipma_b.extend_from_slice(&3u16.to_be_bytes());
    ipma_b.push(3);
    ipma_b.extend_from_slice(&[1, 2, 4]);
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_b);
    let iprp = make_box(b"iprp", &[ipco.as_slice(), ipma.as_slice()].concat());

    let mut tmap_payload = vec![0u8, 0, 0, 0, 0, 0];
    tmap_payload.extend_from_slice(&0u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
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

    let mut idat_payload = Vec::new();
    let off1 = 0u32;
    idat_payload.extend_from_slice(AV1_1X1_8BIT);
    let off2 = idat_payload.len() as u32;
    idat_payload.extend_from_slice(AV1_1X1_8BIT);
    let off3 = idat_payload.len() as u32;
    idat_payload.extend_from_slice(&tmap_payload);
    let idat = make_box(b"idat", &idat_payload);

    let mut iloc_b = vec![0x44u8, 0x00];
    iloc_b.extend_from_slice(&3u16.to_be_bytes());
    for (id, off, len) in [
        (1u16, off1, AV1_1X1_8BIT.len() as u32),
        (2u16, off2, AV1_1X1_8BIT.len() as u32),
        (3u16, off3, tmap_payload.len() as u32),
    ] {
        iloc_b.extend_from_slice(&id.to_be_bytes());
        iloc_b.extend_from_slice(&1u16.to_be_bytes());
        iloc_b.extend_from_slice(&0u16.to_be_bytes());
        iloc_b.extend_from_slice(&1u16.to_be_bytes());
        iloc_b.extend_from_slice(&off.to_be_bytes());
        iloc_b.extend_from_slice(&len.to_be_bytes());
    }
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_b);

    let meta = make_full_box(
        b"meta",
        0,
        0,
        &[
            hdlr.as_slice(),
            pitm.as_slice(),
            iinf.as_slice(),
            iref.as_slice(),
            grpl.as_slice(),
            iprp.as_slice(),
            iloc.as_slice(),
            idat.as_slice(),
        ]
        .concat(),
    );
    [ftyp.as_slice(), meta.as_slice()].concat()
}

fn check_threshold_sweep(label: &str, data: &[u8]) {
    let mut lo = data.len();
    let mut hi = data.len().saturating_add(16 * 1024 * 1024);
    let probe = |budget: usize| -> (bool, usize, usize) {
        let cfg = AvifDecodeConfig {
            n_threads: 1,
            allocation_limit: Some(budget),
            ..AvifDecodeConfig::default()
        };
        let (res, peak, largest) =
            measure_allocs(|| AvifDecoder::with_config(Cursor::new(data), cfg));
        let ok = res.is_ok();
        drop(res);
        (ok, peak, largest)
    };
    let (ok_hi, peak_hi, largest_hi) = probe(hi);
    assert!(
        peak_hi <= hi && largest_hi <= hi,
        "{label}: hi={hi} peak={peak_hi} largest={largest_hi}"
    );
    if !ok_hi {
        return;
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (ok, peak, largest) = probe(mid);
        assert!(
            peak <= mid && largest <= mid,
            "{label}: ikili arama mid={mid} ok={ok} peak={peak} largest={largest}"
        );
        if ok {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    for budget in [data.len(), data.len() + 1]
        .into_iter()
        .chain(hi.saturating_sub(32)..=hi + 64)
    {
        let (ok, peak, largest) = probe(budget);
        assert!(
            peak <= budget && largest <= budget,
            "{label}: budget={budget} (hi={hi}, ok={ok}) peak={peak} largest={largest}"
        );
    }
    for budget in [hi.saturating_sub(1), hi] {
        let cfg = AvifDecodeConfig {
            n_threads: 1,
            allocation_limit: Some(budget),
            ..AvifDecodeConfig::default()
        };
        let (res, peak, largest) = measure_allocs(|| {
            AvifDecoder::with_config(
                ChunkedReader {
                    data,
                    chunk_size: 256,
                },
                cfg,
            )
        });
        assert!(
            peak <= budget && largest <= budget,
            "{label}: chunk=256 budget={budget} (hi={hi}, ok={}) peak={peak} largest={largest}",
            res.is_ok()
        );
        drop(res);
    }
}

/// `6.I.1` (`P2-2`): Sentetik (`iref`/`grpl`/`tmap`/`ipma`, `cmin` `zlib`/`deflate`/`gzip`,
/// `AvifSequenceDecoder` `.avis`, `read_image`) ve varsa izinli `libavif/tests/data`
/// korpusundaki (`colors-animated-8bpc.avif` dâhil) tüm AVIF dosyalarında ikili-arama eşiği `hi`
/// etrafında (`[len, len+1, hi-32..=hi+64]`) `peak_live <= limit` ve `max_single <= limit` korunur.
#[test]
fn with_config_strict_threshold_sweep_never_exceeds_allocation_limit() {
    let synth_simple = build_padded_avif(1_235, None);
    check_threshold_sweep("synthetic_1235", &synth_simple);

    let synth_rich = build_rich_avif_with_iref_grpl_tmap_and_props();
    check_threshold_sweep("synthetic_rich_iref_grpl_tmap", &synth_rich);

    // 1. Sentetik `cmin` (`zlib`, `deflate`, `gzip`) AVIF üzerinde hem `limit = 4096`
    //    (`InflateState` ~43 KiB ayrılmadan önce reddedilmeli!) hem de ikili-arama `hi` eşiği taraması.
    let ftyp_end = u32::from_be_bytes(synth_simple[0..4].try_into().unwrap()) as usize;
    let ftyp_slice = &synth_simple[..ftyp_end];
    let meta_len =
        u32::from_be_bytes(synth_simple[ftyp_end..ftyp_end + 4].try_into().unwrap()) as usize;
    let meta_payload = &synth_simple[ftyp_end + 8..ftyp_end + meta_len];

    let encode_deflate_stored = |raw: &[u8]| -> Vec<u8> {
        let len = raw.len() as u16;
        let nlen = !len;
        let mut out = Vec::with_capacity(5 + raw.len());
        out.push(0x01); // BFINAL=1, BTYPE=00 (stored)
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(raw);
        out
    };
    let adler32 = |raw: &[u8]| -> u32 {
        let mut a = 1u32;
        let mut b = 0u32;
        for &byte in raw {
            a = (a + u32::from(byte)) % 65_521;
            b = (b + a) % 65_521;
        }
        (b << 16) | a
    };
    let crc32 = |raw: &[u8]| -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &byte in raw {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    };

    for (fourcc, label) in [
        (b"zlib", "cmin_zlib"),
        (b"defl", "cmin_defl"),
        (b"gzip", "cmin_gzip"),
    ] {
        let deflate_block = encode_deflate_stored(meta_payload);
        let compressed = match fourcc {
            b"zlib" => {
                let mut v = vec![0x78, 0x01];
                v.extend_from_slice(&deflate_block);
                v.extend_from_slice(&adler32(meta_payload).to_be_bytes());
                v
            }
            b"defl" => deflate_block,
            _ => {
                let mut v = vec![0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff];
                v.extend_from_slice(&deflate_block);
                v.extend_from_slice(&crc32(meta_payload).to_le_bytes());
                v.extend_from_slice(&(meta_payload.len() as u32).to_le_bytes());
                v
            }
        };
        let mut cmin_body = Vec::with_capacity(4 + compressed.len());
        cmin_body.extend_from_slice(fourcc);
        cmin_body.extend_from_slice(&compressed);
        let cmin_box = make_full_box(b"cmin", 0, 0, &cmin_body);
        let cmin_avif = [ftyp_slice, cmin_box.as_slice()].concat();

        let (res_4k, peak_4k, max_4k) = measure_allocs(|| {
            AvifDecoder::with_config(
                Cursor::new(&cmin_avif),
                AvifDecodeConfig {
                    n_threads: 1,
                    allocation_limit: Some(4_096),
                    ..AvifDecodeConfig::default()
                },
            )
        });
        assert!(
            res_4k.is_err() && peak_4k <= 4_096 && max_4k <= 4_096,
            "{label} limit=4096: ok={} peak={peak_4k} max={max_4k}",
            res_4k.is_ok()
        );
        check_threshold_sweep(label, &cmin_avif);
    }

    // 2. `AvifDecoder::with_config` + `read_image` uçtan uca ikili-arama eşiği (`hi-16..=hi+32`).
    let mut lo_ri = synth_simple.len();
    let mut hi_ri = 512 * 1024usize;
    let probe_read_image = |budget: usize| -> (bool, usize, usize) {
        let mut out_rgb = [0u8; 3];
        let (res, peak, largest) = measure_allocs(|| -> Result<(), ImageError> {
            let dec = AvifDecoder::with_config(
                Cursor::new(&synth_simple),
                AvifDecodeConfig {
                    n_threads: 1,
                    allocation_limit: Some(budget),
                    ..AvifDecodeConfig::default()
                },
            )?;
            dec.read_image(&mut out_rgb)
        });
        (res.is_ok(), peak, largest)
    };
    while lo_ri < hi_ri {
        let mid = lo_ri + (hi_ri - lo_ri) / 2;
        let (ok, peak, largest) = probe_read_image(mid);
        assert!(
            peak <= mid && largest <= mid,
            "read_image mid={mid} ok={ok} peak={peak} largest={largest}"
        );
        if ok {
            hi_ri = mid;
        } else {
            lo_ri = mid + 1;
        }
    }
    for budget in hi_ri.saturating_sub(16)..=hi_ri + 32 {
        let (ok, peak, largest) = probe_read_image(budget);
        assert!(
            peak <= budget && largest <= budget,
            "read_image budget={budget} (hi={hi_ri}, ok={ok}) peak={peak} largest={largest}"
        );
    }

    // 3. `AvifSequenceDecoder::with_config` üzerinde `top_meta` içermeyen saf `.avis` ikili-arama eşiği.
    let build_pure_avis = || -> Vec<u8> {
        let mut ftyp_p = Vec::new();
        ftyp_p.extend_from_slice(b"avis");
        ftyp_p.extend_from_slice(&0u32.to_be_bytes());
        ftyp_p.extend_from_slice(b"msf1avifmiaf");
        let ftyp = make_box(b"ftyp", &ftyp_p);
        let mut mvhd_b = vec![0u8; 96];
        mvhd_b[8..12].copy_from_slice(&30u32.to_be_bytes());
        mvhd_b[12..16].copy_from_slice(&1u32.to_be_bytes());
        let mvhd = make_full_box(b"mvhd", 0, 0, &mvhd_b);
        let make_trak = |off: u32| -> Vec<u8> {
            let mut tkhd_b = vec![0u8; 80];
            tkhd_b[8..12].copy_from_slice(&1u32.to_be_bytes());
            tkhd_b[16..20].copy_from_slice(&1u32.to_be_bytes());
            tkhd_b[36..40].copy_from_slice(&0x0001_0000i32.to_be_bytes());
            tkhd_b[52..56].copy_from_slice(&0x0001_0000i32.to_be_bytes());
            tkhd_b[68..72].copy_from_slice(&0x4000_0000i32.to_be_bytes());
            tkhd_b[72..76].copy_from_slice(&(1u32 << 16).to_be_bytes());
            tkhd_b[76..80].copy_from_slice(&(1u32 << 16).to_be_bytes());
            let tkhd = make_full_box(b"tkhd", 0, 3, &tkhd_b);
            let mut mdhd_b = vec![0u8; 20];
            mdhd_b[8..12].copy_from_slice(&30u32.to_be_bytes());
            mdhd_b[12..16].copy_from_slice(&1u32.to_be_bytes());
            let mdhd = make_full_box(b"mdhd", 0, 0, &mdhd_b);
            let mut hdlr_b = vec![0u8; 4];
            hdlr_b.extend_from_slice(b"pict");
            hdlr_b.extend_from_slice(&[0u8; 13]);
            let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_b);
            let mut av01_p = vec![0u8; 78];
            av01_p[24..26].copy_from_slice(&1u16.to_be_bytes());
            av01_p[26..28].copy_from_slice(&1u16.to_be_bytes());
            av01_p.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0c, 0x00]));
            let mut stsd_b = 1u32.to_be_bytes().to_vec();
            stsd_b.extend_from_slice(&make_box(b"av01", &av01_p));
            let stsd = make_full_box(b"stsd", 0, 0, &stsd_b);
            let mut stts_b = Vec::new();
            stts_b.extend_from_slice(&1u32.to_be_bytes());
            stts_b.extend_from_slice(&1u32.to_be_bytes());
            stts_b.extend_from_slice(&1u32.to_be_bytes());
            let stts = make_full_box(b"stts", 0, 0, &stts_b);
            let mut stsc_b = Vec::new();
            for _ in 0..4 {
                stsc_b.extend_from_slice(&1u32.to_be_bytes());
            }
            let stsc = make_full_box(b"stsc", 0, 0, &stsc_b);
            let mut stsz_b = Vec::new();
            stsz_b.extend_from_slice(&(AV1_1X1_8BIT.len() as u32).to_be_bytes());
            stsz_b.extend_from_slice(&1u32.to_be_bytes());
            let stsz = make_full_box(b"stsz", 0, 0, &stsz_b);
            let mut stco_b = 1u32.to_be_bytes().to_vec();
            stco_b.extend_from_slice(&off.to_be_bytes());
            let stco = make_full_box(b"stco", 0, 0, &stco_b);
            let stbl = make_box(
                b"stbl",
                &[
                    stsd.as_slice(),
                    stts.as_slice(),
                    stsc.as_slice(),
                    stsz.as_slice(),
                    stco.as_slice(),
                ]
                .concat(),
            );
            let minf = make_box(b"minf", &stbl);
            let mdia = make_box(
                b"mdia",
                &[mdhd.as_slice(), hdlr.as_slice(), minf.as_slice()].concat(),
            );
            make_box(b"trak", &[tkhd.as_slice(), mdia.as_slice()].concat())
        };
        let dummy_moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(0)].concat());
        let data_off = (ftyp.len() + dummy_moov.len() + 8) as u32;
        let moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(data_off)].concat());
        let mdat = make_box(b"mdat", AV1_1X1_8BIT);
        [ftyp.as_slice(), moov.as_slice(), mdat.as_slice()].concat()
    };
    let check_seq_sweep = |label: &str, seq_bytes: &[u8]| {
        let mut lo = 0usize;
        let mut hi = 256 * 1024usize;
        let probe_seq = |budget: usize| -> (bool, usize, usize) {
            let cfg = AvifDecodeConfig {
                n_threads: 1,
                allocation_limit: Some(budget),
                ..AvifDecodeConfig::default()
            };
            let (res, peak, largest) =
                measure_allocs(|| av1oxide::avif::AvifSequenceDecoder::with_config(seq_bytes, cfg));
            let ok = res.is_ok();
            drop(res);
            (ok, peak, largest)
        };
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (ok, peak, largest) = probe_seq(mid);
            assert!(
                peak <= mid && largest <= mid,
                "{label} seq mid={mid} ok={ok} peak={peak} largest={largest}"
            );
            if ok {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        for budget in hi.saturating_sub(16)..=hi + 32 {
            let (ok, peak, largest) = probe_seq(budget);
            assert!(
                peak <= budget && largest <= budget,
                "{label} seq budget={budget} (hi={hi}, ok={ok}) peak={peak} largest={largest}"
            );
        }
    };
    let pure_avis = build_pure_avis();
    check_seq_sweep("pure_avis_no_top_meta", &pure_avis);

    let ws_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    let mut candidates = vec![
        ws_root.join(".cache/libavif-src/tests/data"),
        ws_root.join(".cache/libavif/tests/data"),
    ];
    if let Some(parent) = ws_root.parent() {
        candidates
            .push(parent.join("av1oxide_planlar/avif_dogrulama/kaynaklar/libavif/tests/data"));
    }
    let corpus_dir = candidates.into_iter().find(|p| p.exists());
    let Some(corpus_dir) = corpus_dir else {
        return;
    };

    let is_forbidden = |name: &str| -> bool {
        let lower = name.to_ascii_lowercase();
        lower.starts_with("arc_triomphe")
            || lower.starts_with("cosmos")
            || lower.contains("chimera")
            || lower.contains("netflix")
    };

    let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)
        .expect("read_dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("avif"))
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| !is_forbidden(n))
        })
        .collect();
    entries.sort();

    for path in &entries {
        let name = path.file_name().unwrap().to_string_lossy();
        let data = std::fs::read(path).expect("read avif");
        if name == "colors-animated-8bpc.avif" {
            let (res_2275, peak_2275, max_2275) = measure_allocs(|| {
                AvifDecoder::with_config(
                    Cursor::new(&data),
                    AvifDecodeConfig {
                        n_threads: 1,
                        allocation_limit: Some(2_275),
                        ..AvifDecodeConfig::default()
                    },
                )
            });
            assert!(
                peak_2275 <= 2_275 && max_2275 <= 2_275,
                "colors-animated-8bpc.avif limit=2275: ok={} peak={peak_2275} max={max_2275}",
                res_2275.is_ok()
            );
            drop(res_2275);
            check_seq_sweep("colors-animated-8bpc.avif", &data);
        }
        check_threshold_sweep(&name, &data);
    }
    assert!(
        entries.len() >= 50,
        "izinli libavif korpusunda en az 50 .avif dosyası taranmalı, taranan: {}",
        entries.len()
    );
}
