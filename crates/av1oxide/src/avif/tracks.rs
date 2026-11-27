//! ISOBMFF (`ISO/IEC 14496-12` / `ISO/IEC 23008-12`) animasyonlu AVIF (`.avis`) parça
//! (`moov` / `trak` / `stbl`) ayrıştırıcısı.

#![forbid(unsafe_code)]

use alloc::borrow::Cow;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::avif::boxes::{
    ALPHA_AUX_URN_CICP, ALPHA_AUX_URN_HEVC, AvifContainer, ItemProperty, decompress_by_fourcc,
    is_supported_brand, parse_meta_box, parse_property_boxes, read_box_header, read_full_box,
    read_i16_be, read_i32_be, read_i64_be, read_u32_be, read_u64_be,
};
use crate::error::{Error, Result};
use crate::frame::butce::{Butce, ButceTutucu};

/// `VisualSampleEntry` sabit başlık uzunluğu (ISO/IEC 14496-12 §12.1.3.2).
const VISUAL_SAMPLE_ENTRY_SIZE: usize = 78;
/// Tek bir parça için bellek taşmasını önleyen varsayılan örnek sayısı üst sınırı.
const MAX_TRACK_SAMPLES: usize = 1_000_000;
/// `tkhd` v0 belirsiz süre belirteci (`UINT32_MAX`).
const INDEFINITE_DURATION_32: u32 = u32::MAX;
/// `tkhd` v1 belirsiz süre belirteci (`UINT64_MAX`).
const INDEFINITE_DURATION_64: u64 = u64::MAX;

/// Animasyonlu AVIF (`.avis`) tekrarlama (döngü) sayısı.
///
/// `libavif` `repetitionCount` sözleşmesiyle uyumludur:
/// - `LoopCount::Count(0)`: görüntü dizisi ek tekrar olmadan yalnızca bir kez oynatılır.
/// - `LoopCount::Count(n)`: görüntü dizisi toplam `n + 1` kez oynatılır.
/// - `LoopCount::Infinite`: görüntü dizisi sonsuz döngüde oynatılır (`AVIF_REPETITION_COUNT_INFINITE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopCount {
    /// Sonsuz tekrar (`AVIF_REPETITION_COUNT_INFINITE`).
    Infinite,
    /// Belirli sayıda ek tekrar (`0` = bir kez oynatma).
    Count(u32),
}

/// Parça örnek tablosundan (`stco`/`co64` + `stsc` + `stsz` + `stts` + `stss`) çözümlenen tekil örnek.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrackSample {
    pub(crate) offset: usize,
    pub(crate) size: usize,
    pub(crate) pts_in_timescale: u64,
    pub(crate) duration_in_timescale: u64,
    pub(crate) is_keyframe: bool,
}

/// `tref` kutusu içindeki parça referansı (`auxl`, `prem`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrackReference {
    pub(crate) ref_type: [u8; 4],
    pub(crate) to_track_id: u32,
}

#[derive(Clone, Copy, Debug)]
struct SampleToChunk {
    first_chunk: u32,
    samples_per_chunk: u32,
}

#[derive(Clone, Copy, Debug)]
struct TimeToSample {
    sample_count: u32,
    sample_delta: u32,
}

#[derive(Clone, Copy, Debug)]
struct CompositionOffset {
    sample_count: u32,
    sample_offset: i64,
}

#[derive(Clone, Debug, Default)]
struct RawSampleTable {
    chunks: Vec<u64>,
    sample_to_chunks: Vec<SampleToChunk>,
    all_samples_size: u32,
    sample_sizes: Vec<u32>,
    time_to_samples: Vec<TimeToSample>,
    composition_offsets: Vec<CompositionOffset>,
    has_stss: bool,
    sync_samples: Vec<u32>,
    is_av01: bool,
    properties: Vec<ItemProperty>,
}

/// `elst` (`EditListBox`) içindeki tekil zaman çizelgesi girdisi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EditListEntry {
    pub(crate) segment_duration: u64,
    pub(crate) media_time: i64,
    pub(crate) media_rate_integer: i16,
    pub(crate) media_rate_fraction: i16,
}

/// Ayrıştırılmış `elst` kutusu.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EditList {
    pub(crate) flags: u32,
    pub(crate) entries: Vec<EditListEntry>,
}

/// `elst` zaman çizelgesinden üretilen tekil sunum karesi kaydı.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PresentedSample {
    pub(crate) raw_index: usize,
    pub(crate) pts_in_timescale: u64,
    pub(crate) duration_in_timescale: u64,
}

/// `tkhd` 3×3 dönüşüm matrisinin birim 2×2 $D_4$ alt matrisi (`[[a, b], [c, d]]`, elemanlar `{-1, 0, 1}`).
pub(crate) const TKHD_IDENTITY_MATRIX: [[i8; 2]; 2] = [[1, 0], [0, 1]];

/// `irot`/`imir` çifti ile `tkhd` 2×2 $D_4$ matrisini (`stsd` → `tkhd` sırasıyla) birleştirir.
pub(crate) fn compose_d4_transforms(
    stsd_irot: Option<u8>,
    stsd_imir: Option<u8>,
    tkhd_mat: [[i8; 2]; 2],
) -> (Option<u8>, Option<u8>) {
    let mul = |lhs: [[i8; 2]; 2], rhs: [[i8; 2]; 2]| -> [[i8; 2]; 2] {
        [
            [
                lhs[0][0] * rhs[0][0] + lhs[0][1] * rhs[1][0],
                lhs[0][0] * rhs[0][1] + lhs[0][1] * rhs[1][1],
            ],
            [
                lhs[1][0] * rhs[0][0] + lhs[1][1] * rhs[1][0],
                lhs[1][0] * rhs[0][1] + lhs[1][1] * rhs[1][1],
            ],
        ]
    };
    let rot_mat = match stsd_irot.map(|a| a & 3) {
        Some(1) => [[0, -1], [1, 0]],
        Some(2) => [[-1, 0], [0, -1]],
        Some(3) => [[0, 1], [-1, 0]],
        _ => TKHD_IDENTITY_MATRIX,
    };
    let mir_mat = match stsd_imir.map(|m| m & 1) {
        Some(0) => [[1, 0], [0, -1]],
        Some(1) => [[-1, 0], [0, 1]],
        _ => TKHD_IDENTITY_MATRIX,
    };
    let combined = mul(mul(rot_mat, mir_mat), tkhd_mat);
    match combined {
        [[1, 0], [0, 1]] => (None, None),
        [[0, -1], [1, 0]] => (Some(1), None),
        [[-1, 0], [0, -1]] => (Some(2), None),
        [[0, 1], [-1, 0]] => (Some(3), None),
        [[1, 0], [0, -1]] => (None, Some(0)),
        [[-1, 0], [0, 1]] => (None, Some(1)),
        [[0, 1], [1, 0]] => (Some(1), Some(0)),
        [[0, -1], [-1, 0]] => (Some(1), Some(1)),
        _ => (None, None),
    }
}

/// Ayrıştırılmış tekil ISOBMFF `trak` kaydı.
#[derive(Clone, Debug)]
pub(crate) struct ParsedTrack<'a> {
    pub(crate) track_id: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tkhd_matrix: [[i8; 2]; 2],
    pub(crate) handler_type: [u8; 4],
    pub(crate) media_timescale: u64,
    pub(crate) media_duration: u64,
    pub(crate) movie_timescale: u64,
    pub(crate) edit_list: Option<EditList>,
    pub(crate) repetition_count: LoopCount,
    pub(crate) is_av01: bool,
    pub(crate) properties: Vec<ItemProperty>,
    pub(crate) references: Vec<TrackReference>,
    pub(crate) samples: Vec<TrackSample>,
    pub(crate) meta: Option<AvifContainer<'a>>,
    pub(crate) _guard: Arc<ButceTutucu>,
}

impl ParsedTrack<'_> {
    pub(crate) fn has_reference_type(&self, ref_type: &[u8; 4]) -> bool {
        self.references.iter().any(|r| &r.ref_type == ref_type)
    }

    pub(crate) fn references_track(&self, ref_type: &[u8; 4], target_id: u32) -> bool {
        self.references
            .iter()
            .any(|r| &r.ref_type == ref_type && r.to_track_id == target_id)
    }
}

/// Ayrıştırılmış üst düzey ISOBMFF kapsayıcısı (`ftyp` + isteğe bağlı `moov` + isteğe bağlı `meta`).
#[derive(Debug)]
pub(crate) struct ParsedMovie<'a> {
    pub(crate) major_brand: [u8; 4],
    #[allow(dead_code)]
    pub(crate) compatible_brands: Vec<[u8; 4]>,
    pub(crate) tracks: Vec<ParsedTrack<'a>>,
    pub(crate) top_meta: Option<AvifContainer<'a>>,
    pub(crate) _guard: ButceTutucu,
}

impl<'a> ParsedMovie<'a> {
    /// `libavif` `AVIF_DECODER_SOURCE_AUTO` kuralına göre parça (`.avis`) kipinin seçilip seçilmeyeceği.
    pub(crate) fn prefers_tracks(&self) -> bool {
        if &self.major_brand == b"avis" {
            true
        } else if &self.major_brand == b"avif" {
            false
        } else {
            !self.tracks.is_empty()
        }
    }

    /// Birincil renk ve varsa yardımcı alfa parçasının indekslerini döndürür (`libavif` uyumlu).
    pub(crate) fn select_color_and_alpha_tracks(&self) -> Result<(usize, Option<usize>)> {
        let color_idx = self
            .tracks
            .iter()
            .position(|t| {
                t.track_id != 0
                    && t.is_av01
                    && !t.samples.is_empty()
                    && !t.has_reference_type(b"auxl")
            })
            .ok_or(Error::InvalidData(
                "AVIF görüntü dizisinde geçerli AV1 renk parçası bulunamadı",
            ))?;

        let color_track_id = self.tracks[color_idx].track_id;
        let alpha_idx = self.tracks.iter().position(|t| {
            if t.track_id == 0
                || !t.is_av01
                || t.samples.is_empty()
                || !t.references_track(b"auxl", color_track_id)
            {
                return false;
            }
            // `auxi` özelliği mevcutsa mutlaka standart alfa URN taşımalıdır (derinlik vb. yoksayılır).
            // Eski libavif dosyalarıyla geriye dönük uyumluluk için `auxi` yoksa alfa kabul edilir.
            let auxi = t.properties.iter().find_map(|p| match p {
                ItemProperty::AuxC { aux_type, .. } => Some(aux_type.as_str()),
                _ => None,
            });
            match auxi {
                Some(urn) => urn == ALPHA_AUX_URN_CICP || urn == ALPHA_AUX_URN_HEVC,
                None => true,
            }
        });

        Ok((color_idx, alpha_idx))
    }
}

/// Üst düzey `ftyp`, `moov` ve varsa `meta`/`cmin` kutularını ayrıştırır.
pub(crate) fn parse_movie<'a>(data: &'a [u8], butce: &Arc<Butce>) -> Result<ParsedMovie<'a>> {
    let mut offset = 0usize;
    let mut major_brand: Option<[u8; 4]> = None;
    let mut compatible_brands: Vec<[u8; 4]> = Vec::new();
    let mut ftyp_guard = ButceTutucu::bos(butce);
    let mut has_supported_brand = false;
    let mut moov_payload: Option<&'a [u8]> = None;
    let mut meta_payload: Option<Cow<'a, [u8]>> = None;
    let mut _cmin_guard: Option<ButceTutucu> = None;

    while let Some(raw_box) = read_box_header(data, offset)? {
        offset = raw_box.next_offset;
        match &raw_box.box_type {
            b"ftyp" => {
                if raw_box.payload.len() < 8 {
                    return Err(Error::InvalidData("ftyp kutusu 8 bayttan kısa"));
                }
                let major = [
                    raw_box.payload[0],
                    raw_box.payload[1],
                    raw_box.payload[2],
                    raw_box.payload[3],
                ];
                major_brand = Some(major);
                if is_supported_brand(&major) {
                    has_supported_brand = true;
                }
                let compat_chunks = raw_box.payload[8..].as_chunks::<4>().0;
                let compat_bytes = compat_chunks
                    .len()
                    .saturating_mul(core::mem::size_of::<[u8; 4]>());
                ftyp_guard.ekle(compat_bytes)?;
                compatible_brands
                    .try_reserve_exact(compat_chunks.len())
                    .map_err(|_| Error::AllocationFailed {
                        requested: compat_bytes,
                    })?;
                for &compat in compat_chunks {
                    compatible_brands.push(compat);
                    if is_supported_brand(&compat) {
                        has_supported_brand = true;
                    }
                }
            }
            b"moov" => {
                if moov_payload.is_some() {
                    return Err(Error::InvalidData("birden fazla üst düzey moov kutusu"));
                }
                moov_payload = Some(raw_box.payload);
            }
            b"meta" => {
                if meta_payload.is_some() {
                    return Err(Error::InvalidData("birden fazla üst düzey meta kutusu"));
                }
                meta_payload = Some(Cow::Borrowed(raw_box.payload));
            }
            b"cmin" => {
                if meta_payload.is_some() {
                    return Err(Error::InvalidData("birden fazla üst düzey meta kutusu"));
                }
                let (ver, _flags, body) = read_full_box(raw_box.payload)?;
                if ver != 0 || body.len() < 4 {
                    return Err(Error::InvalidData("geçersiz cmin kutusu"));
                }
                let comp_type = [body[0], body[1], body[2], body[3]];
                let payload_slice = &body[4..];
                let (decompressed, guard) =
                    decompress_by_fourcc(&comp_type, payload_slice, payload_slice.len(), butce)?;
                meta_payload = Some(Cow::Owned(decompressed));
                _cmin_guard = Some(guard);
            }
            _ => {}
        }
    }

    let major_brand = major_brand.ok_or(Error::InvalidData("AVIF dosyasında ftyp kutusu yok"))?;
    if !has_supported_brand {
        return Err(Error::InvalidData(
            "ftyp kutusu desteklenen bir AVIF/AVIS/MIAF markası içermiyor",
        ));
    }

    let top_meta = if let Some(mraw) = meta_payload {
        let mut brands_copy = Vec::new();
        let copy_bytes = compatible_brands
            .len()
            .saturating_mul(core::mem::size_of::<[u8; 4]>());
        let _tmp_guard = Butce::tut(butce, copy_bytes)?;
        brands_copy
            .try_reserve_exact(compatible_brands.len())
            .map_err(|_| Error::AllocationFailed {
                requested: copy_bytes,
            })?;
        brands_copy.extend_from_slice(&compatible_brands);
        drop(_tmp_guard);
        let parsed = parse_meta_box(data, mraw, major_brand, brands_copy, false, butce)?;
        _cmin_guard = None;
        Some(parsed)
    } else {
        None
    };

    let mut tracks = Vec::new();
    let mut tracks_guard = ButceTutucu::bos(butce);
    if let Some(moov) = moov_payload {
        parse_moov(
            data,
            moov,
            major_brand,
            &compatible_brands,
            butce,
            &mut tracks,
            &mut tracks_guard,
        )?;
    }

    ftyp_guard.birlestir(tracks_guard);
    Ok(ParsedMovie {
        major_brand,
        compatible_brands,
        tracks,
        top_meta,
        _guard: ftyp_guard,
    })
}

fn parse_moov<'a>(
    file_data: &'a [u8],
    moov_payload: &'a [u8],
    major_brand: [u8; 4],
    compatible_brands: &[[u8; 4]],
    butce: &Arc<Butce>,
    tracks: &mut Vec<ParsedTrack<'a>>,
    tracks_guard: &mut ButceTutucu,
) -> Result<()> {
    let mut trak_count = 0usize;
    let mut scan_off = 0usize;
    while let Some(child) = read_box_header(moov_payload, scan_off)? {
        scan_off = child.next_offset;
        if &child.box_type == b"trak" {
            trak_count += 1;
        }
    }
    if trak_count > 0 {
        let tracks_bytes = trak_count.saturating_mul(core::mem::size_of::<ParsedTrack<'_>>());
        tracks_guard.ekle(tracks_bytes)?;
        tracks.try_reserve_exact(trak_count).map_err(|_| {
            tracks_guard.dusur(tracks_bytes);
            Error::AllocationFailed {
                requested: tracks_bytes,
            }
        })?;
    }

    let mut offset = 0usize;
    let mut seen_mvhd = false;
    let mut movie_timescale = 0u64;
    while let Some(child) = read_box_header(moov_payload, offset)? {
        offset = child.next_offset;
        match &child.box_type {
            b"mvhd" => {
                if seen_mvhd {
                    return Err(Error::InvalidData("moov içinde birden fazla mvhd kutusu"));
                }
                seen_mvhd = true;
                movie_timescale = parse_mvhd(child.payload)?;
            }
            b"trak" => {
                let track = parse_trak(
                    file_data,
                    child.payload,
                    major_brand,
                    compatible_brands,
                    movie_timescale,
                    butce,
                )?;
                if matches!(&track.handler_type, b"pict" | b"vide" | b"auxv")
                    && (track.width == 0 || track.height == 0)
                {
                    return Err(Error::InvalidData("görüntü parçası boyutu sıfır olamaz"));
                }
                tracks.push(track);
            }
            _ => {}
        }
    }
    if movie_timescale != 0 {
        for track in tracks.iter_mut() {
            track.movie_timescale = movie_timescale;
        }
    }
    Ok(())
}

fn parse_mvhd(payload: &[u8]) -> Result<u64> {
    let (ver, _flags, body) = read_full_box(payload)?;
    let (min_len, ts_off) = match ver {
        0 => (96usize, 8usize),
        1 => (108usize, 16usize),
        _ => return Err(Error::Unsupported("desteklenmeyen mvhd sürümü")),
    };
    if body.len() < min_len {
        return Err(Error::InvalidData("kesik mvhd kutusu"));
    }
    let timescale =
        u64::from(read_u32_be(body, ts_off).ok_or(Error::InvalidData("kesik mvhd timescale"))?);
    if timescale == 0 {
        return Err(Error::InvalidData("mvhd timescale sıfır olamaz"));
    }
    Ok(timescale)
}

fn parse_trak<'a>(
    file_data: &'a [u8],
    trak_payload: &'a [u8],
    major_brand: [u8; 4],
    compatible_brands: &[[u8; 4]],
    movie_timescale: u64,
    butce: &Arc<Butce>,
) -> Result<ParsedTrack<'a>> {
    let mut offset = 0usize;
    let mut tkhd_seen = false;
    let mut edts_seen = false;
    let mut track_id = 0u32;
    let mut track_duration = 0u64;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut tkhd_matrix = TKHD_IDENTITY_MATRIX;
    let mut handler_type = [0u8; 4];
    let mut media_timescale = 0u64;
    let mut media_duration = 0u64;
    let mut edit_list: Option<EditList> = None;
    let mut references = Vec::new();
    let mut meta_guard = ButceTutucu::bos(butce);
    let mut raw_stbl: Option<(RawSampleTable, ButceTutucu, ButceTutucu)> = None;
    let mut track_meta: Option<AvifContainer<'a>> = None;

    while let Some(child) = read_box_header(trak_payload, offset)? {
        offset = child.next_offset;
        match &child.box_type {
            b"tkhd" => {
                if tkhd_seen {
                    return Err(Error::InvalidData("trak içinde birden fazla tkhd kutusu"));
                }
                tkhd_seen = true;
                let tkhd = parse_tkhd(child.payload)?;
                track_id = tkhd.track_id;
                track_duration = tkhd.duration;
                width = tkhd.width;
                height = tkhd.height;
                tkhd_matrix = tkhd.matrix;
            }
            b"edts" => {
                if edts_seen {
                    return Err(Error::InvalidData("trak içinde birden fazla edts kutusu"));
                }
                edts_seen = true;
                edit_list = Some(parse_edts(child.payload, &mut meta_guard)?);
            }
            b"tref" => {
                parse_tref(child.payload, &mut references, &mut meta_guard)?;
            }
            b"meta" => {
                let mut brands_copy = Vec::new();
                let copy_bytes = compatible_brands
                    .len()
                    .saturating_mul(core::mem::size_of::<[u8; 4]>());
                let _tmp_guard = Butce::tut(butce, copy_bytes)?;
                brands_copy
                    .try_reserve_exact(compatible_brands.len())
                    .map_err(|_| Error::AllocationFailed {
                        requested: copy_bytes,
                    })?;
                brands_copy.extend_from_slice(compatible_brands);
                drop(_tmp_guard);
                track_meta = Some(parse_meta_box(
                    file_data,
                    Cow::Borrowed(child.payload),
                    major_brand,
                    brands_copy,
                    false,
                    butce,
                )?);
            }
            b"mdia" => {
                parse_mdia(
                    child.payload,
                    &mut handler_type,
                    &mut media_timescale,
                    &mut media_duration,
                    &mut raw_stbl,
                    butce,
                )?;
            }
            _ => {}
        }
    }

    if !tkhd_seen {
        return Err(Error::InvalidData("trak kutusunda zorunlu tkhd eksik"));
    }

    let repetition_count = match &edit_list {
        Some(el) if (el.flags & 1) != 0 => {
            if el.entries.len() != 1 {
                return Err(Error::InvalidData("tekrarlayan elst entry_count != 1"));
            }
            let segment_duration = el.entries[0].segment_duration;
            if segment_duration == 0 {
                return Err(Error::InvalidData("elst segment_duration sıfır olamaz"));
            }
            if track_duration == INDEFINITE_DURATION_64 {
                LoopCount::Infinite
            } else {
                if track_duration == 0 {
                    return Err(Error::InvalidData(
                        "tekrarlayan edts için geçersiz track veya segment süresi (0)",
                    ));
                }
                let reps = track_duration.div_ceil(segment_duration).saturating_sub(1);
                if reps > u64::from(i32::MAX as u32) {
                    LoopCount::Infinite
                } else {
                    LoopCount::Count(reps as u32)
                }
            }
        }
        _ => LoopCount::Count(0),
    };

    let (is_av01, properties, samples, mut track_guard) =
        if let Some((mut stbl, mut prop_guard, table_guard)) = raw_stbl {
            let is_av01 = stbl.is_av01;
            let (samples, sample_guard) = if is_av01 {
                build_track_samples(&stbl, file_data.len(), butce)?
            } else {
                (Vec::new(), ButceTutucu::bos(butce))
            };
            let properties = core::mem::take(&mut stbl.properties);
            drop(stbl);
            drop(table_guard);
            prop_guard.birlestir(sample_guard);
            (is_av01, properties, samples, prop_guard)
        } else {
            (false, Vec::new(), Vec::new(), ButceTutucu::bos(butce))
        };
    track_guard.birlestir(meta_guard);

    Ok(ParsedTrack {
        track_id,
        width,
        height,
        tkhd_matrix,
        handler_type,
        media_timescale,
        media_duration,
        movie_timescale,
        edit_list,
        repetition_count,
        is_av01,
        properties,
        references,
        samples,
        meta: track_meta,
        _guard: track_guard.into_arc()?,
    })
}

fn parse_tkhd_matrix(body: &[u8], matrix_off: usize) -> Result<[[i8; 2]; 2]> {
    const P: i32 = 0x0001_0000;
    const N: i32 = -0x0001_0000;
    const W: i32 = 0x4000_0000;

    let mut m = [0i32; 9];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = read_i32_be(body, matrix_off + i * 4)
            .ok_or(Error::InvalidData("kesik tkhd dönüşüm matrisi"))?;
    }
    let [a, b, u, c, d, v, _x, _y, w] = m;
    if u != 0 || v != 0 || w != W {
        return Err(Error::Unsupported("desteklenmeyen tkhd dönüşüm matrisi"));
    }
    match [a, b, c, d] {
        [P, 0, 0, P] => Ok([[1, 0], [0, 1]]),
        [0, N, P, 0] => Ok([[0, -1], [1, 0]]),
        [N, 0, 0, N] => Ok([[-1, 0], [0, -1]]),
        [0, P, N, 0] => Ok([[0, 1], [-1, 0]]),
        [P, 0, 0, N] => Ok([[1, 0], [0, -1]]),
        [N, 0, 0, P] => Ok([[-1, 0], [0, 1]]),
        [0, P, P, 0] => Ok([[0, 1], [1, 0]]),
        [0, N, N, 0] => Ok([[0, -1], [-1, 0]]),
        _ => Err(Error::Unsupported("desteklenmeyen tkhd dönüşüm matrisi")),
    }
}

struct ParsedTkhd {
    track_id: u32,
    duration: u64,
    width: u32,
    height: u32,
    matrix: [[i8; 2]; 2],
}

fn parse_tkhd(payload: &[u8]) -> Result<ParsedTkhd> {
    let (ver, _flags, body) = read_full_box(payload)?;
    let (track_id, duration, after_dur) = match ver {
        0 => {
            let tid = read_u32_be(body, 8).ok_or(Error::InvalidData("kesik tkhd v0 track_id"))?;
            let dur32 =
                read_u32_be(body, 16).ok_or(Error::InvalidData("kesik tkhd v0 duration"))?;
            let dur = if dur32 == INDEFINITE_DURATION_32 {
                INDEFINITE_DURATION_64
            } else {
                u64::from(dur32)
            };
            (tid, dur, 20usize)
        }
        1 => {
            let tid = read_u32_be(body, 16).ok_or(Error::InvalidData("kesik tkhd v1 track_id"))?;
            let dur = read_u64_be(body, 24).ok_or(Error::InvalidData("kesik tkhd v1 duration"))?;
            (tid, dur, 32usize)
        }
        _ => return Err(Error::Unsupported("desteklenmeyen tkhd sürümü")),
    };
    let matrix = parse_tkhd_matrix(body, after_dur + 16)?;
    let dims_off = after_dur + 52;
    let w_fixed = read_u32_be(body, dims_off).ok_or(Error::InvalidData("kesik tkhd width"))?;
    let h_fixed = read_u32_be(body, dims_off + 4).ok_or(Error::InvalidData("kesik tkhd height"))?;
    Ok(ParsedTkhd {
        track_id,
        duration,
        width: w_fixed >> 16,
        height: h_fixed >> 16,
        matrix,
    })
}

fn parse_edts(payload: &[u8], guard: &mut ButceTutucu) -> Result<EditList> {
    let mut offset = 0usize;
    let mut seen_elst = false;
    let mut result = EditList::default();
    while let Some(child) = read_box_header(payload, offset)? {
        offset = child.next_offset;
        if &child.box_type == b"elst" {
            if seen_elst {
                return Err(Error::InvalidData("edts içinde birden fazla elst kutusu"));
            }
            seen_elst = true;
            result = parse_elst(child.payload, guard)?;
        }
    }
    if !seen_elst {
        return Err(Error::InvalidData("edts kutusu elst alt kutusu içermiyor"));
    }
    Ok(result)
}

fn parse_elst(payload: &[u8], guard: &mut ButceTutucu) -> Result<EditList> {
    let (ver, flags, body) = read_full_box(payload)?;
    if ver > 1 {
        return Err(Error::Unsupported("desteklenmeyen elst sürümü"));
    }
    let entry_count =
        read_u32_be(body, 0).ok_or(Error::InvalidData("kesik elst entry_count"))? as usize;
    let wire_stride = if ver == 0 { 12usize } else { 20usize };
    let needed = entry_count
        .checked_mul(wire_stride)
        .ok_or(Error::InvalidData("elst boyut taşması"))?;
    if needed > body.len().saturating_sub(4) {
        return Err(Error::InvalidData("kesik elst giriş dizisi"));
    }
    if (flags & 1) != 0 && entry_count != 1 {
        return Err(Error::InvalidData("tekrarlayan elst entry_count != 1"));
    }
    reserve_table_budget(entry_count, core::mem::size_of::<EditListEntry>(), guard)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entry_count)
        .map_err(|_| Error::AllocationFailed {
            requested: entry_count.saturating_mul(core::mem::size_of::<EditListEntry>()),
        })?;
    let mut pos = 4usize;
    for _ in 0..entry_count {
        let (segment_duration, media_time) = if ver == 0 {
            let sd = u64::from(
                read_u32_be(body, pos)
                    .ok_or(Error::InvalidData("kesik elst v0 segment_duration"))?,
            );
            let mt = i64::from(
                read_i32_be(body, pos + 4).ok_or(Error::InvalidData("kesik elst v0 media_time"))?,
            );
            pos += 8;
            (sd, mt)
        } else {
            let sd = read_u64_be(body, pos)
                .ok_or(Error::InvalidData("kesik elst v1 segment_duration"))?;
            let mt =
                read_i64_be(body, pos + 8).ok_or(Error::InvalidData("kesik elst v1 media_time"))?;
            pos += 16;
            (sd, mt)
        };
        let media_rate_integer =
            read_i16_be(body, pos).ok_or(Error::InvalidData("kesik elst media_rate_integer"))?;
        let media_rate_fraction = read_i16_be(body, pos + 2)
            .ok_or(Error::InvalidData("kesik elst media_rate_fraction"))?;
        pos += 4;

        if media_time < -1 {
            return Err(Error::InvalidData("elst media_time -1 değerinden küçük"));
        }
        if media_time >= 0 && (media_rate_integer != 1 || media_rate_fraction != 0) {
            return Err(Error::Unsupported("elst media_rate != 1.0 desteklenmiyor"));
        }
        if (flags & 1) != 0 && segment_duration == 0 {
            return Err(Error::InvalidData("elst segment_duration sıfır olamaz"));
        }
        entries.push(EditListEntry {
            segment_duration,
            media_time,
            media_rate_integer,
            media_rate_fraction,
        });
    }
    Ok(EditList { flags, entries })
}

const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

const fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Parçanın `elst` düzenleme listesini uygulayarak sunum zaman çizelgesini (`PresentedSample`),
/// toplam sunum süresini ve etkin zaman ölçeğini (`lcm(media_timescale, movie_timescale)`)
/// `O((S + E + K) log S)` karmaşıklıkta üretir.
pub(crate) fn build_presented_timeline(
    track: &ParsedTrack<'_>,
    butce: &Arc<Butce>,
    guard: &mut ButceTutucu,
) -> Result<(Vec<PresentedSample>, u64, u64)> {
    let media_ts = track.media_timescale.max(1);
    let identity_timeline = |guard: &mut ButceTutucu| -> Result<(Vec<PresentedSample>, u64, u64)> {
        let n = track.samples.len();
        reserve_table_budget(n, core::mem::size_of::<PresentedSample>(), guard)?;
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| Error::AllocationFailed {
                requested: n.saturating_mul(core::mem::size_of::<PresentedSample>()),
            })?;
        for (raw_index, s) in track.samples.iter().enumerate() {
            out.push(PresentedSample {
                raw_index,
                pts_in_timescale: s.pts_in_timescale,
                duration_in_timescale: s.duration_in_timescale,
            });
        }
        Ok((out, track.media_duration, track.media_timescale))
    };

    let Some(ref edit_list) = track.edit_list else {
        return identity_timeline(guard);
    };
    if edit_list.entries.is_empty() {
        return identity_timeline(guard);
    }

    let movie_ts = if track.movie_timescale > 0 {
        track.movie_timescale
    } else {
        media_ts
    };
    let g = gcd_u64(media_ts, movie_ts).max(1);
    let m_mul = u128::from(movie_ts / g);
    let v_mul = u128::from(media_ts / g);
    let mut eff_ts_u128 = u128::from(media_ts)
        .checked_mul(m_mul)
        .ok_or(Error::InvalidData("elst ortak zaman ölçeği taşması"))?;

    // Tek girdili, `media_time == 0` düzenleme listesi:
    // - Tekrarlayan (`flags & 1 != 0`) AVIS döngüsünde medya kırpma yapmıyorsa (`seg_eff >= total_media_end`),
    // - Tekrarlamayan (`flags & 1 == 0`) durumda da tam medya süresine birebir eşitse (`seg_eff == total_media_end`)
    //   ve `track.media_duration` ile örtüşüyorsa ham `mdhd.timescale` zaman çizelgesini doğrudan koru.
    if edit_list.entries.len() == 1 {
        let e = edit_list.entries[0];
        if e.media_time == 0 {
            let total_media_end = track
                .samples
                .iter()
                .map(|s| {
                    u128::from(s.pts_in_timescale)
                        .saturating_add(u128::from(s.duration_in_timescale))
                        .saturating_mul(m_mul)
                })
                .max()
                .unwrap_or(0);
            let seg_eff = if e.segment_duration == 0 {
                None
            } else {
                Some(
                    u128::from(e.segment_duration)
                        .checked_mul(v_mul)
                        .ok_or(Error::InvalidData("elst segment_duration taşması"))?,
                )
            };
            let is_repeating = (edit_list.flags & 1) != 0;
            let is_identity = if is_repeating {
                seg_eff.is_none_or(|d| d >= total_media_end)
            } else {
                seg_eff.is_none_or(|d| d == total_media_end)
                    && u128::from(track.media_duration).saturating_mul(m_mul) == total_media_end
            };
            if is_identity {
                return identity_timeline(guard);
            }
        }
    }

    let n = track.samples.len();
    if n == 0 {
        return Err(Error::InvalidData(
            "elst zaman çizelgesi hiç görünür kare üretmedi",
        ));
    }

    #[derive(Clone, Copy)]
    struct SampleInterval {
        start: u128,
        end: u128,
        raw_index: usize,
    }

    #[derive(Clone, Copy)]
    struct RawPresented {
        raw_index: usize,
        pts: u128,
        dur: u128,
    }

    // Geçici sıralı aralık dizisi + düz maksimum-bitiş aralık ağacı (`O(S)` bellek, `Butce` altında).
    let tree_size = n.next_power_of_two();
    let intervals_bytes = n
        .checked_mul(core::mem::size_of::<SampleInterval>())
        .ok_or(Error::InvalidData("elst aralık dizisi boyut taşması"))?;
    let tree_nodes = tree_size
        .checked_mul(2)
        .ok_or(Error::InvalidData("elst aralık ağacı boyut taşması"))?;
    let tree_bytes = tree_nodes
        .checked_mul(core::mem::size_of::<u128>())
        .ok_or(Error::InvalidData("elst aralık ağacı bayt taşması"))?;
    let stack_cap = 128usize;
    let stack_bytes = stack_cap * core::mem::size_of::<(usize, usize, usize)>();
    let temp_total = intervals_bytes
        .checked_add(tree_bytes)
        .and_then(|v| v.checked_add(stack_bytes))
        .ok_or(Error::InvalidData("elst geçici bellek taşması"))?;
    let _temp_guard = Butce::tut(butce, temp_total)?;

    let mut intervals: Vec<SampleInterval> = Vec::new();
    intervals
        .try_reserve_exact(n)
        .map_err(|_| Error::AllocationFailed {
            requested: intervals_bytes,
        })?;
    for (raw_index, sample) in track.samples.iter().enumerate() {
        let start = u128::from(sample.pts_in_timescale)
            .checked_mul(m_mul)
            .ok_or(Error::InvalidData("elst örnek zaman damgası taşması"))?;
        let dur = u128::from(sample.duration_in_timescale)
            .checked_mul(m_mul)
            .ok_or(Error::InvalidData("elst örnek süresi taşması"))?;
        let end = start.saturating_add(dur);
        intervals.push(SampleInterval {
            start,
            end,
            raw_index,
        });
    }
    intervals.sort_unstable_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| a.raw_index.cmp(&b.raw_index))
    });

    let mut tree_max_end: Vec<u128> = Vec::new();
    tree_max_end
        .try_reserve_exact(tree_nodes)
        .map_err(|_| Error::AllocationFailed {
            requested: tree_bytes,
        })?;
    tree_max_end.resize(tree_nodes, 0u128);
    for (i, iv) in intervals.iter().enumerate() {
        tree_max_end[tree_size + i] = iv.end;
    }
    for node in (1..tree_size).rev() {
        tree_max_end[node] = tree_max_end[node * 2].max(tree_max_end[node * 2 + 1]);
    }

    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    stack
        .try_reserve_exact(stack_cap)
        .map_err(|_| Error::AllocationFailed {
            requested: stack_bytes,
        })?;

    let mut raw_presented: Vec<RawPresented> = Vec::new();
    let mut raw_pres_guard = ButceTutucu::bos(butce);
    let mut timeline_pts_u128: u128 = 0;

    let push_sample = |raw_presented: &mut Vec<RawPresented>,
                       raw_pres_guard: &mut ButceTutucu,
                       raw_index: usize,
                       pres_pts_u128: u128,
                       clipped_dur_u128: u128,
                       seg_max_end: &mut u128|
     -> Result<()> {
        if raw_presented.len() >= MAX_TRACK_SAMPLES {
            return Err(Error::BudgetExceeded {
                requested: raw_presented.len() + 1,
                limit: MAX_TRACK_SAMPLES,
            });
        }
        if raw_presented.len() == raw_presented.capacity() {
            let old_cap = raw_presented.capacity();
            let new_cap = if old_cap == 0 {
                n.clamp(1, 64)
            } else {
                old_cap.saturating_mul(2).min(MAX_TRACK_SAMPLES)
            };
            let add_cap = new_cap - old_cap;
            let add_bytes = add_cap.saturating_mul(core::mem::size_of::<RawPresented>());
            raw_pres_guard.ekle(add_bytes)?;
            raw_presented.try_reserve_exact(add_cap).map_err(|_| {
                raw_pres_guard.dusur(add_bytes);
                Error::AllocationFailed {
                    requested: add_bytes,
                }
            })?;
        }
        raw_presented.push(RawPresented {
            raw_index,
            pts: pres_pts_u128,
            dur: clipped_dur_u128,
        });
        *seg_max_end = (*seg_max_end).max(pres_pts_u128.saturating_add(clipped_dur_u128));
        Ok(())
    };

    for entry in &edit_list.entries {
        let seg_dur_eff = if entry.segment_duration == 0 {
            None
        } else {
            Some(
                u128::from(entry.segment_duration)
                    .checked_mul(v_mul)
                    .ok_or(Error::InvalidData("elst segment_duration taşması"))?,
            )
        };
        if entry.media_time == -1 {
            timeline_pts_u128 = timeline_pts_u128.saturating_add(seg_dur_eff.unwrap_or(0));
            continue;
        }
        let edit_start = u128::from(entry.media_time as u64)
            .checked_mul(m_mul)
            .ok_or(Error::InvalidData("elst media_time taşması"))?;
        let edit_end = seg_dur_eff.map(|d| edit_start.saturating_add(d));
        let mut seg_max_end = timeline_pts_u128;
        let seg_start_idx = raw_presented.len();

        // 1. Pozitif kesişimli örnekleri (`s_start < edit_end && s_end > edit_start`)
        //    düz maksimum-bitiş aralık ağacında `O(log S + k)` sürede bul.
        if tree_max_end[1] > edit_start && edit_end.is_none_or(|ee| intervals[0].start < ee) {
            stack.clear();
            stack.push((1usize, 0usize, tree_size - 1));
            while let Some((node, l, r)) = stack.pop() {
                if l >= n || tree_max_end[node] <= edit_start {
                    continue;
                }
                if edit_end.is_some_and(|ee| intervals[l].start >= ee) {
                    continue;
                }
                if l == r {
                    let iv = intervals[l];
                    let clip_start = iv.start.max(edit_start);
                    let clip_end = match edit_end {
                        Some(ee) => iv.end.min(ee),
                        None => iv.end,
                    };
                    if clip_end > clip_start {
                        let clipped_dur = clip_end - clip_start;
                        let pres_pts = timeline_pts_u128.saturating_add(clip_start - edit_start);
                        push_sample(
                            &mut raw_presented,
                            &mut raw_pres_guard,
                            iv.raw_index,
                            pres_pts,
                            clipped_dur,
                            &mut seg_max_end,
                        )?;
                    }
                    continue;
                }
                let mid = l + (r - l) / 2;
                let left = node * 2;
                let right = left + 1;
                if mid + 1 < n
                    && tree_max_end[right] > edit_start
                    && edit_end.is_none_or(|ee| intervals[mid + 1].start < ee)
                {
                    stack.push((right, mid + 1, r));
                }
                if tree_max_end[left] > edit_start
                    && edit_end.is_none_or(|ee| intervals[l].start < ee)
                {
                    stack.push((left, l, mid));
                }
            }
        }

        // 2. Sıfır süreli sınır durumu (`duration == 0 && s_start == edit_start && s_start < edit_end`):
        //    sıralı aralık dizisinde `O(log S)` ikili arama ile yakala.
        if edit_end.is_none_or(|ee| edit_start < ee) {
            let mut idx = intervals.partition_point(|iv| iv.start < edit_start);
            while idx < n && intervals[idx].start == edit_start {
                let iv = intervals[idx];
                if iv.end == iv.start {
                    push_sample(
                        &mut raw_presented,
                        &mut raw_pres_guard,
                        iv.raw_index,
                        timeline_pts_u128,
                        0,
                        &mut seg_max_end,
                    )?;
                }
                idx += 1;
            }
        }

        if raw_presented.len() > seg_start_idx + 1 {
            raw_presented[seg_start_idx..].sort_unstable_by(|a, b| {
                a.pts
                    .cmp(&b.pts)
                    .then_with(|| a.raw_index.cmp(&b.raw_index))
            });
        }

        timeline_pts_u128 = match seg_dur_eff {
            Some(d) => timeline_pts_u128.saturating_add(d),
            None => seg_max_end,
        };
    }

    drop((intervals, tree_max_end, stack, _temp_guard));

    if raw_presented.is_empty() {
        return Err(Error::InvalidData(
            "elst zaman çizelgesi hiç görünür kare üretmedi",
        ));
    }

    // `u128` uzayında zaman ölçeği sadeleştirmesi:
    // 1. Tüm sunum zaman damgaları, süreler ve toplam süre `m_mul`'a tam bölünüyorsa
    //    doğrudan `media_ts` ölçeğine indirge (`u64::try_from` dönüşümünden ÖNCE).
    // 2. Tam bölünmüyorsa ama herhangi bir değer `u64::MAX` sınırını aşıyorsa ortak bölen (`gcd`)
    //    ile sadeleştir (`u64` içine zaten sığan alt-birim zaman çizelgelerinin ölçeğine dokunulmaz).
    if m_mul > 1
        && timeline_pts_u128.is_multiple_of(m_mul)
        && raw_presented
            .iter()
            .all(|p| p.pts.is_multiple_of(m_mul) && p.dur.is_multiple_of(m_mul))
    {
        for p in &mut raw_presented {
            p.pts /= m_mul;
            p.dur /= m_mul;
        }
        timeline_pts_u128 /= m_mul;
        eff_ts_u128 = u128::from(media_ts);
    } else if eff_ts_u128 > u128::from(u64::MAX)
        || timeline_pts_u128 > u128::from(u64::MAX)
        || raw_presented
            .iter()
            .any(|p| p.pts > u128::from(u64::MAX) || p.dur > u128::from(u64::MAX))
    {
        let mut g_red = gcd_u128(m_mul, timeline_pts_u128);
        for p in &raw_presented {
            g_red = gcd_u128(g_red, gcd_u128(p.pts, p.dur));
            if g_red <= 1 {
                break;
            }
        }
        if g_red > 1 {
            for p in &mut raw_presented {
                p.pts /= g_red;
                p.dur /= g_red;
            }
            timeline_pts_u128 /= g_red;
            eff_ts_u128 /= g_red;
        }
    }

    let eff_ts = u64::try_from(eff_ts_u128)
        .map_err(|_| Error::InvalidData("elst ortak zaman ölçeği u64 taşması"))?;
    let total_duration = u64::try_from(timeline_pts_u128)
        .map_err(|_| Error::InvalidData("elst toplam süre u64 taşması"))?;

    let pres_len = raw_presented.len();
    let pres_bytes = pres_len.saturating_mul(core::mem::size_of::<PresentedSample>());
    guard.ekle(pres_bytes)?;
    let mut presented = Vec::new();
    presented
        .try_reserve_exact(pres_len)
        .map_err(|_| Error::AllocationFailed {
            requested: pres_bytes,
        })?;
    for p in &raw_presented {
        let pres_pts = u64::try_from(p.pts)
            .map_err(|_| Error::InvalidData("elst sunum zaman damgası u64 taşması"))?;
        let clipped_dur = u64::try_from(p.dur)
            .map_err(|_| Error::InvalidData("elst kırpılmış süre u64 taşması"))?;
        presented.push(PresentedSample {
            raw_index: p.raw_index,
            pts_in_timescale: pres_pts,
            duration_in_timescale: clipped_dur,
        });
    }
    drop((raw_presented, raw_pres_guard));

    Ok((presented, total_duration, eff_ts))
}

fn parse_tref(
    payload: &[u8],
    refs: &mut Vec<TrackReference>,
    guard: &mut ButceTutucu,
) -> Result<()> {
    let mut total_valid = 0usize;
    let mut scan_off = 0usize;
    while let Some(child) = read_box_header(payload, scan_off)? {
        scan_off = child.next_offset;
        if matches!(&child.box_type, b"auxl" | b"prem") {
            if child.payload.len() < 4 {
                return Err(Error::InvalidData("tref referans kutusu 4 bayttan kısa"));
            }
            for chunk in child.payload.as_chunks::<4>().0 {
                if u32::from_be_bytes(*chunk) != 0 {
                    total_valid += 1;
                }
            }
        }
    }
    if total_valid > 0 {
        let bytes = total_valid.saturating_mul(core::mem::size_of::<TrackReference>());
        guard.ekle(bytes)?;
        refs.try_reserve_exact(total_valid).map_err(|_| {
            guard.dusur(bytes);
            Error::AllocationFailed { requested: bytes }
        })?;
    }

    let mut offset = 0usize;
    while let Some(child) = read_box_header(payload, offset)? {
        offset = child.next_offset;
        if matches!(&child.box_type, b"auxl" | b"prem") {
            for chunk in child.payload.as_chunks::<4>().0 {
                let to_id = u32::from_be_bytes(*chunk);
                if to_id != 0 {
                    refs.push(TrackReference {
                        ref_type: child.box_type,
                        to_track_id: to_id,
                    });
                }
            }
        }
    }
    Ok(())
}

fn parse_mdia(
    payload: &[u8],
    handler_type: &mut [u8; 4],
    media_timescale: &mut u64,
    media_duration: &mut u64,
    raw_stbl: &mut Option<(RawSampleTable, ButceTutucu, ButceTutucu)>,
    butce: &Arc<Butce>,
) -> Result<()> {
    let mut offset = 0usize;
    while let Some(child) = read_box_header(payload, offset)? {
        offset = child.next_offset;
        match &child.box_type {
            b"mdhd" => {
                let (ts, dur) = parse_mdhd(child.payload)?;
                *media_timescale = ts;
                *media_duration = dur;
            }
            b"hdlr" => {
                let (ver, _flags, body) = read_full_box(child.payload)?;
                if ver != 0 || body.len() < 8 {
                    return Err(Error::InvalidData("geçersiz mdia hdlr kutusu"));
                }
                *handler_type = [body[4], body[5], body[6], body[7]];
            }
            b"minf" => {
                parse_minf(child.payload, raw_stbl, butce)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_mdhd(payload: &[u8]) -> Result<(u64, u64)> {
    let (ver, _flags, body) = read_full_box(payload)?;
    match ver {
        0 => {
            let ts = read_u32_be(body, 8).ok_or(Error::InvalidData("kesik mdhd v0 timescale"))?;
            let dur = read_u32_be(body, 12).ok_or(Error::InvalidData("kesik mdhd v0 duration"))?;
            Ok((u64::from(ts), u64::from(dur)))
        }
        1 => {
            let ts = read_u32_be(body, 16).ok_or(Error::InvalidData("kesik mdhd v1 timescale"))?;
            let dur = read_u64_be(body, 20).ok_or(Error::InvalidData("kesik mdhd v1 duration"))?;
            Ok((u64::from(ts), dur))
        }
        _ => Err(Error::Unsupported("desteklenmeyen mdhd sürümü")),
    }
}

fn parse_minf(
    payload: &[u8],
    raw_stbl: &mut Option<(RawSampleTable, ButceTutucu, ButceTutucu)>,
    butce: &Arc<Butce>,
) -> Result<()> {
    let mut offset = 0usize;
    while let Some(child) = read_box_header(payload, offset)? {
        offset = child.next_offset;
        if &child.box_type == b"stbl" {
            if raw_stbl.is_some() {
                return Err(Error::InvalidData("minf içinde birden fazla stbl kutusu"));
            }
            *raw_stbl = Some(parse_stbl(child.payload, butce)?);
        }
    }
    Ok(())
}

fn reserve_table_budget(entries: usize, entry_bytes: usize, guard: &mut ButceTutucu) -> Result<()> {
    if entries > MAX_TRACK_SAMPLES {
        return Err(Error::BudgetExceeded {
            requested: entries,
            limit: MAX_TRACK_SAMPLES,
        });
    }
    let bytes = entries.saturating_mul(entry_bytes);
    guard.ekle(bytes)
}

fn parse_stbl(
    payload: &[u8],
    butce: &Arc<Butce>,
) -> Result<(RawSampleTable, ButceTutucu, ButceTutucu)> {
    let mut stbl = RawSampleTable::default();
    let mut prop_guard = ButceTutucu::bos(butce);
    let mut table_guard = ButceTutucu::bos(butce);
    let mut offset = 0usize;
    while let Some(child) = read_box_header(payload, offset)? {
        offset = child.next_offset;
        match &child.box_type {
            b"stsd" => parse_stsd(child.payload, &mut stbl, &mut prop_guard)?,
            b"stco" => parse_chunk_offsets(child.payload, false, &mut stbl, &mut table_guard)?,
            b"co64" => parse_chunk_offsets(child.payload, true, &mut stbl, &mut table_guard)?,
            b"stsc" => parse_stsc(child.payload, &mut stbl, &mut table_guard)?,
            b"stsz" => parse_stsz(child.payload, &mut stbl, &mut table_guard)?,
            b"stts" => parse_stts(child.payload, &mut stbl, &mut table_guard)?,
            b"ctts" => parse_ctts(child.payload, &mut stbl, &mut table_guard)?,
            b"stss" => parse_stss(child.payload, &mut stbl, &mut table_guard)?,
            _ => {}
        }
    }
    Ok((stbl, prop_guard, table_guard))
}

fn parse_stsd(
    payload: &[u8],
    stbl: &mut RawSampleTable,
    prop_guard: &mut ButceTutucu,
) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver > 1 {
        return Err(Error::InvalidData("stsd sürümü 0 veya 1 olmalıdır"));
    }
    let entry_count = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik stsd entry_count"))?;
    let mut pos = 4usize;
    for _ in 0..entry_count {
        let Some(entry) = read_box_header(body, pos)? else {
            return Err(Error::InvalidData("kesik stsd örnek girişi"));
        };
        pos = entry.next_offset;
        if &entry.box_type == b"av01" {
            if entry.payload.len() < VISUAL_SAMPLE_ENTRY_SIZE {
                return Err(Error::InvalidData("kesik av01 VisualSampleEntry başlığı"));
            }
            stbl.is_av01 = true;
            parse_property_boxes(
                &entry.payload[VISUAL_SAMPLE_ENTRY_SIZE..],
                &mut stbl.properties,
                true,
                prop_guard,
            )?;
        }
    }
    Ok(())
}

fn parse_chunk_offsets(
    payload: &[u8],
    large: bool,
    stbl: &mut RawSampleTable,
    guard: &mut ButceTutucu,
) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver != 0 {
        return Err(Error::InvalidData("stco/co64 sürümü 0 olmalıdır"));
    }
    let count =
        read_u32_be(body, 0).ok_or(Error::InvalidData("kesik stco/co64 entry_count"))? as usize;
    let elem_size = if large { 8usize } else { 4usize };
    let needed = count
        .checked_mul(elem_size)
        .ok_or(Error::InvalidData("stco/co64 boyut taşması"))?;
    if needed > body.len().saturating_sub(4) {
        return Err(Error::InvalidData("kesik stco/co64 ofset dizisi"));
    }
    reserve_table_budget(count, core::mem::size_of::<u64>(), guard)?;
    stbl.chunks
        .try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: count.saturating_mul(core::mem::size_of::<u64>()),
        })?;
    for i in 0..count {
        let off = if large {
            read_u64_be(body, 4 + i * 8).ok_or(Error::InvalidData("kesik co64 ofseti"))?
        } else {
            u64::from(read_u32_be(body, 4 + i * 4).ok_or(Error::InvalidData("kesik stco ofseti"))?)
        };
        stbl.chunks.push(off);
    }
    Ok(())
}

fn parse_stsc(payload: &[u8], stbl: &mut RawSampleTable, guard: &mut ButceTutucu) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver != 0 {
        return Err(Error::InvalidData("stsc sürümü 0 olmalıdır"));
    }
    let count = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik stsc entry_count"))? as usize;
    let needed = count
        .checked_mul(12)
        .ok_or(Error::InvalidData("stsc boyut taşması"))?;
    if needed > body.len().saturating_sub(4) {
        return Err(Error::InvalidData("kesik stsc giriş dizisi"));
    }
    reserve_table_budget(count, core::mem::size_of::<SampleToChunk>(), guard)?;
    stbl.sample_to_chunks
        .try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: count.saturating_mul(core::mem::size_of::<SampleToChunk>()),
        })?;
    let mut prev_first = 0u32;
    for i in 0..count {
        let base = 4 + i * 12;
        let first_chunk = read_u32_be(body, base).ok_or(Error::InvalidData("kesik stsc"))?;
        let samples_per_chunk =
            read_u32_be(body, base + 4).ok_or(Error::InvalidData("kesik stsc"))?;
        let _desc_idx = read_u32_be(body, base + 8).ok_or(Error::InvalidData("kesik stsc"))?;
        if i == 0 {
            if first_chunk != 1 {
                return Err(Error::InvalidData("stsc first_chunk 1 ile başlamalıdır"));
            }
        } else if first_chunk <= prev_first {
            return Err(Error::InvalidData(
                "stsc first_chunk kesin artan sırada olmalıdır",
            ));
        }
        prev_first = first_chunk;
        stbl.sample_to_chunks.push(SampleToChunk {
            first_chunk,
            samples_per_chunk,
        });
    }
    Ok(())
}

fn parse_stsz(payload: &[u8], stbl: &mut RawSampleTable, guard: &mut ButceTutucu) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver != 0 {
        return Err(Error::InvalidData("stsz sürümü 0 olmalıdır"));
    }
    let sample_size = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik stsz sample_size"))?;
    let sample_count =
        read_u32_be(body, 4).ok_or(Error::InvalidData("kesik stsz sample_count"))? as usize;
    if sample_size > 0 {
        stbl.all_samples_size = sample_size;
    } else {
        let needed = sample_count
            .checked_mul(4)
            .ok_or(Error::InvalidData("stsz boyut taşması"))?;
        if needed > body.len().saturating_sub(8) {
            return Err(Error::InvalidData("kesik stsz örnek boyut dizisi"));
        }
        reserve_table_budget(sample_count, core::mem::size_of::<u32>(), guard)?;
        stbl.sample_sizes
            .try_reserve_exact(sample_count)
            .map_err(|_| Error::AllocationFailed {
                requested: sample_count.saturating_mul(core::mem::size_of::<u32>()),
            })?;
        for i in 0..sample_count {
            let sz = read_u32_be(body, 8 + i * 4).ok_or(Error::InvalidData("kesik stsz"))?;
            stbl.sample_sizes.push(sz);
        }
    }
    Ok(())
}

fn parse_stts(payload: &[u8], stbl: &mut RawSampleTable, guard: &mut ButceTutucu) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver != 0 {
        return Err(Error::InvalidData("stts sürümü 0 olmalıdır"));
    }
    let count = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik stts entry_count"))? as usize;
    let needed = count
        .checked_mul(8)
        .ok_or(Error::InvalidData("stts boyut taşması"))?;
    if needed > body.len().saturating_sub(4) {
        return Err(Error::InvalidData("kesik stts giriş dizisi"));
    }
    reserve_table_budget(count, core::mem::size_of::<TimeToSample>(), guard)?;
    stbl.time_to_samples
        .try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: count.saturating_mul(core::mem::size_of::<TimeToSample>()),
        })?;
    for i in 0..count {
        let base = 4 + i * 8;
        let sample_count = read_u32_be(body, base).ok_or(Error::InvalidData("kesik stts"))?;
        let sample_delta = read_u32_be(body, base + 4).ok_or(Error::InvalidData("kesik stts"))?;
        stbl.time_to_samples.push(TimeToSample {
            sample_count,
            sample_delta,
        });
    }
    Ok(())
}

fn parse_ctts(payload: &[u8], stbl: &mut RawSampleTable, guard: &mut ButceTutucu) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver > 1 {
        return Err(Error::Unsupported("ctts sürümü > 1 desteklenmiyor"));
    }
    let count = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik ctts entry_count"))? as usize;
    let needed = count
        .checked_mul(8)
        .ok_or(Error::InvalidData("ctts boyut taşması"))?;
    if needed > body.len().saturating_sub(4) {
        return Err(Error::InvalidData("kesik ctts giriş dizisi"));
    }
    reserve_table_budget(count, core::mem::size_of::<CompositionOffset>(), guard)?;
    stbl.composition_offsets
        .try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: count.saturating_mul(core::mem::size_of::<CompositionOffset>()),
        })?;
    for i in 0..count {
        let base = 4 + i * 8;
        let sample_count = read_u32_be(body, base).ok_or(Error::InvalidData("kesik ctts"))?;
        let sample_offset = if ver == 0 {
            i64::from(read_u32_be(body, base + 4).ok_or(Error::InvalidData("kesik ctts v0"))?)
        } else {
            i64::from(read_i32_be(body, base + 4).ok_or(Error::InvalidData("kesik ctts v1"))?)
        };
        stbl.composition_offsets.push(CompositionOffset {
            sample_count,
            sample_offset,
        });
    }
    Ok(())
}

fn parse_stss(payload: &[u8], stbl: &mut RawSampleTable, guard: &mut ButceTutucu) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver != 0 {
        return Err(Error::InvalidData("stss sürümü 0 olmalıdır"));
    }
    let count = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik stss entry_count"))? as usize;
    let needed = count
        .checked_mul(4)
        .ok_or(Error::InvalidData("stss boyut taşması"))?;
    if needed > body.len().saturating_sub(4) {
        return Err(Error::InvalidData("kesik stss senkron örnek dizisi"));
    }
    reserve_table_budget(count, core::mem::size_of::<u32>(), guard)?;
    stbl.has_stss = true;
    stbl.sync_samples
        .try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: count.saturating_mul(core::mem::size_of::<u32>()),
        })?;
    for i in 0..count {
        let num = read_u32_be(body, 4 + i * 4).ok_or(Error::InvalidData("kesik stss"))?;
        stbl.sync_samples.push(num);
    }
    Ok(())
}

/// İleri yönlü `O(1)` amortize `stsc` imleci (`stsc` `first_chunk` kesin artan sıradadır).
struct StscCursor<'a> {
    entries: &'a [SampleToChunk],
    idx: usize,
}

impl<'a> StscCursor<'a> {
    fn new(entries: &'a [SampleToChunk]) -> Self {
        Self { entries, idx: 0 }
    }

    #[inline]
    fn samples_for_chunk(&mut self, chunk_index: usize) -> u32 {
        if self.entries.is_empty() {
            return 0;
        }
        let chunk_num = (chunk_index as u32).saturating_add(1);
        while self.idx + 1 < self.entries.len()
            && self.entries[self.idx + 1].first_chunk <= chunk_num
        {
            self.idx += 1;
        }
        if self.entries[self.idx].first_chunk <= chunk_num {
            self.entries[self.idx].samples_per_chunk
        } else {
            0
        }
    }
}

/// İleri yönlü `O(1)` amortize `stts` imleci.
struct SttsCursor<'a> {
    entries: &'a [TimeToSample],
    idx: usize,
    rem_in_entry: u32,
    cur_delta: u64,
}

impl<'a> SttsCursor<'a> {
    fn new(entries: &'a [TimeToSample]) -> Self {
        Self {
            entries,
            idx: 0,
            rem_in_entry: 0,
            cur_delta: 0,
        }
    }

    #[inline]
    fn next_delta(&mut self) -> u64 {
        while self.rem_in_entry == 0 {
            let Some(entry) = self.entries.get(self.idx) else {
                return 0;
            };
            self.idx += 1;
            self.rem_in_entry = entry.sample_count;
            self.cur_delta = u64::from(entry.sample_delta);
        }
        self.rem_in_entry -= 1;
        self.cur_delta
    }
}

/// İleri yönlü `O(1)` amortize `ctts` (`CompositionOffsetBox`) imleci.
struct CttsCursor<'a> {
    entries: &'a [CompositionOffset],
    idx: usize,
    rem_in_entry: u32,
    cur_offset: i64,
}

impl<'a> CttsCursor<'a> {
    fn new(entries: &'a [CompositionOffset]) -> Self {
        Self {
            entries,
            idx: 0,
            rem_in_entry: 0,
            cur_offset: 0,
        }
    }

    #[inline]
    fn next_offset(&mut self) -> i64 {
        while self.rem_in_entry == 0 {
            let Some(entry) = self.entries.get(self.idx) else {
                return 0;
            };
            self.idx += 1;
            self.rem_in_entry = entry.sample_count;
            self.cur_offset = entry.sample_offset;
        }
        self.rem_in_entry -= 1;
        self.cur_offset
    }
}

fn build_track_samples(
    stbl: &RawSampleTable,
    file_len: usize,
    butce: &Arc<Butce>,
) -> Result<(Vec<TrackSample>, ButceTutucu)> {
    let mut sample_guard = ButceTutucu::bos(butce);
    if stbl.chunks.is_empty() {
        return Ok((Vec::new(), sample_guard));
    }
    let mut total_samples = 0usize;
    let mut count_stsc = StscCursor::new(&stbl.sample_to_chunks);
    for chunk_idx in 0..stbl.chunks.len() {
        let count = count_stsc.samples_for_chunk(chunk_idx) as usize;
        if count == 0 {
            return Err(Error::InvalidData("stbl içinde 0 örnekli yığın (chunk)"));
        }
        total_samples = total_samples
            .checked_add(count)
            .ok_or(Error::InvalidData("parça toplam örnek sayısı taşması"))?;
        if total_samples > MAX_TRACK_SAMPLES {
            return Err(Error::BudgetExceeded {
                requested: total_samples,
                limit: MAX_TRACK_SAMPLES,
            });
        }
    }
    reserve_table_budget(
        total_samples,
        core::mem::size_of::<TrackSample>(),
        &mut sample_guard,
    )?;

    let mut samples = Vec::new();
    samples
        .try_reserve_exact(total_samples)
        .map_err(|_| Error::AllocationFailed {
            requested: total_samples.saturating_mul(core::mem::size_of::<TrackSample>()),
        })?;
    let mut stsc_cursor = StscCursor::new(&stbl.sample_to_chunks);
    let mut stts_cursor = SttsCursor::new(&stbl.time_to_samples);
    let mut ctts_cursor = CttsCursor::new(&stbl.composition_offsets);
    let has_ctts = !stbl.composition_offsets.is_empty();
    let mut size_idx = 0usize;
    let mut dts = 0u64;

    for (chunk_idx, &chunk_off) in stbl.chunks.iter().enumerate() {
        let count = stsc_cursor.samples_for_chunk(chunk_idx) as usize;
        let mut cur_off = chunk_off;
        for _ in 0..count {
            let sz_u32 = if stbl.all_samples_size > 0 {
                stbl.all_samples_size
            } else {
                *stbl
                    .sample_sizes
                    .get(size_idx)
                    .ok_or(Error::InvalidData("kesik stsz örnek boyut tablosu"))?
            };
            if sz_u32 == 0 {
                return Err(Error::InvalidData("stbl içinde sıfır baytlık örnek"));
            }
            let end_u64 = cur_off
                .checked_add(u64::from(sz_u32))
                .ok_or(Error::InvalidData("örnek dosya ofseti taşması"))?;
            let off_usize = usize::try_from(cur_off)
                .map_err(|_| Error::InvalidData("örnek ofseti usize sınırını aşıyor"))?;
            let sz_usize = usize::try_from(sz_u32)
                .map_err(|_| Error::InvalidData("örnek boyutu usize sınırını aşıyor"))?;
            let end_usize = usize::try_from(end_u64)
                .map_err(|_| Error::InvalidData("örnek bitiş ofseti usize sınırını aşıyor"))?;
            if end_usize > file_len {
                return Err(Error::InvalidData(
                    "örnek veri aralığı dosya sınırları dışında",
                ));
            }

            let dur = stts_cursor.next_delta();
            let pts = if has_ctts {
                let ctts_off = ctts_cursor.next_offset();
                let pts_i128 = i128::from(dts) + i128::from(ctts_off);
                if pts_i128 < 0 || pts_i128 > i128::from(u64::MAX) {
                    return Err(Error::InvalidData(
                        "ctts bileşik zaman damgası (pts) aralık dışında",
                    ));
                }
                pts_i128 as u64
            } else {
                dts
            };
            samples.push(TrackSample {
                offset: off_usize,
                size: sz_usize,
                pts_in_timescale: pts,
                duration_in_timescale: dur,
                is_keyframe: !stbl.has_stss,
            });
            dts = dts.saturating_add(dur);
            cur_off = end_u64;
            size_idx += 1;
        }
    }

    if stbl.has_stss {
        for &sample_num in &stbl.sync_samples {
            if let Some(idx) = sample_num.checked_sub(1).map(|v| v as usize)
                && let Some(sample) = samples.get_mut(idx)
            {
                sample.is_keyframe = true;
            }
        }
    }
    if let Some(first) = samples.first_mut() {
        first.is_keyframe = true;
    }

    Ok((samples, sample_guard))
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{
        CompositionOffset, RawSampleTable, SampleToChunk, TimeToSample, build_track_samples,
        parse_ctts,
    };
    use crate::error::Error;
    use crate::frame::butce::{Butce, ButceTutucu};

    #[test]
    fn build_track_samples_linear_with_100k_entries() {
        let n = 100_000usize;
        let mut chunks = Vec::with_capacity(n);
        let mut stsc = Vec::with_capacity(n);
        let mut stts = Vec::with_capacity(n);
        for i in 0..n {
            chunks.push((i as u64) * 16);
            stsc.push(SampleToChunk {
                first_chunk: (i as u32) + 1,
                samples_per_chunk: 1,
            });
            stts.push(TimeToSample {
                sample_count: 1,
                sample_delta: 30,
            });
        }
        let stbl = RawSampleTable {
            chunks,
            sample_to_chunks: stsc,
            all_samples_size: 16,
            sample_sizes: Vec::new(),
            time_to_samples: stts,
            composition_offsets: Vec::new(),
            has_stss: false,
            sync_samples: Vec::new(),
            is_av01: true,
            properties: Vec::new(),
        };
        let butce = Arc::new(Butce::yeni(None, 0));
        let (samples, _g) =
            build_track_samples(&stbl, n * 16, &butce).expect("100k örnek hızla üretilmeli");
        assert_eq!(samples.len(), n);
        assert_eq!(samples[0].pts_in_timescale, 0);
        assert_eq!(samples[n - 1].pts_in_timescale, ((n - 1) as u64) * 30);
    }

    #[test]
    fn ctts_v0_and_signed_v1_adjust_pts_and_reject_negative_pts() {
        let butce = Arc::new(Butce::yeni(None, 0));

        // ctts v0 (u32 ofsetler): dts=[0, 10, 20], offset=[20, 0, 10] -> pts=[20, 10, 30]
        let mut ctts_v0 = vec![0u8, 0, 0, 0]; // ver=0, flags=0
        ctts_v0.extend_from_slice(&3u32.to_be_bytes());
        for (cnt, off) in [(1u32, 20u32), (1, 0), (1, 10)] {
            ctts_v0.extend_from_slice(&cnt.to_be_bytes());
            ctts_v0.extend_from_slice(&off.to_be_bytes());
        }
        let mut stbl_v0 = RawSampleTable {
            chunks: vec![0],
            sample_to_chunks: vec![SampleToChunk {
                first_chunk: 1,
                samples_per_chunk: 3,
            }],
            all_samples_size: 4,
            time_to_samples: vec![TimeToSample {
                sample_count: 3,
                sample_delta: 10,
            }],
            ..RawSampleTable::default()
        };
        let mut g0 = ButceTutucu::bos(&butce);
        parse_ctts(&ctts_v0, &mut stbl_v0, &mut g0).expect("ctts v0 ayrıştırılmalı");
        let (s0, _) = build_track_samples(&stbl_v0, 64, &butce).unwrap();
        assert_eq!(
            s0.iter().map(|s| s.pts_in_timescale).collect::<Vec<_>>(),
            vec![20, 10, 30]
        );

        // ctts v1 (işaretli i32 ofsetler): dts=[0, 10, 20], offset=[+10, -5, -10] -> pts=[10, 5, 10]
        let mut ctts_v1 = vec![1u8, 0, 0, 0]; // ver=1, flags=0
        ctts_v1.extend_from_slice(&3u32.to_be_bytes());
        for (cnt, off) in [(1u32, 10i32), (1, -5), (1, -10)] {
            ctts_v1.extend_from_slice(&cnt.to_be_bytes());
            ctts_v1.extend_from_slice(&off.to_be_bytes());
        }
        let mut stbl_v1 = RawSampleTable {
            chunks: vec![0],
            sample_to_chunks: vec![SampleToChunk {
                first_chunk: 1,
                samples_per_chunk: 3,
            }],
            all_samples_size: 4,
            time_to_samples: vec![TimeToSample {
                sample_count: 3,
                sample_delta: 10,
            }],
            ..RawSampleTable::default()
        };
        let mut g1 = ButceTutucu::bos(&butce);
        parse_ctts(&ctts_v1, &mut stbl_v1, &mut g1).expect("ctts v1 ayrıştırılmalı");
        let (s1, _) = build_track_samples(&stbl_v1, 64, &butce).unwrap();
        assert_eq!(
            s1.iter().map(|s| s.pts_in_timescale).collect::<Vec<_>>(),
            vec![10, 5, 10]
        );

        // Negatif pts (dts=0 + offset=-1) reddedilmeli.
        let stbl_neg = RawSampleTable {
            chunks: vec![0],
            sample_to_chunks: vec![SampleToChunk {
                first_chunk: 1,
                samples_per_chunk: 1,
            }],
            all_samples_size: 4,
            time_to_samples: vec![TimeToSample {
                sample_count: 1,
                sample_delta: 10,
            }],
            composition_offsets: vec![CompositionOffset {
                sample_count: 1,
                sample_offset: -1,
            }],
            ..RawSampleTable::default()
        };
        assert!(matches!(
            build_track_samples(&stbl_neg, 64, &butce),
            Err(Error::InvalidData(_))
        ));
    }
}
