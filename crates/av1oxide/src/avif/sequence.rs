//! Akışlı animasyonlu AVIF / Image Sequence (`.avis`) çözücü katmanı (`AvifSequenceDecoder`).

#![forbid(unsafe_code)]

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::api::{Config, Decoder, Packet};
use crate::avif::boxes::{Av1CProperty, ItemProperty};
use crate::avif::decode::{
    LIMITED_TO_FULL_Y_8BIT, PlanarBuffer, apply_imir, apply_irot, av1c_to_format,
    check_pixel_and_byte_budget, compute_clap_rect, crop_planar, decode_with,
    extract_metadata_items, extract_plane, extract_track_metadata_items, limited_to_full_y_u16,
    parse_info, resolve_effective_threads,
};
use crate::avif::image::{
    AvifDecodeConfig, AvifImage, CleanAperture, ColorInfo, ContentLightLevel, MasteringDisplay,
    PixelAspectRatio, PlaneStorage,
};
use crate::avif::tracks::{LoopCount, ParsedTrack, TrackSample, parse_movie};
use crate::bitstream::obu::parse_sequence_header_obus;
use crate::bitstream::seq_header::PixelLayout;
use crate::error::{Error, Result, SizeStage};
use crate::frame::{BitDepth, ChromaFormat, Frame, PlaneType};

/// Animasyonlu AVIF (`.avis`) veya tek karelik `.avif` dizisinin üst bilgisi.
#[derive(Clone, Debug, PartialEq)]
pub struct AvifSequenceInfo {
    /// Dönüşümler (`clap`, `irot`, `imir`) sonrası nihai genişlik (piksel).
    pub width: u32,
    /// Dönüşümler (`clap`, `irot`, `imir`) sonrası nihai yükseklik (piksel).
    pub height: u32,
    /// Kanal başına bit derinliği (`8`, `10` veya `12` bit).
    pub bit_depth: BitDepth,
    /// Kroma alt örnekleme biçimi (`4:2:0`, `4:2:2`, `4:4:4` veya `Monochrome`).
    pub chroma_format: ChromaFormat,
    /// Yardımcı alfa parçasının / kanalının bulunup bulunmadığı.
    pub has_alpha: bool,
    /// Renk kanallarının alfa ile ön-çarpımlı (`prem`) olup olmadığı.
    pub premultiplied_alpha: bool,
    /// Dizideki toplam kare sayısı.
    pub frame_count: usize,
    /// Medya zaman tabanı (`mdhd.timescale`, saniye başına birim).
    pub timescale: u64,
    /// Toplam süre (`timescale` biriminde).
    pub duration_in_timescale: u64,
    /// Toplam süre (saniye).
    pub duration_sec: f64,
    /// Tekrarlama (döngü) sayısı.
    pub repetition_count: LoopCount,
    /// CICP / H.273 renk tanımlayıcıları.
    pub color: ColorInfo,
    /// Varsa gömülü ICC renk profili baytları.
    pub icc_profile: Option<Vec<u8>>,
    /// Varsa TIFF başlığından başlayan ham Exif metaverisi.
    pub exif: Option<Vec<u8>>,
    /// Varsa ham XMP (`application/rdf+xml`) metaverisi.
    pub xmp: Option<Vec<u8>>,
    /// Varsa içerik ışık düzeyi (`clli`) bilgisi.
    pub clli: Option<ContentLightLevel>,
    /// Varsa referans ekran renk hacmi (`mdcv`) bilgisi.
    pub mdcv: Option<MasteringDisplay>,
    /// Varsa piksel en-boy oranı (`pasp`) bilgisi.
    pub pasp: Option<PixelAspectRatio>,
}

/// Akışlı [`AvifSequenceDecoder`] tarafından çözülen tekil animasyon karesi.
#[derive(Clone, Debug, PartialEq)]
pub struct AvifFrame {
    /// Dönüşümleri uygulanmış düzlemsel AVIF görüntüsü.
    pub image: AvifImage,
    /// 0-tabanlı kare indeksi.
    pub index: usize,
    /// Karenin sunum zaman damgası (`timescale` biriminde).
    pub pts_in_timescale: u64,
    /// Karenin görünme süresi (`timescale` biriminde).
    pub duration_in_timescale: u64,
    /// Karenin sunum zaman damgası (saniye).
    pub pts_sec: f64,
    /// Karenin görünme süresi (saniye).
    pub duration_sec: f64,
    /// Karenin hem renk hem (varsa) alfa parçası için bağımsız anahtar kare (`stss`) olup olmadığı.
    pub is_keyframe: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TrackTransforms {
    clap: Option<CleanAperture>,
    irot: Option<u8>,
    imir: Option<u8>,
}

struct TrackMetadataBundle {
    av1c_config_obus: Vec<u8>,
    bit_depth: BitDepth,
    chroma_format: ChromaFormat,
    color: Option<ColorInfo>,
    icc_profile: Option<Vec<u8>>,
    clli: Option<ContentLightLevel>,
    mdcv: Option<MasteringDisplay>,
    pasp: Option<PixelAspectRatio>,
    transforms: TrackTransforms,
}

fn extract_track_properties(track: &ParsedTrack<'_>, data: &[u8]) -> Result<TrackMetadataBundle> {
    let mut av1c_opt: Option<&Av1CProperty> = None;
    let mut color: Option<ColorInfo> = None;
    let mut icc_profile: Option<Vec<u8>> = None;
    let mut clli: Option<ContentLightLevel> = None;
    let mut mdcv: Option<MasteringDisplay> = None;
    let mut pasp: Option<PixelAspectRatio> = None;
    let mut transforms = TrackTransforms::default();

    for prop in &track.properties {
        match prop {
            ItemProperty::Av1C(av1c) => av1c_opt = Some(av1c),
            ItemProperty::ColrNclx(c) => color = Some(*c),
            ItemProperty::ColrIcc(icc) => icc_profile = Some(icc.clone()),
            ItemProperty::Clli(c) => clli = Some(*c),
            ItemProperty::Mdcv(m) => mdcv = Some(*m),
            ItemProperty::Pasp(pa) => pasp = Some(*pa),
            ItemProperty::Clap(c) => transforms.clap = Some(*c),
            ItemProperty::Irot(a) => transforms.irot = Some(*a),
            ItemProperty::Imir(m) => transforms.imir = Some(*m),
            _ => {}
        }
    }

    let mut bit_depth: Option<BitDepth> = None;
    let mut chroma_format: Option<ChromaFormat> = None;
    let mut av1c_config_obus = Vec::new();
    let mut seq_color: Option<ColorInfo> = None;

    if let Some(av1c) = av1c_opt {
        let (bd, cf) = av1c_to_format(av1c)?;
        bit_depth = Some(bd);
        chroma_format = Some(cf);
        av1c_config_obus = av1c.config_obus.clone();
        if !av1c.config_obus.is_empty()
            && let Ok(Some(seq)) = parse_sequence_header_obus(&av1c.config_obus)
        {
            seq_color = Some(ColorInfo {
                color_primaries: u16::from(seq.pri),
                transfer_characteristics: u16::from(seq.trc),
                matrix_coefficients: u16::from(seq.mtrx),
                full_range: seq.color_range != 0,
            });
        }
    }

    // `colr nclx` yoksa ilk örneğin AV1 Sequence Header'ından CICP / format bilgisini oku.
    if (color.is_none() || bit_depth.is_none() || chroma_format.is_none())
        && let Some(first) = track.samples.first()
    {
        let sample_bytes = &data[first.offset..first.offset + first.size];
        let search_len = sample_bytes.len().min(4096);
        if let Ok(Some(seq)) = parse_sequence_header_obus(&sample_bytes[..search_len]) {
            if bit_depth.is_none() {
                bit_depth = Some(match seq.hbd {
                    0 => BitDepth::Eight,
                    1 => BitDepth::Ten,
                    _ => BitDepth::Twelve,
                });
            }
            if chroma_format.is_none() {
                chroma_format = Some(match seq.layout {
                    PixelLayout::I400 => ChromaFormat::Monochrome,
                    PixelLayout::I420 => ChromaFormat::C420,
                    PixelLayout::I422 => ChromaFormat::C422,
                    PixelLayout::I444 => ChromaFormat::C444,
                });
            }
            if seq_color.is_none() {
                seq_color = Some(ColorInfo {
                    color_primaries: u16::from(seq.pri),
                    transfer_characteristics: u16::from(seq.trc),
                    matrix_coefficients: u16::from(seq.mtrx),
                    full_range: seq.color_range != 0,
                });
            }
        }
    }

    Ok(TrackMetadataBundle {
        av1c_config_obus,
        bit_depth: bit_depth.unwrap_or(BitDepth::Eight),
        chroma_format: chroma_format.unwrap_or(ChromaFormat::C420),
        color: color.or(seq_color),
        icc_profile,
        clli,
        mdcv,
        pasp,
        transforms,
    })
}

fn apply_track_transforms(
    buf: &mut PlanarBuffer,
    transforms: TrackTransforms,
    allocation_limit: Option<usize>,
) -> Result<()> {
    if let Some(clap) = transforms.clap {
        let (x0, y0, cw, ch) = compute_clap_rect(buf.width, buf.height, &clap)?;
        crop_planar(buf, x0, y0, cw, ch, allocation_limit)?;
    }
    if let Some(angle) = transforms.irot {
        apply_irot(buf, angle);
    }
    if let Some(axis) = transforms.imir {
        apply_imir(buf, axis);
    }
    Ok(())
}

fn transformed_dimensions(
    raw_w: u32,
    raw_h: u32,
    transforms: TrackTransforms,
) -> Result<(u32, u32)> {
    let mut w = raw_w;
    let mut h = raw_h;
    if let Some(clap) = transforms.clap {
        let (_, _, cw, ch) = compute_clap_rect(w, h, &clap)?;
        w = cw;
        h = ch;
    }
    if let Some(angle) = transforms.irot
        && (angle & 1) != 0
    {
        core::mem::swap(&mut w, &mut h);
    }
    Ok((w, h))
}

fn check_alpha_track_transform_compat(
    color_x: TrackTransforms,
    alpha_x: TrackTransforms,
) -> Result<()> {
    if alpha_x.clap.is_none() && alpha_x.irot.is_none() && alpha_x.imir.is_none() {
        return Ok(());
    }
    if alpha_x != color_x {
        return Err(Error::Unsupported(
            "alfa parçası dönüşüm özellikleri renk parçası ile eşleşmiyor",
        ));
    }
    Ok(())
}

struct TrackSequenceState {
    color_samples: Vec<TrackSample>,
    alpha_samples: Option<Vec<TrackSample>>,
    color_config_obus: Vec<u8>,
    alpha_config_obus: Vec<u8>,
    transforms: TrackTransforms,
    color_decoder: Option<Decoder>,
    alpha_decoder: Option<Decoder>,
    next_index: usize,
}

enum SequenceBackend {
    Track(Box<TrackSequenceState>),
    Still { yielded: bool },
}

/// Animasyonlu AVIF (`.avis`) ve tek karelik `.avif` dosyaları için akışlı dizi çözücüsü.
///
/// Tüm kareleri bellekte biriktirmek yerine kalıcı AV1 çözücü örneği üzerinden kare-kare
/// (`next_frame`) veya en yakın anahtar kareden rastgele erişimle (`nth_frame`) çözer.
pub struct AvifSequenceDecoder<'a> {
    data: &'a [u8],
    config: AvifDecodeConfig,
    info: AvifSequenceInfo,
    backend: SequenceBackend,
}

/// Piksel çözme işlemi yapmadan animasyonlu (`.avis`) veya tek karelik `.avif` dizisinin
/// başlık, zamanlama, döngü sayısı ve metaveri bilgilerini ayrıştırır.
pub fn parse_sequence_info(data: &[u8]) -> Result<AvifSequenceInfo> {
    let decoder = AvifSequenceDecoder::new(data)?;
    Ok(decoder.info)
}

impl<'a> AvifSequenceDecoder<'a> {
    /// Varsayılan [`AvifDecodeConfig`] ile bir `.avis` / `.avif` dizi çözücüsü oluşturur.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        Self::with_config(data, AvifDecodeConfig::default())
    }

    /// Belirtilen [`AvifDecodeConfig`] ile bir `.avis` / `.avif` dizi çözücüsü oluşturur.
    pub fn with_config(data: &'a [u8], config: AvifDecodeConfig) -> Result<Self> {
        let movie = parse_movie(data, config.allocation_limit)?;

        if movie.prefers_tracks() {
            let (color_idx, alpha_idx) = movie.select_color_and_alpha_tracks()?;
            let color_track = &movie.tracks[color_idx];
            let alpha_track = alpha_idx.map(|i| &movie.tracks[i]);

            check_pixel_and_byte_budget(color_track.width, color_track.height, 0, &config)?;
            let color_bundle = extract_track_properties(color_track, data)?;
            let (alpha_config_obus, alpha_samples) = if let Some(at) = alpha_track {
                check_pixel_and_byte_budget(at.width, at.height, 0, &config)?;
                let alpha_bundle = extract_track_properties(at, data)?;
                check_alpha_track_transform_compat(
                    color_bundle.transforms,
                    alpha_bundle.transforms,
                )?;
                (alpha_bundle.av1c_config_obus, Some(at.samples.clone()))
            } else {
                (Vec::new(), None)
            };

            let (post_w, post_h) = transformed_dimensions(
                color_track.width,
                color_track.height,
                color_bundle.transforms,
            )?;

            // Exif/XMP önceliği: önce `trak/meta` (libavif `colorTrack->meta`), yoksa üst düzey `meta`.
            let (mut exif, mut xmp) = if let Some(ref tmeta) = color_track.meta {
                extract_track_metadata_items(tmeta, config.allocation_limit)?
            } else {
                (None, None)
            };
            let mut icc_profile = color_bundle.icc_profile;
            if let Some(ref top_meta) = movie.top_meta {
                if (exif.is_none() || xmp.is_none()) && top_meta.primary_item_id != 0 {
                    let (top_exif, top_xmp) = extract_metadata_items(
                        top_meta,
                        top_meta.primary_item_id,
                        config.allocation_limit,
                    )?;
                    if exif.is_none() {
                        exif = top_exif;
                    }
                    if xmp.is_none() {
                        xmp = top_xmp;
                    }
                }
                if icc_profile.is_none()
                    && top_meta.primary_item_id != 0
                    && let Ok(props) = top_meta.item_properties(top_meta.primary_item_id)
                {
                    for (p, _) in props {
                        if let ItemProperty::ColrIcc(icc) = p {
                            icc_profile = Some(icc.clone());
                            break;
                        }
                    }
                }
            }

            let timescale = color_track.media_timescale;
            let duration_in_timescale = color_track.media_duration;
            let duration_sec = if timescale > 0 {
                (duration_in_timescale as f64) / (timescale as f64)
            } else {
                0.0
            };

            let premultiplied_alpha = alpha_track.is_some_and(|at| {
                color_track.references_track(b"prem", at.track_id)
                    || at.references_track(b"prem", color_track.track_id)
            });

            let info = AvifSequenceInfo {
                width: post_w,
                height: post_h,
                bit_depth: color_bundle.bit_depth,
                chroma_format: color_bundle.chroma_format,
                has_alpha: alpha_track.is_some(),
                premultiplied_alpha,
                frame_count: color_track.samples.len(),
                timescale,
                duration_in_timescale,
                duration_sec,
                repetition_count: color_track.repetition_count,
                color: color_bundle.color.unwrap_or_default(),
                icc_profile,
                exif,
                xmp,
                clli: color_bundle.clli,
                mdcv: color_bundle.mdcv,
                pasp: color_bundle.pasp,
            };

            let state = TrackSequenceState {
                color_samples: color_track.samples.clone(),
                alpha_samples,
                color_config_obus: color_bundle.av1c_config_obus,
                alpha_config_obus,
                transforms: color_bundle.transforms,
                color_decoder: None,
                alpha_decoder: None,
                next_index: 0,
            };

            Ok(Self {
                data,
                config,
                info,
                backend: SequenceBackend::Track(Box::new(state)),
            })
        } else {
            // Hareketsiz AVIF dosyası: tek karelik dizi olarak şeffaf destek.
            let still = parse_info(data)?;
            let info = AvifSequenceInfo {
                width: still.width,
                height: still.height,
                bit_depth: still.bit_depth,
                chroma_format: still.chroma_format,
                has_alpha: still.has_alpha,
                premultiplied_alpha: still.premultiplied_alpha,
                frame_count: 1,
                timescale: 1,
                duration_in_timescale: 1,
                duration_sec: 1.0,
                repetition_count: LoopCount::Count(0),
                color: still.color,
                icc_profile: still.icc_profile,
                exif: still.exif,
                xmp: still.xmp,
                clli: still.clli,
                mdcv: still.mdcv,
                pasp: still.pasp,
            };
            Ok(Self {
                data,
                config,
                info,
                backend: SequenceBackend::Still { yielded: false },
            })
        }
    }

    /// Ayrıştırılmış dizi üst bilgisini döndürür.
    #[must_use]
    pub const fn info(&self) -> &AvifSequenceInfo {
        &self.info
    }

    /// Dizideki toplam kare sayısını döndürür.
    #[must_use]
    pub const fn frame_count(&self) -> usize {
        self.info.frame_count
    }

    /// Dizinin tekrarlama (döngü) sayısını döndürür.
    #[must_use]
    pub const fn repetition_count(&self) -> LoopCount {
        self.info.repetition_count
    }

    /// Belirtilen `index` karesinin hem renk hem (varsa) alfa parçası için bağımsız anahtar kare
    /// (`stss`) olup olmadığını döndürür (`libavif` `avifDecoderIsKeyframe` uyumlu).
    #[must_use]
    pub fn is_keyframe(&self, index: usize) -> bool {
        match &self.backend {
            SequenceBackend::Still { .. } => index == 0,
            SequenceBackend::Track(st) => {
                let Some(cs) = st.color_samples.get(index) else {
                    return false;
                };
                if !cs.is_keyframe {
                    return false;
                }
                if let Some(ref alphas) = st.alpha_samples {
                    alphas.get(index).is_some_and(|a| a.is_keyframe)
                } else {
                    true
                }
            }
        }
    }

    /// `index` karesine ulaşmak için geriye dönük en yakın ortak anahtar kare indeksini döndürür
    /// (`libavif` `avifDecoderNearestKeyframe` uyumlu).
    #[must_use]
    pub fn nearest_keyframe(&self, index: usize) -> usize {
        let mut idx = index.min(self.info.frame_count.saturating_sub(1));
        while idx > 0 {
            if self.is_keyframe(idx) {
                return idx;
            }
            idx -= 1;
        }
        0
    }

    /// Çözücüyü başa sarar ve dahili AV1 çözücü durumlarını sıfırlar.
    pub fn reset(&mut self) {
        match &mut self.backend {
            SequenceBackend::Still { yielded } => *yielded = false,
            SequenceBackend::Track(st) => {
                st.color_decoder = None;
                st.alpha_decoder = None;
                st.next_index = 0;
            }
        }
    }

    /// Dizideki sıradaki kareyi çözer; tüm kareler tüketildiyse `Ok(None)` döner.
    pub fn next_frame(&mut self) -> Result<Option<AvifFrame>> {
        match &mut self.backend {
            SequenceBackend::Still { yielded } => {
                if *yielded {
                    return Ok(None);
                }
                let image = decode_with(self.data, &self.config)?;
                *yielded = true;
                Ok(Some(AvifFrame {
                    image,
                    index: 0,
                    pts_in_timescale: 0,
                    duration_in_timescale: 1,
                    pts_sec: 0.0,
                    duration_sec: 1.0,
                    is_keyframe: true,
                }))
            }
            SequenceBackend::Track(_) => {
                let idx = match &self.backend {
                    SequenceBackend::Track(st) => st.next_index,
                    SequenceBackend::Still { .. } => unreachable!(),
                };
                if idx >= self.info.frame_count {
                    return Ok(None);
                }
                self.decode_next_track_frame(true)
            }
        }
    }

    /// Belirtilen `index` karesine rastgele erişimle atlar (`seek`) ve o kareyi döndürür.
    ///
    /// Hedef kare geçerli konumun gerisindeyse veya ileri bir ortak anahtar kareye atlamak
    /// mümkünse çözücü en yakın ortak `stss` anahtar karesinden başlatılır ve ara kareler
    /// bellekte biriktirilmeden yalnızca referans güncellemesi için çözülür.
    pub fn nth_frame(&mut self, index: usize) -> Result<AvifFrame> {
        if index >= self.info.frame_count {
            return Err(Error::InvalidData(
                "istenen kare indeksi AVIF dizisi sınırları dışında",
            ));
        }

        match &self.backend {
            SequenceBackend::Still { .. } => {
                self.reset();
                self.next_frame()?
                    .ok_or(Error::InvalidData("tek karelik AVIF çözülemedi"))
            }
            SequenceBackend::Track(st) => {
                let cur_next = st.next_index;
                if index == cur_next {
                    return self
                        .next_frame()?
                        .ok_or(Error::InvalidData("sıradaki kare çözülemedi"));
                }
                let nearest_kf = self.nearest_keyframe(index);
                if (nearest_kf > cur_next || index < cur_next)
                    && let SequenceBackend::Track(st_mut) = &mut self.backend
                {
                    st_mut.color_decoder = None;
                    st_mut.alpha_decoder = None;
                    st_mut.next_index = nearest_kf;
                }
                loop {
                    let cur = match &self.backend {
                        SequenceBackend::Track(s) => s.next_index,
                        SequenceBackend::Still { .. } => unreachable!(),
                    };
                    let is_target = cur == index;
                    let frame_opt = self.decode_next_track_frame(is_target)?;
                    if is_target {
                        return frame_opt
                            .ok_or(Error::InvalidData("hedef kare AVIF dizisinden çözülemedi"));
                    }
                }
            }
        }
    }

    fn decode_next_track_frame(&mut self, build_output: bool) -> Result<Option<AvifFrame>> {
        let is_keyframe_idx = {
            let idx = match &self.backend {
                SequenceBackend::Track(st) => st.next_index,
                SequenceBackend::Still { .. } => return Ok(None),
            };
            self.is_keyframe(idx)
        };

        let SequenceBackend::Track(st) = &mut self.backend else {
            return Ok(None);
        };
        let idx = st.next_index;
        if idx >= st.color_samples.len() {
            return Ok(None);
        }

        let color_sample = st.color_samples[idx];
        let alpha_sample = match &st.alpha_samples {
            Some(alphas) => {
                let s = *alphas.get(idx).ok_or(Error::InvalidData(
                    "alfa parçasında karşılık gelen örnek bulunamadı",
                ))?;
                Some(s)
            }
            None => None,
        };

        let color_dec =
            get_or_init_decoder(&mut st.color_decoder, &self.config, &st.color_config_obus)?;
        let color_frame = decode_single_sample_frame(
            color_dec,
            &self.data[color_sample.offset..color_sample.offset + color_sample.size],
        )?;
        let color_seq_color = color_dec.sequence_header().map(|seq| ColorInfo {
            color_primaries: u16::from(seq.pri),
            transfer_characteristics: u16::from(seq.trc),
            matrix_coefficients: u16::from(seq.mtrx),
            full_range: seq.color_range != 0,
        });

        let alpha_decoded = if let Some(asamp) = alpha_sample {
            let alpha_dec =
                get_or_init_decoder(&mut st.alpha_decoder, &self.config, &st.alpha_config_obus)?;
            let aframe = decode_single_sample_frame(
                alpha_dec,
                &self.data[asamp.offset..asamp.offset + asamp.size],
            )?;
            let a_full_range = alpha_dec
                .sequence_header()
                .is_none_or(|seq| seq.color_range != 0);
            Some((aframe, a_full_range))
        } else {
            None
        };

        st.next_index = idx + 1;
        if !build_output {
            return Ok(None);
        }

        let mut color_buf =
            frame_to_planar_buffer(&color_frame, false, true, color_seq_color, &self.config)?;
        apply_track_transforms(&mut color_buf, st.transforms, self.config.allocation_limit)?;

        let alpha_plane = if let Some((aframe, a_full_range)) = alpha_decoded {
            let mut alpha_buf =
                frame_to_planar_buffer(&aframe, true, a_full_range, None, &self.config)?;
            apply_track_transforms(&mut alpha_buf, st.transforms, self.config.allocation_limit)?;
            if alpha_buf.width != color_buf.width || alpha_buf.height != color_buf.height {
                return Err(Error::InvalidData(
                    "alfa düzlemi boyutları renk karesi boyutlarıyla eşleşmiyor",
                ));
            }
            if alpha_buf.bit_depth != color_buf.bit_depth {
                return Err(Error::InvalidData(
                    "alfa düzlemi bit derinliği renk karesi bit derinliğiyle eşleşmiyor",
                ));
            }
            Some(alpha_buf.y_plane)
        } else {
            None
        };

        let effective_threads = resolve_effective_threads(self.config.n_threads);
        let image = AvifImage {
            width: color_buf.width,
            height: color_buf.height,
            chroma_width: color_buf.chroma_width,
            chroma_height: color_buf.chroma_height,
            bit_depth: color_buf.bit_depth,
            chroma_format: color_buf.chroma_format,
            y_plane: color_buf.y_plane,
            u_plane: color_buf.u_plane,
            v_plane: color_buf.v_plane,
            alpha_plane,
            premultiplied_alpha: self.info.premultiplied_alpha,
            color: self.info.color,
            icc_profile: self.info.icc_profile.clone(),
            exif: self.info.exif.clone(),
            xmp: self.info.xmp.clone(),
            clli: self.info.clli,
            mdcv: self.info.mdcv,
            pasp: self.info.pasp,
            gain_map_metadata: None,
            gain_map: None,
            effective_threads,
        };

        let ts = self.info.timescale;
        let pts_sec = if ts > 0 {
            (color_sample.pts_in_timescale as f64) / (ts as f64)
        } else {
            0.0
        };
        let duration_sec = if ts > 0 {
            (color_sample.duration_in_timescale as f64) / (ts as f64)
        } else {
            0.0
        };

        Ok(Some(AvifFrame {
            image,
            index: idx,
            pts_in_timescale: color_sample.pts_in_timescale,
            duration_in_timescale: color_sample.duration_in_timescale,
            pts_sec,
            duration_sec,
            is_keyframe: is_keyframe_idx,
        }))
    }
}

fn get_or_init_decoder<'d>(
    slot: &'d mut Option<Decoder>,
    config: &AvifDecodeConfig,
    config_obus: &[u8],
) -> Result<&'d mut Decoder> {
    if slot.is_none() {
        let dec_cfg = Config {
            n_threads: config.n_threads,
            max_frame_delay: 1,
            apply_grain: config.apply_grain,
            strict_std_compliance: config.strict,
            frame_size_limit: config.max_pixels,
            allocation_limit: config.allocation_limit,
            operating_point: 0,
            ..Config::default()
        };
        let mut dec = Decoder::new(dec_cfg)?;
        if !config_obus.is_empty() {
            dec.send_packet(&Packet::new(config_obus))?;
        }
        *slot = Some(dec);
    }
    Ok(slot.as_mut().expect("çözücü başlatılmış olmalıdır"))
}

/// Tek bir ISOBMFF örneğini (`sample`) kalıcı AV1 çözücüsüne besler, ilk görünür kareyi alır
/// ve aynı örnekten kalan olası tamponlu kareleri tüketir (`libavif` `codec_dav1d` uyumlu).
fn decode_single_sample_frame(decoder: &mut Decoder, sample_bytes: &[u8]) -> Result<Frame> {
    decoder.send_packet(&Packet::new(sample_bytes))?;
    let first = decoder
        .receive_frame()?
        .ok_or(Error::InvalidData("AV1 dizi örneği görünür kare üretmedi"))?;
    while decoder.receive_frame()?.is_some() {}
    Ok(first)
}

fn frame_to_planar_buffer(
    frame: &Frame,
    is_alpha: bool,
    alpha_full_range: bool,
    seq_color: Option<ColorInfo>,
    config: &AvifDecodeConfig,
) -> Result<PlanarBuffer> {
    let bit_depth = frame.bit_depth();
    let bps = PlaneStorage::bytes_per_sample(bit_depth);
    let y_samples = frame
        .plane(PlaneType::Y)
        .map_or(0usize, |p| p.width().saturating_mul(p.height()));
    let c_samples = if is_alpha {
        0usize
    } else {
        frame
            .plane(PlaneType::U)
            .map_or(0usize, |p| p.width().saturating_mul(p.height()))
            .saturating_add(
                frame
                    .plane(PlaneType::V)
                    .map_or(0usize, |p| p.width().saturating_mul(p.height())),
            )
    };
    let total_bytes =
        y_samples
            .saturating_add(c_samples)
            .checked_mul(bps)
            .ok_or(Error::SizeOverflow {
                stage: SizeStage::TotalAllocation,
                lhs: y_samples.saturating_add(c_samples),
                rhs: bps,
            })?;
    if let Some(limit) = config.allocation_limit
        && total_bytes > limit
    {
        return Err(Error::BudgetExceeded {
            requested: total_bytes,
            limit,
        });
    }

    let (w, h, mut y_plane) = extract_plane(frame, PlaneType::Y, config.allocation_limit)?;
    let (cw, ch, u_plane, v_plane, chroma_format) = if is_alpha {
        (
            0,
            0,
            PlaneStorage::empty_for(bit_depth),
            PlaneStorage::empty_for(bit_depth),
            ChromaFormat::Monochrome,
        )
    } else {
        let (cw, ch, u) = extract_plane(frame, PlaneType::U, config.allocation_limit)?;
        let (_, _, v) = extract_plane(frame, PlaneType::V, config.allocation_limit)?;
        (cw, ch, u, v, frame.chroma_format())
    };

    if is_alpha && !alpha_full_range {
        match &mut y_plane {
            PlaneStorage::U8(samples) => {
                for s in samples {
                    *s = LIMITED_TO_FULL_Y_8BIT[usize::from(*s)];
                }
            }
            PlaneStorage::U16(samples) => {
                for s in samples {
                    *s = limited_to_full_y_u16(bit_depth, *s);
                }
            }
        }
    }

    Ok(PlanarBuffer {
        width: w,
        height: h,
        chroma_width: cw,
        chroma_height: ch,
        bit_depth,
        chroma_format,
        y_plane,
        u_plane,
        v_plane,
        seq_color,
    })
}
