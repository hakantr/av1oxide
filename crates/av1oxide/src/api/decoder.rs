//! Çözücü API'si ve durum makinesi (BLK-01).
//!
//! ## Açık sözleşme (dav1d genel API'si, `src/lib.c`, tek iş parçacığı)
//!
//! | Çağrı | Anlamı | dav1d |
//! |---|---|---|
//! | `Decoder::new(config)` | Yapılandırmayı doğrular, CPU bayraklarını **bir kez** tespit eder ve DSP tablosunu **bir kez** kurar (DSP-01/3-4) | `dav1d_open` + `dav1d_default_settings` |
//! | `send_packet(&Packet)` | OBU'ları sırayla işler; tamamlanan kareyi hemen çözer; hatada paketin kalanı atılır, önceki OBU'ların etkisi kalır | `dav1d_send_data` (`gen_picture`) |
//! | `receive_frame()` | Sıradaki çıktı karesi (`Ok(None)` = şu an yok); varsayılan yalnız gösterilen kareler ve `show_existing` | `dav1d_get_picture` (`EAGAIN` ↔ `Ok(None)`) |
//! | `reset()` | Bekleyen girdi/çıktı, referans yuvaları (başlık + piksel + CDF + segment haritası), dizi ve kare başlığı silinir; yapılandırma korunur | `dav1d_flush` (`src/lib.c:526-557`) |
//! | `flush()` | Girdinin sonu: yeni paket kabul edilmez ([`DecoderState::Draining`]); kalan çıktılar `receive_frame` ile alınır | son `dav1d_send_data`'dan sonra `dav1d_get_picture` ile boşaltma |
//! | `drain_frame_headers()` | Ayrıştırılan kare başlıklarını **çıktı sırasıyla** verir (differential/kabul aracı) | — |
//! | `drop` | `close` eşdeğeridir | `dav1d_close` |
//!
//! Yaşam döngüsü eşdeğerliği `scripts/oracle/lifecycle_oracle.c` (dav1d genel
//! API'si, ayrı süreç) ile `av1oxide-harness lifecycle` karşılaştırılarak
//! sınanır (S2d.2: ard arda akışlar + `reset`, akış ortasında `reset`, OBU
//! bölme, bozuk paket, gönder/al sırası).
//!
//! ## OBU işleme ve hatalar (dav1d `dav1d_parse_obus`, `src/obu.c:1169-1694`)
//!
//! 1. Paket sırayla OBU'lara bölünür ([`crate::bitstream::split_next_with`]);
//!    `obu_has_size_field = 0` olan OBU paketin kalanını kaplar;
//!    `obu_forbidden_bit` yalnız `strict_std_compliance` iken hatadır.
//! 2. Sequence header tam ayrıştırılır; ilk dizi başlığı bekleyen kareyi,
//!    **farklı** dizi başlığı ayrıca referans yuvalarını temizler
//!    (`src/obu.c:1234-1252`). Operating point süzgeci dav1d gibidir.
//! 3. Frame header / frame / tile group OBU'ları dav1d semantiğiyle işlenir;
//!    kare bütün tile'ları gelince **hemen** çözülür. Erken hata (eksik ya da
//!    uyumsuz referans) yuvaları değiştirmez; geç hata (tile boyutu, tile
//!    çözme) yenilenen yuvaları boşaltır (`src/decode.c:3452-3716`).
//! 4. `show_existing_frame` OBU sonunda uygulanır; başarısız olursa başlık
//!    bekler ve sonraki her OBU yeniden dener (`src/obu.c:1527-1548`).
//! 5. Metadata OBU'su dav1d gibi denetlenir (tür leb128'i, HDR CLL/MDCV
//!    taşması); diğer OBU'lar sınırlarıyla atlanır.
//!
//! Bozuk paket [`Error::Bitstream`] (ya da çözme hatası) döner; panik yoktur
//! (GUV-01). Hatadan **önceki** OBU'ların durum değişiklikleri ve çıktıları
//! kalır, paketin kalanı atılır (`src/lib.c:416-431`).

#![forbid(unsafe_code)]

use crate::api::{Config, DecodeFrameType, InloopFilters, Packet};
use crate::bitstream::frame_header::{
    FrameHeader, FrameHeaderContext, FrameType, TileGroupInfo, parse_tile_group,
};
use crate::bitstream::{
    BitReader, Obu, ObuType, SequenceHeader, SequenceHeaderParseOptions, split_next_with,
};
use crate::cpu::CpuFlags;
use crate::decode::intra::{RefFrame, ToolCounts, decode_frame, validate_frame_refs};
use crate::decode::refs::{HeaderReferenceState, RefSlots};
use crate::dsp::DspDispatch;
use crate::error::{BitstreamError, BitstreamErrorKind, Error, Result};
use crate::frame::butce::{Butce, ButceTutucu};
use crate::frame::{BitDepth, ChromaFormat, Frame};
use crate::sync::{Condvar, Mutex};
use crate::thread::{FrameSlot, WorkerPool};
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

/// Çözücünün çağrı sırası durumu (BLK-01).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecoderState {
    /// Yeni paket kabul edilir.
    Ready,
    /// `flush()` çağrıldı (girdinin sonu): kalan çıktılar `receive_frame` ile
    /// alınır, yeni paket kabul edilmez.
    ///
    /// Bu durumdan [`DecoderState::Ready`]'ye dönüş **yalnız**
    /// [`Decoder::reset`] iledir (dav1d `dav1d_flush` karşılığı). dav1d'de ayrı
    /// bir boşaltma durumu yoktur; tek iş parçacığında çıktılar zaten paket
    /// sırasında hazırdır (`src/lib.c:458-482`).
    Draining,
}

/// HDR Mastering Display Color Volume metaverisi (`OBU_META_HDR_MDCV`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MasteringDisplayMeta {
    pub primaries: [[u16; 2]; 3],
    pub white_point: [u16; 2],
    pub max_luminance: u32,
    pub min_luminance: u32,
}

/// ITU-T T.35 metaverisi (`OBU_META_ITUT_T35`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ItutT35Meta {
    pub country_code: u8,
    pub country_code_extension_byte: u8,
    pub payload: Vec<u8>,
}

/// C-ABI (`Dav1dPicture`) çıktı karesi için eşlik eden başlık, metaveri ve olay bayrakları.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
#[derive(Clone, Debug, Default)]
pub(crate) struct OutputCapiMeta {
    pub seq_hdr: Option<Arc<SequenceHeader>>,
    pub frame_hdr: Option<Arc<FrameHeader>>,
    pub content_light: Option<(u16, u16)>,
    pub mastering_display: Option<MasteringDisplayMeta>,
    pub itut_t35: Vec<ItutT35Meta>,
    pub event_flags: u32,
    pub packet_tag: u64,
    /// Bu çıktı karesinden önce en az bir `OBU_TEMPORAL_DELIMITER` görüldü mü?
    pub new_temporal_unit: bool,
    /// Özel `Dav1dPicAllocator` ile kare gönderim aşamasında ön-tahsis edilmiş resim kimliği (`0` = yok).
    pub prealloc_id: u64,
}

/// Özel `Dav1dPicAllocator` için kare gönderim aşamasında (`dav1d_submit_frame` eşleniği)
/// ön-tahsis isteği.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) struct PreallocRequest<'a> {
    pub width: usize,
    pub height: usize,
    pub chroma: ChromaFormat,
    pub bit_depth: BitDepth,
    pub seq_hdr: Option<&'a SequenceHeader>,
    pub frame_hdr: &'a FrameHeader,
    pub content_light: Option<(u16, u16)>,
    pub mastering_display: Option<MasteringDisplayMeta>,
    pub itut_t35: &'a [ItutT35Meta],
}

/// C-ABI katmanından enjekte edilen özel resim ön-tahsis geri çağrısı.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) type PreallocCb<'a> = dyn FnMut(&PreallocRequest<'_>) -> Result<u64> + 'a;

/// Paketler arasında taşınan ayrıştırma durumu (dav1d `Dav1dContext` başlık
/// alanları; OBU başına güncellenir).
#[derive(Clone, Debug, Default)]
struct ParseState {
    /// Etkin sequence header (son başarıyla ayrıştırılan).
    sequence_header: Option<SequenceHeader>,
    /// Etkin sequence header'ın paylaşılan referansı (C-ABI `seq_hdr_ref` için).
    sequence_header_arc: Option<Arc<SequenceHeader>>,
    /// Sonraki çıktı karesine aktarılacak olay bayrakları (`1 = NEW_SEQUENCE`, `2 = NEW_OP_PARAMS_INFO`).
    pending_event_flags: u32,
    /// Etkin HDR Content Light Level metaverisi.
    content_light: Option<(u16, u16)>,
    /// Etkin HDR Mastering Display metaverisi.
    mastering_display: Option<MasteringDisplayMeta>,
    /// Bir sonraki kareye iliştirilecek ITU-T T.35 metaverileri.
    itut_t35: Vec<ItutT35Meta>,
    /// Şu an işlenen paketin C-ABI kimlik etiketi.
    current_packet_tag: u64,
    /// Şu an işlenen paketin zaman damgası.
    current_packet_timestamp: Option<i64>,
    /// Seçili operating point'in `operating_point_idc` maskesi.
    operating_point_idc: u16,
    /// Seçili operating point'in en büyük spatial katman kimliği.
    max_spatial_id: u8,
    /// Başarıyla uygulanan sequence header sayısı.
    sequence_header_updates: u64,
    /// Operating point katman süzgeciyle atlanan OBU sayısı.
    layer_obus_skipped: u64,
    /// Başlık düzeyi referans durumu (8 yuva + bekleyen kare).
    header_state: HeaderReferenceState,
    /// Ayrıştırılan frame header sayısı (redundant dâhil).
    frame_header_parses: u64,
    /// Tamamlanan (bütün tile'ları gelen) kare sayısı.
    frames_completed: u64,
    /// İşlenen tile group OBU sayısı.
    tile_groups_processed: u64,
    /// Sınırlarıyla atlanan tile verisi (bayt).
    tile_bytes_skipped: u64,
    /// Son işlenen tile group'un başlangıç/bitiş ve boyut bilgisi.
    last_tile_group: Option<TileGroupInfo>,
    /// Bekleyen karenin tile group'ları (dav1d `c->tile[]`): tile boyutları
    /// dav1d gibi **çözme** aşamasında ayrıştırılır (`src/decode.c:3159-3189`).
    pending_groups: Vec<TileGroupRaw>,
    /// Başarısız `show_existing_frame` başlığının yuvası: dav1d bu başlığı
    /// `c->frame_hdr`'de tutar ve sonraki **her** OBU'nun sonunda gösterimi
    /// yeniden dener (`src/obu.c:1527-1548`); yeni frame header ya da yeni
    /// dizi gelene kadar her OBU hata verir.
    dangling_show_existing: Option<usize>,
    /// Bir sonraki çıktı üreten kareye taşınacak zamansal birim sınırı bayrağı
    /// (`OBU_TEMPORAL_DELIMITER` veya yeni dizi başlığı).
    pending_new_temporal_unit: bool,
}

/// Tile group OBU'sunun ham verisi (`tile_start`..=`tile_end`, başlık sonrası).
#[derive(Clone, Debug, Default)]
struct TileGroupRaw {
    start: u16,
    end: u16,
    data: Vec<u8>,
}

/// `receive_frame` çıktısı: çözülen kare ya da tanımlı desteklenmeyen durum.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum OutputEntry {
    /// Çözülmüş kare.
    Frame(Frame),
    /// Bu çıktı karesi kapsam dışı (inter/filtre/…).
    Unsupported(&'static str),
}

/// Kareler arası boru hattında (`F7b`, `n_fc >= 2`) tamamlanan bir karenin sonucu.
struct CompletedFrame {
    ref_frame: Option<Arc<RefFrame>>,
    decoded_tools: ToolCounts,
    output: Option<(Frame, ToolCounts, OutputCapiMeta)>,
    _butce_tutucu: Vec<ButceTutucu>,
}

/// Uçuşta (`in_flight`) çözülen tek bir karenin eşzamanlı görev durumu (`F7b`).
struct InFlightDecodeTask {
    refresh_frame_flags: u8,
    packet_tag: u64,
    prealloc_id: u64,
    out_slot: Arc<FrameSlot>,
    started: AtomicBool,
    result: Mutex<Option<Result<CompletedFrame>>>,
    done_cv: Condvar,
}

impl InFlightDecodeTask {
    fn is_done_nonblocking(&self) -> bool {
        self.result.lock().is_some()
    }
}

/// Gösterim sırasını (`display order`) koruyan uçuş kuyruğu öğesi (`F7b`).
#[allow(clippy::large_enum_variant)]
enum InFlightEntry {
    /// Arka plan işçi havuzunda çözülen kare.
    Decode(Arc<InFlightDecodeTask>),
    /// Önceki uçuş karelerinin arkasına sırayla eklenmiş gecikmeli `show_existing_frame` çıktısı.
    ShowExisting {
        promise: Arc<FrameSlot>,
        output_seq: u64,
        refresh_all_key: bool,
        capi_meta: OutputCapiMeta,
        packet_timestamp: Option<i64>,
    },
}

impl InFlightEntry {
    fn is_done_nonblocking(&self) -> bool {
        match self {
            Self::Decode(task) => task.is_done_nonblocking(),
            Self::ShowExisting { promise, .. } => promise.is_done_nonblocking(),
        }
    }

    const fn is_decode(&self) -> bool {
        matches!(self, Self::Decode(_))
    }
}

/// AV1 çözücüsü.
///
/// Kurulumda CPU bayrakları tespit edilir ve DSP tablosu **bir kez** seçilir;
/// tablo decoder ömrü boyunca **sabittir** (DSP-01/4 — piksel başına tespit
/// yoktur). F2a'da çözücü bit akışını ayrıştırır, sequence/frame header'ları
/// ve başlık düzeyi referans durumunu tutar; kare üretmez.
///
/// `Clone`: bütün çözücü durumunun bağımsız kopyası (ayrıştırma durumu, çıktı
/// kuyruğu, sayaçlar); referans kareler değişmez `Arc` olarak paylaşılır ve
/// klon kendi bağımsız `WorkerPool` havuzunu kurar. Bir kopyaya gönderilen
/// paket diğerini etkilemez. Bozuk girdi koşusu (harness `mutate --target frame`)
/// aynı önek durumundan çatallamak için kullanır.
pub struct Decoder {
    config: Config,
    state: DecoderState,
    cpu_flags: CpuFlags,
    dsp: DspDispatch,
    pool: WorkerPool,
    parse: ParseState,
    /// Çıktı sırasıyla biriken kare başlıkları.
    frame_outputs: Vec<FrameHeader>,
    /// `frame_outputs` ile hizalı: çıktı karesi **gösterilecek** mi?
    ///
    /// Normal karelerde `show_frame` bayrağı başlıkta taşınır; ancak
    /// `show_existing_frame` çıktısının başlığı **gösterilen** karenin
    /// başlığıdır (`show_frame=0` olabilir). dav1d show_existing karesini her
    /// durumda çıktı olarak verir (`obu.c:1551-1560`), bu yüzden gösterim
    /// bilgisi ayrı tutulur; `framehdr` differential'ı başlığı değiştirmez.
    frame_output_flags: Vec<bool>,
    /// Çıktı sırasıyla biriken kareler / desteklenmeyen çıktılar.
    outputs: VecDeque<OutputEntry>,
    /// `outputs` ile hizalı tanılama metinleri (çerçeve başına).
    output_errors: VecDeque<Option<String>>,
    /// `outputs` ile hizalı araç sayaçları: **çıktı karesinin kendi**
    /// çözülen karesi (`show_existing` için sıfır).
    output_tools: VecDeque<Option<crate::decode::intra::ToolCounts>>,
    /// `outputs` ile hizalı C-ABI metaveri kayıtları.
    output_capi_meta: VecDeque<OutputCapiMeta>,
    /// Son çözme hatasını üreten paketin C-ABI kimlik etiketi.
    last_error_packet_tag: Option<u64>,
    /// Piksel düzeyi referans yuvaları (piksel + CDF + segment + tmvs).
    pixel_refs: [Option<Arc<RefFrame>>; 8],
    /// Kareler arası boru hattı (`n_fc >= 2`) için referans yuvası vaatleri.
    slot_promises: [Option<Arc<FrameSlot>>; 8],
    /// Uçuşta (`in_flight`) bekleyen kareler ve `show_existing` çıktıları (FIFO).
    in_flight: VecDeque<InFlightEntry>,
    /// Kareler arası boru hattında paylaşılan atomik bütçe defteri.
    pipeline_butce: Arc<Butce>,
    /// Varsa üst katmandan (ör. `avif`) enjekte edilen ortak bütçe defteri.
    shared_butce: Option<Arc<Butce>>,
    /// `reset()` ve `drop()` için uçuş iptal bayrağı.
    cancel_flag: Arc<AtomicBool>,
    /// Başarıyla çözülen kare sayısı.
    decoded_frames: u64,
    /// Kapsam dışı olduğu için çözülemeyen kare sayısı.
    unsupported_frames: u64,
    /// Son çözüm hatasının açıklaması (tanılama; API davranışını değiştirmez).
    last_decode_error: Option<String>,
    /// Son çözülen karenin araç kapsamı sayaçları (tanılama).
    last_tools: Option<crate::decode::intra::ToolCounts>,
    /// Son **çıktı** karesinin araç sayaçları (çıktıya bağlı; §5.8).
    last_output_tools: Option<crate::decode::intra::ToolCounts>,
    /// Çözülen **her** karenin araç sayaçları (tanılama). Çıktı kuyruğu
    /// `show_existing` yüzünden kare sayısından farklı olabildiği için
    /// tablo doğrulaması bu listeden yapılır (T42: `warp_local`).
    decoded_tools: Vec<crate::decode::intra::ToolCounts>,
    /// Tanılama geçmişi (`frame_outputs`, `frame_output_flags`, `decoded_tools`)
    /// toplansın mı? Varsayılan `false` (uzun ömürlü akışlarda sabit bellek).
    collect_diagnostics: bool,
    /// Etkin halka içi filtreler (`dav1d` `inloop_filters` uyumlu; varsayılan [`InloopFilters::ALL`]).
    inloop_filters: InloopFilters,
    /// Çözülecek kare türü filtresi (`dav1d` `decode_frame_type` uyumlu; varsayılan [`DecodeFrameType::All`]).
    decode_frame_type: DecodeFrameType,
    /// Tanılama: yalnız ayrıştırma kipi (varsayılan kapalı; bkz.
    /// [`Decoder::set_parse_only`]).
    parse_only: bool,
    /// Çıktı kuyruğuna şimdiye kadar eklenen giriş sayısı (kare ya da
    /// desteklenmeyen); film grain izinin resim sırası (tanılama).
    outputs_pushed: u64,
    /// Bütçe defterlerinin gördüğü en büyük sayılan canlı bayt (P38; bkz.
    /// [`Decoder::allocation_peak`]).
    allocation_peak: usize,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.cancel_and_drain_in_flight();
    }
}

impl Clone for Decoder {
    fn clone(&self) -> Self {
        let mut cloned = Self {
            config: self.config,
            state: self.state,
            cpu_flags: self.cpu_flags,
            dsp: self.dsp,
            pool: self.pool.clone(),
            parse: self.parse.clone(),
            frame_outputs: self.frame_outputs.clone(),
            frame_output_flags: self.frame_output_flags.clone(),
            outputs: self.outputs.clone(),
            output_errors: self.output_errors.clone(),
            output_tools: self.output_tools.clone(),
            output_capi_meta: self.output_capi_meta.clone(),
            last_error_packet_tag: self.last_error_packet_tag,
            pixel_refs: self.pixel_refs.clone(),
            slot_promises: [None, None, None, None, None, None, None, None],
            in_flight: VecDeque::new(),
            pipeline_butce: match self.shared_butce {
                Some(ref ust) => Arc::new(Butce::alt_defter(ust)),
                None => Arc::new(Butce::yeni(self.config.allocation_limit, 0)),
            },
            shared_butce: self.shared_butce.clone(),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            decoded_frames: self.decoded_frames,
            unsupported_frames: self.unsupported_frames,
            last_decode_error: self.last_decode_error.clone(),
            last_tools: self.last_tools,
            last_output_tools: self.last_output_tools,
            decoded_tools: self.decoded_tools.clone(),
            collect_diagnostics: self.collect_diagnostics,
            inloop_filters: self.inloop_filters,
            decode_frame_type: self.decode_frame_type,
            parse_only: self.parse_only,
            outputs_pushed: self.outputs_pushed,
            allocation_peak: self.allocation_peak(),
        };
        for entry in &self.in_flight {
            match entry {
                InFlightEntry::ShowExisting {
                    promise,
                    output_seq,
                    refresh_all_key,
                    capi_meta,
                    packet_timestamp,
                } => {
                    while !promise.is_done_nonblocking() {
                        if let Some(work) = self.pool.try_pop_async_task() {
                            work();
                        } else {
                            break;
                        }
                    }
                    if let Some(entry) = promise.wait(&self.cancel_flag) {
                        if *refresh_all_key {
                            for reference in &mut cloned.pixel_refs {
                                *reference = Some(Arc::clone(&entry));
                            }
                        }
                        let out_butce = Butce::alt_defter(&cloned.pipeline_butce);
                        let mut tools = ToolCounts::default();
                        if let Ok(mut frame) = build_output_frame_from_ref(
                            cloned.config.apply_grain,
                            &cloned.dsp,
                            &cloned.pool,
                            &entry,
                            *output_seq,
                            &mut tools,
                            &out_butce,
                        ) {
                            if packet_timestamp.is_some() {
                                frame.metadata_mut().timestamp = *packet_timestamp;
                            }
                            cloned.outputs.push_back(OutputEntry::Frame(frame));
                            cloned.output_errors.push_back(None);
                            cloned.output_tools.push_back(Some(tools));
                            cloned.output_capi_meta.push_back(capi_meta.clone());
                        }
                    }
                }
                InFlightEntry::Decode(task) => {
                    while !task.started.load(Ordering::Acquire) && !task.is_done_nonblocking() {
                        if let Some(work) = self.pool.try_pop_async_task() {
                            work();
                        } else {
                            break;
                        }
                    }
                    let mut guard = task.result.lock();
                    while guard.is_none() {
                        guard = task.done_cv.wait(guard);
                    }
                    match guard.as_ref() {
                        Some(Ok(completed)) => {
                            cloned.decoded_frames += 1;
                            cloned.last_tools = Some(completed.decoded_tools);
                            if cloned.collect_diagnostics {
                                cloned.decoded_tools.push(completed.decoded_tools);
                            }
                            if let Some(ref rf) = completed.ref_frame {
                                for (slot, reference) in cloned.pixel_refs.iter_mut().enumerate() {
                                    if task.refresh_frame_flags & (1 << slot) != 0 {
                                        *reference = Some(Arc::clone(rf));
                                    }
                                }
                            }
                            if let Some((frame, tools, capi_meta)) = &completed.output {
                                cloned.outputs.push_back(OutputEntry::Frame(frame.clone()));
                                cloned.output_errors.push_back(None);
                                cloned.output_tools.push_back(Some(*tools));
                                cloned.output_capi_meta.push_back(capi_meta.clone());
                            }
                        }
                        Some(Err(err)) => {
                            cloned
                                .parse
                                .header_state
                                .clear_slots(task.refresh_frame_flags);
                            for (slot, reference) in cloned.pixel_refs.iter_mut().enumerate() {
                                if task.refresh_frame_flags & (1 << slot) != 0 {
                                    *reference = None;
                                }
                            }
                            cloned.unsupported_frames += 1;
                            cloned.last_decode_error = Some(format!("{err}"));
                            cloned.last_error_packet_tag = Some(task.packet_tag);
                        }
                        None => {}
                    }
                }
            }
        }
        for (i, r) in cloned.pixel_refs.iter().enumerate() {
            if let Some(rf) = r {
                cloned.slot_promises[i] = Some(FrameSlot::ready(Arc::clone(rf)));
            }
        }
        cloned.allocation_peak = cloned.allocation_peak.max(self.allocation_peak());
        cloned.pipeline_butce.taban_guncelle(cloned.kalici_bayt());
        cloned
    }
}

impl Decoder {
    /// Yapılandırmayı doğrulayıp çözücüyü kurar.
    ///
    /// Kurulumda: (1) [`Config::validate`], (2) [`CpuFlags::detect`] **bir kez**,
    /// (3) kullanıcı maskesi düşülerek [`DspDispatch::select`] **bir kez**.
    ///
    /// # Hatalar
    ///
    /// [`Error::InvalidArgument`]: bkz. [`Config::validate`].
    pub fn new(config: Config) -> Result<Self> {
        Self::new_inner(config, None)
    }

    /// Dış katmanla (ör. `avif` kapsayıcı/karo/renk hattı) tek bir atomik `Arc<Butce>`
    /// defterini paylaşan çözücü oluşturur.
    pub(crate) fn new_with_shared_budget(
        mut config: Config,
        shared_butce: Arc<Butce>,
    ) -> Result<Self> {
        config.allocation_limit = shared_butce.sinir();
        Self::new_inner(config, Some(shared_butce))
    }

    fn new_inner(config: Config, shared_butce: Option<Arc<Butce>>) -> Result<Self> {
        config.validate()?;
        let detected = CpuFlags::detect();
        let effective = detected.difference(config.cpu_mask);
        let pool = WorkerPool::new(config.effective_threads());
        let pipeline_butce = match shared_butce {
            Some(ref ust) => Arc::new(Butce::alt_defter(ust)),
            None => Arc::new(Butce::yeni(config.allocation_limit, 0)),
        };
        Ok(Self {
            config,
            state: DecoderState::Ready,
            cpu_flags: effective,
            dsp: DspDispatch::select(effective),
            pool,
            parse: ParseState::default(),
            frame_outputs: Vec::new(),
            frame_output_flags: Vec::new(),
            outputs: VecDeque::new(),
            output_errors: VecDeque::new(),
            output_tools: VecDeque::new(),
            output_capi_meta: VecDeque::new(),
            last_error_packet_tag: None,
            pixel_refs: [None, None, None, None, None, None, None, None],
            slot_promises: [None, None, None, None, None, None, None, None],
            in_flight: VecDeque::new(),
            pipeline_butce,
            shared_butce,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            decoded_frames: 0,
            unsupported_frames: 0,
            last_decode_error: None,
            last_tools: None,
            last_output_tools: None,
            decoded_tools: Vec::new(),
            collect_diagnostics: false,
            inloop_filters: InloopFilters::ALL,
            decode_frame_type: DecodeFrameType::All,
            parse_only: false,
            outputs_pushed: 0,
            allocation_peak: 0,
        })
    }

    /// Bütçe defterinin tabanı (P38, `Config::allocation_limit`): kare dışında
    /// yaşayan sayılan ayırmalar — benzersiz referans kareleri
    /// ([`RefFrame::bayt`]), çıktı kuyruğundaki kareler, bekleyen tile verisi
    /// ve kare başına büyüyen tanılama kuyrukları (başlık ve araç sayaçları).
    fn kalici_bayt(&self) -> usize {
        let mut toplam = 0usize;
        for (i, slot) in self.pixel_refs.iter().enumerate() {
            if let Some(r) = slot
                && !self.pixel_refs[..i]
                    .iter()
                    .flatten()
                    .any(|o| Arc::ptr_eq(o, r))
            {
                toplam = toplam.saturating_add(r.bayt());
            }
        }
        for entry in &self.outputs {
            if let OutputEntry::Frame(f) = entry {
                toplam = toplam.saturating_add(f.allocation_bytes());
            }
        }
        for group in &self.parse.pending_groups {
            toplam = toplam.saturating_add(group.data.len());
        }
        toplam
            .saturating_add(core::mem::size_of_val(self.frame_outputs.as_slice()))
            .saturating_add(core::mem::size_of_val(self.decoded_tools.as_slice()))
            .saturating_add(
                self.output_tools.len()
                    * core::mem::size_of::<Option<crate::decode::intra::ToolCounts>>(),
            )
    }

    /// `ek` bayt daha canlıyken yeni bütçe defteri.
    fn butce_defteri(&self, ek: usize) -> Butce {
        if self.shared_butce.is_some() {
            self.pipeline_butce.taban_guncelle(self.kalici_bayt());
            Butce::alt_defter_tabanla(&self.pipeline_butce, ek)
        } else {
            Butce::yeni(
                self.config.allocation_limit,
                self.kalici_bayt().saturating_add(ek),
            )
        }
    }

    #[inline]
    fn sync_shared_kalici(&self) {
        if self.shared_butce.is_some() {
            self.pipeline_butce.taban_guncelle(self.kalici_bayt());
        }
    }

    /// Defterin tepesini çözücünün tepesine katar.
    fn tepe_kat(&mut self, butce: &Butce) {
        self.allocation_peak = self.allocation_peak.max(butce.tepe());
    }

    /// Şimdiye kadar bütçe denetiminden geçen en büyük **sayılan canlı bayt**
    /// (tanılama, P38).
    ///
    /// Aynı girdi aynı yapılandırmayla yeniden çözülürse
    /// `allocation_limit = Some(allocation_peak())` bütün ayırmaları geçirir,
    /// bir bayt eksiği en az bir [`Error::BudgetExceeded`] üretir (sayım
    /// belirlenimlidir). Sayılan ve sayılmayan ayırmalar:
    /// `docs/baseline/kaynak-siniri.md`. [`Decoder::reset`] bu değeri
    /// sıfırlamaz.
    #[must_use]
    pub fn allocation_peak(&self) -> usize {
        self.allocation_peak.max(self.pipeline_butce.tepe())
    }

    /// Tanılama: yalnız ayrıştırma kipi (varsayılan **kapalı**).
    ///
    /// Açıkken [`Decoder::send_packet`] OBU, sequence/frame header ve tile
    /// group ayrıştırmasını ve başlık düzeyi referans durumunu aynen yürütür;
    /// **piksel çözme yapılmaz**: her kare işi `Unsupported("yalnız
    /// ayrıştırma")` çıktısı üretir ve piksel referans yuvaları boşalır.
    /// Başlık dökümü araçları (harness `seqdump`/`framedump`) içindir: superres
    /// ve LR açıldıktan sonra büyük karelerin (ör. 8192×4320 4:4:4) çözümü
    /// başlık doğrulamasına bir şey katmadan dakikalar sürüyordu (S2c.3).
    /// Varsayılan yolun davranışını değiştirmez.
    #[doc(hidden)]
    pub fn set_parse_only(&mut self, on: bool) {
        self.parse_only = on;
    }

    /// Kare başlığı ve araç geçmişi tanılama kayıtlarını (`drain_frame_headers`,
    /// `drain_frame_output_flags`, `pending_frame_headers`, `decoded_tool_counts`)
    /// etkinleştirir veya kapatır (varsayılan: `false`).
    ///
    /// Uzun ömürlü akışlarda bellek büyümesini önlemek için varsayılan olarak
    /// kapalıdır; [`Decoder::last_tool_counts`], [`Decoder::decoded_frames`] ve
    /// [`Decoder::frames_completed`] gibi `O(1)` sayaçlar bu ayardan bağımsız
    /// olarak her zaman güncellenir.
    pub fn set_collect_diagnostics(&mut self, enable: bool) {
        self.collect_diagnostics = enable;
    }

    /// Kare-bazlı tanılama geçmişinin toplanıp toplanmadığını döndürür.
    #[must_use]
    pub const fn collect_diagnostics(&self) -> bool {
        self.collect_diagnostics
    }

    /// Etkin halka içi filtreleri (`DEBLOCK`, `CDEF`, `RESTORATION`) ayarlar.
    pub fn set_inloop_filters(&mut self, filters: InloopFilters) {
        self.inloop_filters = filters;
    }

    /// Etkin halka içi filtre maskesini döndürür.
    #[must_use]
    pub const fn inloop_filters(&self) -> InloopFilters {
        self.inloop_filters
    }

    /// Çözülecek kare türleri filtresini (`All`, `Reference`, `Intra`, `Key`) ayarlar.
    pub fn set_decode_frame_type(&mut self, frame_type: DecodeFrameType) {
        self.decode_frame_type = frame_type;
    }

    /// Geçerli kare türü filtresini döndürür.
    #[must_use]
    pub const fn decode_frame_type(&self) -> DecodeFrameType {
        self.decode_frame_type
    }

    /// `dav1d` `src/obu.c:1531-1543` (`show_existing_frame`) kare türü süzgeci.
    #[inline]
    fn should_skip_show_existing(&self, frame_type: FrameType) -> bool {
        match frame_type {
            FrameType::Inter | FrameType::Switch => {
                matches!(
                    self.decode_frame_type,
                    DecodeFrameType::Intra | DecodeFrameType::Key
                )
            }
            FrameType::Intra => self.decode_frame_type == DecodeFrameType::Key,
            FrameType::Key => false,
        }
    }

    /// `dav1d` `src/obu.c:1641-1657` normal kare türü süzgeci.
    #[inline]
    fn should_skip_frame(&self, frame_type: FrameType, refresh_frame_flags: u8) -> bool {
        match frame_type {
            FrameType::Inter | FrameType::Switch => {
                matches!(
                    self.decode_frame_type,
                    DecodeFrameType::Intra | DecodeFrameType::Key
                ) || (self.decode_frame_type == DecodeFrameType::Reference
                    && refresh_frame_flags == 0)
            }
            FrameType::Intra => {
                self.decode_frame_type == DecodeFrameType::Key
                    || (self.decode_frame_type == DecodeFrameType::Reference
                        && refresh_frame_flags == 0)
            }
            FrameType::Key => false,
        }
    }

    /// Girdi paketini OBU'lara böler ve işler (dav1d `dav1d_send_data` +
    /// `gen_picture`, `src/lib.c:412-457`).
    ///
    /// Paket **bir veya birden çok OBU** içerebilir; `obu_has_size_field = 0`
    /// olan OBU paketin kalanını kaplar. OBU'lar sırayla uygulanır; tamamlanan
    /// kare **hemen** çözülür (dav1d `dav1d_submit_frame`, `n_fc == 1`).
    ///
    /// # Hatalar
    ///
    /// - [`Error::InvalidState`]: [`DecoderState::Draining`] durumunda çağrıldı
    ///   (`flush()` sonrası yeni paket kabul edilmez; önce `reset()`).
    /// - [`Error::Bitstream`] / [`Error::InvalidData`] / [`Error::Unsupported`]:
    ///   bir OBU ayrıştırılamadı ya da tamamlanan kare çözülemedi. dav1d gibi
    ///   paketin **kalanı atılır**, hatadan önceki OBU'ların etkisi (ve
    ///   çıktıları) kalır (`src/lib.c:416-431`).
    pub fn send_packet(&mut self, packet: &Packet<'_>) -> Result<()> {
        self.send_packet_capi(packet, 0)
    }

    /// C-ABI (`dav1d_send_data`) için paket kimlik etiketiyle (`packet_tag`) paket işler.
    pub(crate) fn send_packet_capi(&mut self, packet: &Packet<'_>, packet_tag: u64) -> Result<()> {
        if self.state != DecoderState::Ready {
            return Err(Error::InvalidState(
                "flush sonrası send_packet: önce reset() çağrılmalı",
            ));
        }
        self.parse.current_packet_tag = packet_tag;
        self.parse.current_packet_timestamp = packet.timestamp();
        let data = packet.data();
        let opts = ParseOpts {
            frame_size_limit: self.config.frame_size_limit,
            strict: self.config.strict_std_compliance,
        };
        let mut offset = 0usize;
        let res = loop {
            match split_next_with(data, &mut offset, opts.strict) {
                Ok(Some(obu)) => {
                    if let Err(err) = self.process_obu(&obu, opts, None) {
                        break Err(err);
                    }
                }
                Ok(None) => break Ok(()),
                Err(err) => break Err(err),
            }
        };
        self.sync_shared_kalici();
        res
    }

    /// C-ABI (`gen_picture`) için verilen paket diliminden tam olarak bir OBU ayırıp işler.
    ///
    /// - `Ok(true)`: bir OBU işlendi ve `*offset` ilerletildi.
    /// - `Ok(false)`: dilimde işlenecek başka OBU kalmadı.
    /// - `Err(err)`: OBU ayrıştırma veya kare çözme hatası.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    pub(crate) fn step_one_obu_capi(
        &mut self,
        data: &[u8],
        offset: &mut usize,
        timestamp: Option<i64>,
        packet_tag: u64,
        prealloc: Option<&mut PreallocCb<'_>>,
    ) -> Result<bool> {
        if self.state != DecoderState::Ready {
            return Err(Error::InvalidState(
                "flush sonrası send_packet: önce reset() çağrılmalı",
            ));
        }
        self.parse.current_packet_tag = packet_tag;
        self.parse.current_packet_timestamp = timestamp;
        let opts = ParseOpts {
            frame_size_limit: self.config.frame_size_limit,
            strict: self.config.strict_std_compliance,
        };
        let res = match split_next_with(data, offset, opts.strict) {
            Ok(Some(obu)) => {
                self.process_obu(&obu, opts, prealloc)?;
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(err) => Err(err),
        };
        self.sync_shared_kalici();
        res
    }

    /// Tek OBU (dav1d `dav1d_parse_obus`, `src/obu.c:1169-1694`).
    fn process_obu(
        &mut self,
        obu: &Obu<'_>,
        opts: ParseOpts,
        mut prealloc: Option<&mut PreallocCb<'_>>,
    ) -> Result<()> {
        match obu.header.obu_type {
            ObuType::SequenceHeader => {
                let header = SequenceHeader::parse_with(
                    obu.payload,
                    SequenceHeaderParseOptions {
                        strict_std_compliance: opts.strict,
                    },
                )
                .map_err(|err| rebase_payload_error(err, obu.payload_offset_bits()))?;
                let op_index = header.operating_point_index(self.config.operating_point);
                let idc = header.operating_points[op_index].idc;
                self.parse.operating_point_idc = idc;
                let spatial_mask = idc >> 8;
                self.parse.max_spatial_id = if spatial_mask == 0 {
                    0
                } else {
                    15 - (spatial_mask.leading_zeros() as u8)
                };
                match self.parse.sequence_header.as_ref() {
                    // dav1d `src/obu.c:1234-1236`: ilk dizi başlığı yalnız
                    // bekleyen kareyi temizler (yuvalar `dav1d_flush` /
                    // `reset` ile zaten boştur).
                    None => {
                        self.parse.header_state.clear_current();
                        self.parse.pending_groups.clear();
                        self.parse.dangling_show_existing = None;
                        self.parse.pending_event_flags |= 1;
                    }
                    // dav1d `src/obu.c:1239-1252`: farklı dizi → bekleyen kare
                    // ve referans yuvaları (başlık + piksel) temizlenir.
                    Some(previous) if !previous.same_coded_sequence(&header) => {
                        let mut first_err = None;
                        while !self.in_flight.is_empty() {
                            if let Err(e) = self.retire_front_in_flight(true)
                                && first_err.is_none()
                            {
                                first_err = Some(e);
                            }
                        }
                        self.parse.header_state.reset();
                        self.parse.pending_groups.clear();
                        self.parse.dangling_show_existing = None;
                        self.parse.pending_new_temporal_unit = true;
                        self.parse.content_light = None;
                        self.parse.mastering_display = None;
                        self.parse.pending_event_flags |= 1;
                        self.pixel_refs = [None, None, None, None, None, None, None, None];
                        self.slot_promises = [None, None, None, None, None, None, None, None];
                        self.pool.clear_buf_pool();
                        self.pipeline_butce.taban_guncelle(self.kalici_bayt());
                        self.parse.sequence_header_arc = Some(Arc::new(header.clone()));
                        self.parse.sequence_header = Some(header);
                        self.parse.sequence_header_updates += 1;
                        if let Some(e) = first_err {
                            return Err(e);
                        }
                        return Ok(());
                    }
                    Some(previous) => {
                        if previous.operating_parameter_info != header.operating_parameter_info {
                            self.parse.pending_event_flags |= 2;
                        }
                    }
                }
                self.parse.sequence_header_arc = Some(Arc::new(header.clone()));
                self.parse.sequence_header = Some(header);
                self.parse.sequence_header_updates += 1;
            }
            ObuType::TemporalDelimiter => {
                self.parse.pending_new_temporal_unit = true;
            }
            _ => {
                // dav1d `src/obu.c:1201-1208`: seçili operating point dışındaki
                // katmanın OBU'su sonu denetimi de yapılmadan atlanır.
                if let Some(extension) = obu.header.extension
                    && self.parse.operating_point_idc != 0
                {
                    let idc = self.parse.operating_point_idc;
                    let in_temporal_layer = (idc >> extension.temporal_id) & 1 != 0;
                    let in_spatial_layer = (idc >> (extension.spatial_id + 8)) & 1 != 0;
                    if !in_temporal_layer || !in_spatial_layer {
                        self.parse.layer_obus_skipped += 1;
                        return Ok(());
                    }
                }
                match obu.header.obu_type {
                    ObuType::FrameHeader | ObuType::RedundantFrameHeader | ObuType::Frame => {
                        self.process_frame_header_obu(obu, opts, prealloc.as_deref_mut())?;
                    }
                    ObuType::TileGroup => {
                        self.process_tile_group(
                            obu.payload,
                            obu.payload_offset_bits(),
                            prealloc.as_deref_mut(),
                        )?;
                    }
                    ObuType::Metadata => self.process_metadata_obu(obu, opts.strict)?,
                    _ => {}
                }
            }
        }
        // dav1d `src/obu.c:1527-1548`: bekleyen `show_existing` başlığı her OBU
        // sonunda (yeniden) uygulanır.
        if self.parse.sequence_header.is_some()
            && let Some(slot) = self.parse.dangling_show_existing
        {
            self.show_existing(slot, opts, prealloc)?;
        }
        Ok(())
    }

    /// `parse_frame_hdr()` (+ frame OBU ise tile group), dav1d
    /// `src/obu.c:1264-1321`.
    fn process_frame_header_obu(
        &mut self,
        obu: &Obu<'_>,
        opts: ParseOpts,
        prealloc: Option<&mut PreallocCb<'_>>,
    ) -> Result<()> {
        let is_frame_obu = obu.header.obu_type == ObuType::Frame;
        // dav1d `src/obu.c:1264-1265`: redundant frame header, bekleyen bir
        // başlık (bekleyen `show_existing` dâhil) varsa yok sayılır.
        if obu.header.obu_type == ObuType::RedundantFrameHeader
            && (self.parse.header_state.current().is_some()
                || self.parse.dangling_show_existing.is_some())
        {
            return Ok(());
        }
        let Some(sequence_header) = self.parse.sequence_header.as_ref() else {
            return Err(payload_error(
                obu,
                BitstreamErrorKind::InvalidFrameHeader(
                    "frame header: sequence header yok (obu.c 1266)",
                ),
            ));
        };
        // Yeni başlık önceki bekleyen kareyi geçersiz kılar (dav1d
        // `c->frame_hdr` aynı tampona yeniden ayrıştırılır, `src/obu.c:1279-1282`).
        self.parse.dangling_show_existing = None;
        let extension = obu.header.extension;
        let (temporal_id, spatial_id) =
            extension.map_or((0, 0), |ext| (ext.temporal_id, ext.spatial_id));
        let mut reader = BitReader::new(obu.payload);
        let parsed = {
            let ctx = FrameHeaderContext {
                sequence_header,
                refs: self.parse.header_state.refs(),
                temporal_id,
                spatial_id,
                strict_std_compliance: opts.strict,
            };
            FrameHeader::parse(&mut reader, &ctx)
        };
        let header = match parsed {
            Ok(header) => header,
            Err(err) => {
                // dav1d `src/obu.c:1283-1286`: `c->frame_hdr = NULL`.
                self.parse.header_state.clear_current();
                self.parse.pending_groups.clear();
                return Err(rebase_payload_error(err, obu.payload_offset_bits()));
            }
        };
        self.parse.frame_header_parses += 1;
        // dav1d `src/obu.c:1287-1290`: eski tile verisi atılır.
        self.parse.header_state.clear_current();
        self.parse.pending_groups.clear();
        if !is_frame_obu {
            // dav1d `src/obu.c:1291-1298`: frame header OBU'sunda trailing bits.
            reader
                .read_trailing_bits(opts.strict)
                .map_err(|err| rebase_payload_error(err, obu.payload_offset_bits()))?;
        }
        // dav1d `src/obu.c:1300-1307`: sıfırdan farklı `frame_size_limit`,
        // `width[1] * height` sınırı aşan başlığı reddeder (`ERANGE`).
        if frame_size_exceeds(header.width[1], header.height, opts.frame_size_limit) {
            return Err(payload_error(
                obu,
                BitstreamErrorKind::InvalidFrameHeader(
                    "kare boyutu frame_size_limit sınırını aşıyor (obu.c 1300-1307)",
                ),
            ));
        }
        if !is_frame_obu {
            if header.show_existing_frame {
                // Gösterim OBU sonu denetiminde yapılır (`process_obu`).
                self.parse.dangling_show_existing = Some(usize::from(header.existing_frame_idx));
            } else {
                self.parse.header_state.set_current(header);
            }
            return Ok(());
        }
        // dav1d `src/obu.c:1311-1315`: OBU_FRAME show_existing_frame taşıyamaz.
        if header.show_existing_frame {
            return Err(payload_error(
                obu,
                BitstreamErrorKind::InvalidFrameHeader(
                    "frame OBU show_existing_frame taşıyamaz (obu.c 1311-1315)",
                ),
            ));
        }
        // OBU_FRAME: başlık bekleyen kare olur, bayt hizalanır, kalan baytlar
        // tile grubudur (dav1d `src/obu.c:1317-1321`, fall-through).
        self.parse.header_state.set_current(header);
        reader.align_to_byte();
        let start_byte = reader.byte_position();
        let Some(tile_payload) = obu.payload.get(start_byte..) else {
            return Err(payload_error(
                obu,
                BitstreamErrorKind::InvalidFrameHeader("frame OBU: tile verisi sınırı aşıldı"),
            ));
        };
        self.process_tile_group(
            tile_payload,
            obu.payload_offset_bits() + start_byte * 8,
            prealloc,
        )
    }

    /// `DAV1D_OBU_TILE_GRP` (`src/obu.c:1322-1352`) + kare tamamlanınca çözüm
    /// (`src/obu.c:1640-1666`).
    fn process_tile_group(
        &mut self,
        payload: &[u8],
        bit_base: usize,
        prealloc: Option<&mut PreallocCb<'_>>,
    ) -> Result<()> {
        let Some(current) = self.parse.header_state.current() else {
            return Err(BitstreamError::new(
                BitstreamErrorKind::InvalidTileGroup(
                    "tile group: bekleyen frame header yok (obu.c 1323)",
                ),
                bit_base,
            )
            .into());
        };
        let skip_frame = self.should_skip_frame(current.frame_type, current.refresh_frame_flags);
        let (start, end, data_offset) = tile_group_header(payload, current)
            .map_err(|err| rebase_payload_error(err, bit_base))?;
        // dav1d `src/obu.c:1341-1349`: sıra bozuksa birikim sıfırlanır, bekleyen
        // kare kalır.
        if start > end || usize::from(start) != self.parse.header_state.n_tiles() {
            self.parse.header_state.reset_tiles();
            self.parse.pending_groups.clear();
            return Err(BitstreamError::new(
                BitstreamErrorKind::InvalidTileGroup(
                    "tile group sırası bozuk: start > end ya da start != önceki tile sayısı (obu.c 1341-1349)",
                ),
                bit_base,
            )
            .into());
        }
        if let Ok(info) = parse_tile_group(payload, current) {
            self.parse.tile_bytes_skipped += info.data_bytes as u64;
            self.parse.last_tile_group = Some(info);
        }
        // `decode_frame_type` ile atlanacak karelerde karo verisi kopyalanmaz;
        // yine de başlık ve karo grubu sınırları tam doğrulanır (`dav1d` `obu.c:1640-1688`).
        if !skip_frame {
            // P38: tile verisinin kopyası bütçe defterinden (düşebilir); aşımda
            // grup eklenmez, bekleyen kare tamamlanmaz.
            let butce = self.butce_defteri(0);
            let data = butce.kopya(&payload[data_offset..]);
            self.tepe_kat(&butce);
            self.parse.pending_groups.push(TileGroupRaw {
                start,
                end,
                data: data?,
            });
        }
        self.parse
            .header_state
            .add_tiles(usize::from(end - start) + 1);
        self.parse.tile_groups_processed += 1;

        let expected = usize::from(current_tiles(&self.parse.header_state));
        if expected == 0 || self.parse.header_state.n_tiles() != expected {
            return Ok(());
        }
        let prev_slots = self.parse.header_state.slots();
        let Some(header) = self.parse.header_state.complete_frame() else {
            return Ok(());
        };
        if self.should_skip_frame(header.frame_type, header.refresh_frame_flags) {
            // `dav1d` `src/obu.c:1641-1688`: atlanan karenin referans başlık yuvaları
            // (`header_state`) güncellenmiş olarak kalır, karşılık gelen piksel/vaat
            // yuvaları ise temizlenir; böylece sonraki `show_existing_frame` başlıkları
            // gerçek kare türünü görür ve yanlışlıkla eski KEY karesini tekrarlamaz.
            for slot in 0..8 {
                if header.refresh_frame_flags & (1 << slot) != 0 {
                    self.pixel_refs[slot] = None;
                    self.slot_promises[slot] = None;
                }
            }
            self.parse.pending_groups.clear();
            return Ok(());
        }
        if self.collect_diagnostics {
            self.frame_outputs.push(header.clone());
            self.frame_output_flags.push(header.show_frame);
        }
        self.parse.frames_completed += 1;
        let groups = core::mem::take(&mut self.parse.pending_groups);
        let seq = self.parse.sequence_header.clone();
        self.decode_completed(header, &groups, seq.as_ref(), &prev_slots, prealloc)
    }

    /// Tamamlanan kareyi çözer (`n_fc == 1` senkron hızlı yol veya `n_fc >= 2`
    /// kareler arası asenkron boru hattı; dav1d `dav1d_submit_frame` + `dav1d_decode_frame`).
    ///
    /// - Erken hata (eksik/uyumsuz referans, `src/decode.c:3452-3478`): yuvalar
    ///   **değişmez** — başlık yuvaları anlık görüntüden geri yüklenir.
    /// - Geç hata (tile boyutu `src/decode.c:3163-3174`, tile çözme
    ///   `src/decode.c:2749-2750`): yenilenen yuvalar boşaltılır
    ///   (`src/decode.c:3704-3716`).
    fn decode_completed(
        &mut self,
        header: FrameHeader,
        groups: &[TileGroupRaw],
        seq: Option<&SequenceHeader>,
        prev_slots: &RefSlots,
        prealloc: Option<&mut PreallocCb<'_>>,
    ) -> Result<()> {
        if self.parse_only {
            while !self.in_flight.is_empty() {
                let _ = self.retire_front_in_flight(true);
            }
            self.unsupported_frames += 1;
            for (slot, reference) in self.pixel_refs.iter_mut().enumerate() {
                if header.refresh_frame_flags & (1 << slot) != 0 {
                    *reference = None;
                }
            }
            for (slot, promise) in self.slot_promises.iter_mut().enumerate() {
                if header.refresh_frame_flags & (1 << slot) != 0 {
                    *promise = None;
                }
            }
            self.outputs
                .push_back(OutputEntry::Unsupported("yalnız ayrıştırma"));
            self.outputs_pushed += 1;
            self.output_errors
                .push_back(Some("yalnız ayrıştırma kipi".into()));
            self.output_tools.push_back(None);
            self.output_capi_meta.push_back(OutputCapiMeta::default());
            return Ok(());
        }
        let Some(seq) = seq else {
            return Err(Error::Unsupported("sequence header yok"));
        };
        if self.config.effective_frame_delay() > 1 {
            return self.decode_completed_pipeline(header, groups, seq, prev_slots, prealloc);
        }
        let early = frame_format(seq).and_then(|(chroma, bd)| {
            validate_frame_refs(&header, chroma, bd, &self.pixel_refs)?;
            Ok((chroma, bd))
        });
        let (early_chroma, early_bd) = match early {
            Ok(pair) => pair,
            Err(err) => {
                self.parse
                    .header_state
                    .restore_slots(prev_slots, header.refresh_frame_flags);
                self.unsupported_frames += 1;
                self.last_decode_error = Some(format!("{err}"));
                self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                return Err(err);
            }
        };
        let want_output = header.show_frame || self.config.output_invisible_frames;
        let prealloc_id = if want_output && let Some(cb) = prealloc {
            let req = PreallocRequest {
                width: header.width[1] as usize,
                height: header.height as usize,
                chroma: early_chroma,
                bit_depth: early_bd,
                seq_hdr: Some(seq),
                frame_hdr: &header,
                content_light: self.parse.content_light,
                mastering_display: self.parse.mastering_display,
                itut_t35: &self.parse.itut_t35,
            };
            match cb(&req) {
                Ok(id) => id,
                Err(err) => {
                    self.parse
                        .header_state
                        .restore_slots(prev_slots, header.refresh_frame_flags);
                    self.unsupported_frames += 1;
                    self.last_decode_error = Some(format!("{err}"));
                    self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                    return Err(err);
                }
            }
        } else {
            0
        };
        #[cfg(feature = "msac-trace")]
        {
            crate::trace::frame(
                u32::from(header.frame_type.raw()),
                u32::try_from(header.width[0]).unwrap_or(0),
                u32::try_from(header.height).unwrap_or(0),
                u32::from(seq.sb128),
            );
        }
        // P38: kare içi ayırmalar bütçe defterinden; taban, kare dışındaki
        // sayılan ayırmalar + bu karenin tile verisi (`groups`).
        let grup_bayti: usize = groups.iter().map(|g| g.data.len()).sum();
        let butce = self.butce_defteri(grup_bayti);
        let inloop_filters = self.inloop_filters;
        let decoded = split_tile_groups(groups, &header).and_then(|tiles| {
            decode_frame(
                seq,
                &header,
                &tiles,
                &self.dsp,
                &self.pixel_refs,
                &butce,
                &self.pool,
                inloop_filters,
            )
        });
        self.tepe_kat(&butce);
        let mc_identity = seq.mtrx == 0;
        let has_limit = self.config.allocation_limit.is_some();

        // `4.A.1`: Referans olarak saklanmayan (`refresh_frame_flags == 0`) karelerde
        // ikincil `Frame` veya `RefFrame` ayırmadan `decoded.pic` doğrudan yerinde
        // (`in-place`) çıktı `Frame`'ine dönüştürülür ve şerit bazlı Film Grain uygulanır.
        if header.refresh_frame_flags == 0 {
            let decoded = match decoded {
                Ok(d) => d,
                Err(err) => {
                    self.unsupported_frames += 1;
                    self.last_decode_error = Some(format!("{err}"));
                    self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                    return Err(err);
                }
            };
            let decoded_tools = decoded.tools;
            let (pic, chroma, bd, segmap, tmvs) = (
                decoded.pic,
                decoded.chroma,
                decoded.bd,
                decoded.segmap,
                decoded.tmvs,
            );
            self.pool
                .buf_pool()
                .recycle_segmap_tmvs(segmap, tmvs, has_limit);
            let cikti = if want_output {
                let out_butce = self.butce_defteri(grup_bayti);
                let mut tools = decoded_tools;
                let frame_res = build_output_frame_in_place(
                    self.config.apply_grain,
                    &self.dsp,
                    &self.pool,
                    pic,
                    chroma,
                    bd,
                    &header.film_grain,
                    mc_identity,
                    self.outputs_pushed,
                    &mut tools,
                    &out_butce,
                );
                self.tepe_kat(&out_butce);
                match frame_res {
                    Ok(frame) => Some((frame, tools)),
                    Err(err) => {
                        self.unsupported_frames += 1;
                        self.last_decode_error = Some(format!("{err}"));
                        self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                        return Err(err);
                    }
                }
            } else {
                self.pool.buf_pool().recycle_pic(pic, has_limit);
                None
            };
            self.decoded_frames += 1;
            self.last_tools = Some(decoded_tools);
            if self.collect_diagnostics {
                self.decoded_tools.push(decoded_tools);
            }
            let itut_t35 = core::mem::take(&mut self.parse.itut_t35);
            if let Some((mut frame, tools)) = cikti {
                if self.parse.current_packet_timestamp.is_some() {
                    frame.metadata_mut().timestamp = self.parse.current_packet_timestamp;
                }
                let capi_meta = OutputCapiMeta {
                    seq_hdr: self.parse.sequence_header_arc.clone(),
                    frame_hdr: Some(Arc::new(header)),
                    content_light: self.parse.content_light,
                    mastering_display: self.parse.mastering_display,
                    itut_t35,
                    event_flags: core::mem::take(&mut self.parse.pending_event_flags),
                    packet_tag: self.parse.current_packet_tag,
                    new_temporal_unit: core::mem::take(&mut self.parse.pending_new_temporal_unit),
                    prealloc_id,
                };
                self.outputs.push_back(OutputEntry::Frame(frame));
                self.outputs_pushed += 1;
                self.output_errors.push_back(None);
                self.output_tools.push_back(Some(tools));
                self.output_capi_meta.push_back(capi_meta);
            }
            return Ok(());
        }

        // Çıktı kopyası (grain'li ya da düz) yuvalar değişmeden **önce**
        // ayrılır: bütçe/ayırma hatası geç hata gibi işlenir, yarım kare
        // yuvaya ya da çıktıya girmez.
        let decoded = decoded.and_then(|mut decoded| {
            if !want_output {
                return Ok((decoded, None));
            }
            let out_butce = self.butce_defteri(grup_bayti.saturating_add(decoded.bayt()));
            let mut tools = decoded.tools;
            let frame_res = build_output_frame_from_mut_pic(
                self.config.apply_grain,
                &self.dsp,
                &self.pool,
                &mut decoded.pic,
                decoded.chroma,
                decoded.bd,
                &header.film_grain,
                mc_identity,
                self.outputs_pushed,
                &mut tools,
                &out_butce,
            );
            self.tepe_kat(&out_butce);
            Ok((decoded, Some((frame_res?, tools))))
        });
        let (decoded, cikti) = match decoded {
            Ok(decoded) => decoded,
            Err(err) => {
                self.parse
                    .header_state
                    .clear_slots(header.refresh_frame_flags);
                for (slot, reference) in self.pixel_refs.iter_mut().enumerate() {
                    if header.refresh_frame_flags & (1 << slot) != 0 {
                        *reference = None;
                    }
                }
                self.unsupported_frames += 1;
                self.last_decode_error = Some(format!("{err}"));
                self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                return Err(err);
            }
        };
        self.decoded_frames += 1;
        self.last_tools = Some(decoded.tools);
        if self.collect_diagnostics {
            self.decoded_tools.push(decoded.tools);
        }
        // Referans yuvası grain'siz `PicBuf`'u saklar (`src/lib.c:311-330`).
        let entry = Arc::new(RefFrame::from_decoded(decoded, header.clone(), mc_identity));
        let mut evicted: [Option<Arc<RefFrame>>; 8] = [const { None }; 8];
        for (slot, reference) in self.pixel_refs.iter_mut().enumerate() {
            if header.refresh_frame_flags & (1 << slot) != 0
                && let Some(old) = reference.replace(Arc::clone(&entry))
            {
                evicted[slot] = Some(old);
            }
        }
        if let Ok(rf) = Arc::try_unwrap(entry) {
            self.pool.buf_pool().recycle_ref_frame(rf, has_limit);
        }
        for old in evicted.into_iter().flatten() {
            if let Ok(rf) = Arc::try_unwrap(old) {
                self.pool.buf_pool().recycle_ref_frame(rf, has_limit);
            }
        }
        let itut_t35 = core::mem::take(&mut self.parse.itut_t35);
        // dav1d `src/decode.c:3551`: görünmez kare (`show_frame = 0`) yalnız
        // `output_invisible_frames` ile çıktıdır (çıktı kopyası yukarıda).
        if let Some((mut frame, tools)) = cikti {
            if self.parse.current_packet_timestamp.is_some() {
                frame.metadata_mut().timestamp = self.parse.current_packet_timestamp;
            }
            let capi_meta = OutputCapiMeta {
                seq_hdr: self.parse.sequence_header_arc.clone(),
                frame_hdr: Some(Arc::new(header)),
                content_light: self.parse.content_light,
                mastering_display: self.parse.mastering_display,
                itut_t35,
                event_flags: core::mem::take(&mut self.parse.pending_event_flags),
                packet_tag: self.parse.current_packet_tag,
                new_temporal_unit: core::mem::take(&mut self.parse.pending_new_temporal_unit),
                prealloc_id,
            };
            self.outputs.push_back(OutputEntry::Frame(frame));
            self.outputs_pushed += 1;
            self.output_errors.push_back(None);
            self.output_tools.push_back(Some(tools));
            self.output_capi_meta.push_back(capi_meta);
        }
        Ok(())
    }

    /// `n_fc >= 2` (`Config::effective_frame_delay() > 1`) için kareler arası
    /// bağımlılık grafiği (`FrameSlot` vaatleri) ve gecikmeli Film Grain örtüşmeli
    /// boru hattı gönderimi (`ADR-0004` §2.3).
    fn decode_completed_pipeline(
        &mut self,
        header: FrameHeader,
        groups: &[TileGroupRaw],
        seq: &SequenceHeader,
        prev_slots: &RefSlots,
        prealloc: Option<&mut PreallocCb<'_>>,
    ) -> Result<()> {
        let n_fc = self.config.effective_frame_delay();
        // 1. Zaten tamamlanmış ön uçuş görevlerini bloklamadan emekliye ayır.
        while self
            .in_flight
            .front()
            .is_some_and(InFlightEntry::is_done_nonblocking)
        {
            self.retire_front_in_flight(false)?;
        }
        // 2. Uçuş kuyruğundaki aktif çözme görevi sayısı `n_fc` sınırındaysa
        //    en eski uçuş karesini bekleyip emekliye ayır.
        while self.in_flight.iter().filter(|e| e.is_decode()).count() >= n_fc {
            self.retire_front_in_flight(true)?;
        }
        // 3. Eğer bu karenin başvurduğu yuvalardan biri arka planda hata vermiş
        //    bir uçuş karesine aitse, erken doğrulamadan önce o kareyi emekliye
        //    ayır ki geç hata durumu deterministik sırada işlensin.
        if matches!(header.frame_type, FrameType::Inter | FrameType::Switch) {
            let mut has_failed_dep = false;
            for &ridx in &header.refidx {
                let slot = usize::from(ridx as u8);
                if self
                    .slot_promises
                    .get(slot)
                    .and_then(Option::as_ref)
                    .is_some_and(|s| s.is_failed_nonblocking())
                {
                    has_failed_dep = true;
                    break;
                }
            }
            if has_failed_dep {
                while !self.in_flight.is_empty() {
                    self.retire_front_in_flight(true)?;
                }
            }
        }

        // 4. Erken doğrulama (`validate_frame_refs_slots`): hata durumunda yuvalar değişmez.
        let early = frame_format(seq).and_then(|(chroma, bd)| {
            validate_frame_refs_slots(&header, chroma, bd, &self.slot_promises)?;
            Ok((chroma, bd))
        });
        let (chroma, bd) = match early {
            Ok(pair) => pair,
            Err(err) => {
                self.parse
                    .header_state
                    .restore_slots(prev_slots, header.refresh_frame_flags);
                self.unsupported_frames += 1;
                self.last_decode_error = Some(format!("{err}"));
                self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                return Err(err);
            }
        };

        let want_output = header.show_frame || self.config.output_invisible_frames;
        let prealloc_id = if want_output && let Some(cb) = prealloc {
            let req = PreallocRequest {
                width: header.width[1] as usize,
                height: header.height as usize,
                chroma,
                bit_depth: bd,
                seq_hdr: Some(seq),
                frame_hdr: &header,
                content_light: self.parse.content_light,
                mastering_display: self.parse.mastering_display,
                itut_t35: &self.parse.itut_t35,
            };
            match cb(&req) {
                Ok(id) => id,
                Err(err) => {
                    self.parse
                        .header_state
                        .restore_slots(prev_slots, header.refresh_frame_flags);
                    self.unsupported_frames += 1;
                    self.last_decode_error = Some(format!("{err}"));
                    self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                    return Err(err);
                }
            }
        } else {
            0
        };

        // 5. Paylaşılan bütçe tabanını senkronla, tile grup baytlarını ve tile
        //    boyut sınırlarını (`split_tile_groups`) doğrula.
        self.pipeline_butce.taban_guncelle(self.kalici_bayt());
        let grup_bayti: usize = groups.iter().map(|g| g.data.len()).sum();
        let grup_tutucu = match Butce::tut(&self.pipeline_butce, grup_bayti) {
            Ok(t) => t,
            Err(err) => {
                self.allocation_peak = self.allocation_peak.max(self.pipeline_butce.tepe());
                self.clear_slots_late_error(header.refresh_frame_flags);
                self.unsupported_frames += 1;
                self.last_decode_error = Some(format!("{err}"));
                self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                return Err(err);
            }
        };
        if let Err(err) = split_tile_groups(groups, &header) {
            drop(grup_tutucu);
            self.allocation_peak = self.allocation_peak.max(self.pipeline_butce.tepe());
            self.clear_slots_late_error(header.refresh_frame_flags);
            self.unsupported_frames += 1;
            self.last_decode_error = Some(format!("{err}"));
            self.last_error_packet_tag = Some(self.parse.current_packet_tag);
            return Err(err);
        }

        // 6. Girdi referans yuvası vaatlerini yakala ve bu karenin yenilediği
        //    yuvalara yeni `Pending` `FrameSlot` vaadini yerleştir.
        let in_slots = self.slot_promises.clone();
        let out_slot =
            FrameSlot::pending(header.width[1] as usize, header.height as usize, chroma, bd);
        for (slot, promise) in self.slot_promises.iter_mut().enumerate() {
            if header.refresh_frame_flags & (1 << slot) != 0 {
                *promise = Some(Arc::clone(&out_slot));
            }
        }

        let output_seq = if want_output {
            let s = self.outputs_pushed;
            self.outputs_pushed += 1;
            s
        } else {
            0
        };
        let itut_t35 = core::mem::take(&mut self.parse.itut_t35);
        let packet_tag = self.parse.current_packet_tag;
        let packet_timestamp = self.parse.current_packet_timestamp;
        let capi_meta_w = if want_output {
            Some(OutputCapiMeta {
                seq_hdr: self.parse.sequence_header_arc.clone(),
                frame_hdr: Some(Arc::new(header.clone())),
                content_light: self.parse.content_light,
                mastering_display: self.parse.mastering_display,
                itut_t35,
                event_flags: core::mem::take(&mut self.parse.pending_event_flags),
                packet_tag,
                new_temporal_unit: core::mem::take(&mut self.parse.pending_new_temporal_unit),
                prealloc_id,
            })
        } else {
            None
        };

        let task = Arc::new(InFlightDecodeTask {
            refresh_frame_flags: header.refresh_frame_flags,
            packet_tag,
            prealloc_id,
            out_slot: Arc::clone(&out_slot),
            started: AtomicBool::new(false),
            result: Mutex::new(None),
            done_cv: Condvar::new(),
        });
        self.in_flight
            .push_back(InFlightEntry::Decode(Arc::clone(&task)));

        let task_worker = Arc::clone(&task);
        let seq_w = seq.clone();
        let hdr_w = header;
        let groups_w = groups.to_vec();
        let dsp_w = self.dsp;
        let pool_w = self.pool.handle();
        let pipeline_butce_w = Arc::clone(&self.pipeline_butce);
        let cancel_worker = Arc::clone(&self.cancel_flag);
        let apply_grain_flag = self.config.apply_grain;
        let has_allocation_limit = self.config.allocation_limit.is_some();
        let inloop_filters_w = self.inloop_filters;

        self.pool.spawn_async(move || {
            task_worker.started.store(true, Ordering::Release);
            let body = || -> Result<CompletedFrame> {
                if cancel_worker.load(Ordering::Acquire) {
                    return Err(Error::InvalidState("iptal edildi"));
                }
                let mut resolved_refs: [Option<Arc<RefFrame>>; 8] =
                    [None, None, None, None, None, None, None, None];
                let is_inter = matches!(hdr_w.frame_type, FrameType::Inter | FrameType::Switch);
                if is_inter {
                    for &ridx in &hdr_w.refidx {
                        let slot = usize::from(ridx as u8);
                        if resolved_refs[slot].is_none() {
                            let Some(promise) = in_slots[slot].as_ref() else {
                                return Err(Error::Unsupported(
                                    "referans çözülmedi: inter referans",
                                ));
                            };
                            let Some(rf) = promise.wait(&cancel_worker) else {
                                return Err(Error::Unsupported(
                                    "referans çözülmedi: inter referans",
                                ));
                            };
                            resolved_refs[slot] = Some(rf);
                        }
                    }
                }
                if hdr_w.primary_ref_frame != 7 {
                    let pr = usize::from(hdr_w.primary_ref_frame);
                    let slot = usize::from(hdr_w.refidx[pr] as u8);
                    if resolved_refs[slot].is_none() {
                        let Some(promise) = in_slots[slot].as_ref() else {
                            return Err(Error::Unsupported(
                                "referans çözülmedi: primary_ref_frame",
                            ));
                        };
                        let Some(rf) = promise.wait(&cancel_worker) else {
                            return Err(Error::Unsupported(
                                "referans çözülmedi: primary_ref_frame",
                            ));
                        };
                        resolved_refs[slot] = Some(rf);
                    }
                }
                drop(in_slots);
                if cancel_worker.load(Ordering::Acquire) {
                    return Err(Error::InvalidState("iptal edildi"));
                }

                let tiles = split_tile_groups(&groups_w, &hdr_w)?;
                let frame_butce = Butce::alt_defter(&pipeline_butce_w);
                let decoded = decode_frame(
                    &seq_w,
                    &hdr_w,
                    &tiles,
                    &dsp_w,
                    &resolved_refs,
                    &frame_butce,
                    &pool_w,
                    inloop_filters_w,
                )?;
                for r in resolved_refs.into_iter().flatten() {
                    if let Ok(rf) = Arc::try_unwrap(r) {
                        pool_w
                            .buf_pool()
                            .recycle_ref_frame(rf, has_allocation_limit);
                    }
                }
                let decoded_tools = decoded.tools;
                let mc_identity = seq_w.mtrx == 0;
                let mut tutucular = Vec::with_capacity(3);
                tutucular.push(grup_tutucu);

                // `4.A.1`: Referans olarak saklanmayan (`refresh_frame_flags == 0`) boru hattı
                // karelerinde `RefFrame` tutulmaz; `segmap`/`tmvs` anında havuza iade edilir
                // ve `decoded.pic` yerinde (`in-place`) çıktı `Frame`'ine dönüştürülür.
                if hdr_w.refresh_frame_flags == 0 {
                    let (pic, chroma, bd, segmap, tmvs) = (
                        decoded.pic,
                        decoded.chroma,
                        decoded.bd,
                        decoded.segmap,
                        decoded.tmvs,
                    );
                    pool_w
                        .buf_pool()
                        .recycle_segmap_tmvs(segmap, tmvs, has_allocation_limit);
                    let output = if let Some(capi_meta) = capi_meta_w {
                        let (_, frame_total) =
                            Frame::plane_byte_lens_and_total(pic.width, pic.height, chroma, bd)?;
                        let frame_tutucu = frame_butce.devret(frame_total);
                        drop(frame_butce);
                        let mut frame = Frame::from_owned_pic(
                            pic.width,
                            pic.height,
                            chroma,
                            bd,
                            pic.data,
                            pic.stride,
                            !has_allocation_limit,
                        )?;
                        let mut tools = decoded_tools;
                        if apply_grain_flag
                            && crate::decode::filmgrain::has_grain(&hdr_w.film_grain.data)
                        {
                            fill_fg_tools(&hdr_w.film_grain, &mut tools);
                            let fg_butce = Butce::alt_defter(&pipeline_butce_w);
                            let fg_pool = pool_w.single_thread_handle();
                            crate::dsp::profil_bd!(bd);
                            crate::decode::filmgrain::apply_grain_in_place(
                                &mut frame,
                                &hdr_w.film_grain.data,
                                mc_identity,
                                dsp_w.for_bit_depth(bd).fg(),
                                crate::decode::filmgrain::trace_picture(output_seq),
                                &fg_butce,
                                &fg_pool,
                            )?;
                        }
                        if packet_timestamp.is_some() {
                            frame.metadata_mut().timestamp = packet_timestamp;
                        }
                        if let Some(t) = frame_tutucu {
                            tutucular.push(t);
                        }
                        Some((frame, tools, capi_meta))
                    } else {
                        drop(frame_butce);
                        pool_w.buf_pool().recycle_pic(pic, has_allocation_limit);
                        None
                    };
                    return Ok(CompletedFrame {
                        ref_frame: None,
                        decoded_tools,
                        output,
                        _butce_tutucu: tutucular,
                    });
                }

                let decoded_bytes = decoded.bayt();
                let ref_tutucu = frame_butce.devret(decoded_bytes);
                drop(frame_butce);

                let ref_frame =
                    Arc::new(RefFrame::from_decoded(decoded, hdr_w.clone(), mc_identity));

                // Gecikmeli Film Grain örtüşmesi (`dav1d` `DAV1D_TASK_TYPE_FG_APPLY`):
                // Loop Restoration biter bitmez grain'siz `RefFrame` anında `out_slot`'a
                // yayınlanır; bağımlı sonraki kare bu karenin `apply_grain` adımıyla
                // eşzamanlı olarak çözmeye başlar.
                if !has_allocation_limit {
                    task_worker.out_slot.publish_ready(Arc::clone(&ref_frame));
                }

                if let Some(t) = ref_tutucu {
                    tutucular.push(t);
                }

                let output = if let Some(capi_meta) = capi_meta_w {
                    let out_butce = Butce::alt_defter(&pipeline_butce_w);
                    let mut tools = decoded_tools;
                    // `n_fc >= 2` boru hattında `out_slot.publish_ready` sonrası
                    // sonraki kareler diğer işçilerde eşzamanlı çözülmeye başladığından,
                    // bu karenin gecikmeli Film Grain adımı işçi üzerinde doğrudan
                    // `single_thread_handle` ile çekişmesiz yürütülür.
                    let fg_pool = pool_w.single_thread_handle();
                    let mut frame = build_output_frame_from_ref(
                        apply_grain_flag,
                        &dsp_w,
                        &fg_pool,
                        &ref_frame,
                        output_seq,
                        &mut tools,
                        &out_butce,
                    )?;
                    if packet_timestamp.is_some() {
                        frame.metadata_mut().timestamp = packet_timestamp;
                    }
                    if let Some(t) = out_butce.devret(frame.allocation_bytes()) {
                        tutucular.push(t);
                    }
                    drop(out_butce);
                    Some((frame, tools, capi_meta))
                } else {
                    None
                };

                if has_allocation_limit {
                    task_worker.out_slot.publish_ready(Arc::clone(&ref_frame));
                }

                Ok(CompletedFrame {
                    ref_frame: Some(ref_frame),
                    decoded_tools,
                    output,
                    _butce_tutucu: tutucular,
                })
            };
            #[cfg(feature = "std")]
            let run_res: Result<CompletedFrame> =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
                    Ok(res) => res,
                    Err(_) => Err(Error::InvalidState("asenkron kare çözme görevi panikledi")),
                };
            #[cfg(not(feature = "std"))]
            let run_res: Result<CompletedFrame> = body();

            if run_res.is_err() {
                task_worker.out_slot.mark_failed();
            }
            let mut guard = task_worker.result.lock();
            *guard = Some(run_res);
            task_worker.done_cv.notify_all();
        });

        Ok(())
    }

    fn clear_slots_late_error(&mut self, refresh_frame_flags: u8) {
        self.parse.header_state.clear_slots(refresh_frame_flags);
        for slot in 0..8 {
            if refresh_frame_flags & (1 << slot) != 0 {
                self.pixel_refs[slot] = None;
                self.slot_promises[slot] = None;
            }
        }
    }

    /// Uçuş kuyruğunun (`in_flight`) başındaki tek bir öğeyi emekliye ayırır.
    ///
    /// - `block == false`: başındaki görev henüz tamamlanmamışsa beklemez ve
    ///   `Ok(false)` döner.
    /// - `block == true`: başındaki görev tamamlanana kadar bekler (henüz
    ///   başlamamışsa kuyruktan çekip bizzat çalıştırır veya `active_batches`
    ///   alt-partilerine yardım eder) ve emekliye ayırıp `Ok(true)` döner.
    fn retire_front_in_flight(&mut self, block: bool) -> Result<bool> {
        let Some(front) = self.in_flight.front() else {
            return Ok(false);
        };
        if !block && !front.is_done_nonblocking() {
            return Ok(false);
        }
        let Some(entry) = self.in_flight.pop_front() else {
            return Ok(false);
        };
        match entry {
            InFlightEntry::ShowExisting {
                promise,
                output_seq,
                refresh_all_key,
                capi_meta,
                packet_timestamp,
            } => {
                while !promise.is_done_nonblocking() {
                    if let Some(work) = self.pool.try_pop_async_task() {
                        work();
                    } else if !self.pool.help_active_batch() {
                        break;
                    }
                }
                let Some(entry_rf) = promise.wait(&self.cancel_flag) else {
                    self.last_error_packet_tag = Some(capi_meta.packet_tag);
                    return Err(BitstreamError::new(
                        BitstreamErrorKind::InvalidFrameHeader(
                            "show_existing_frame: referans resmi yok (obu.c 1544)",
                        ),
                        0,
                    )
                    .into());
                };
                self.pipeline_butce.taban_guncelle(self.kalici_bayt());
                let out_butce = Butce::alt_defter(&self.pipeline_butce);
                let mut tools = ToolCounts::default();
                let fg_pool = self.pool.single_thread_handle();
                let frame_res = build_output_frame_from_ref(
                    self.config.apply_grain,
                    &self.dsp,
                    &fg_pool,
                    &entry_rf,
                    output_seq,
                    &mut tools,
                    &out_butce,
                );
                self.allocation_peak = self.allocation_peak.max(self.pipeline_butce.tepe());
                let mut frame = frame_res.inspect_err(|_| {
                    self.last_error_packet_tag = Some(capi_meta.packet_tag);
                })?;
                if packet_timestamp.is_some() {
                    frame.metadata_mut().timestamp = packet_timestamp;
                }
                drop(out_butce);
                if refresh_all_key {
                    for reference in &mut self.pixel_refs {
                        *reference = Some(Arc::clone(&entry_rf));
                    }
                }
                self.outputs.push_back(OutputEntry::Frame(frame));
                self.output_errors.push_back(None);
                self.output_tools.push_back(Some(tools));
                self.output_capi_meta.push_back(capi_meta);
                self.pipeline_butce.taban_guncelle(self.kalici_bayt());
                Ok(true)
            }
            InFlightEntry::Decode(task) => {
                // Görev henüz bir işçi tarafından başlatılmamışsa FIFO kuyruğundan
                // çekip doğrudan ana iş parçacığında çalıştır.
                while !task.started.load(Ordering::Acquire) && !task.is_done_nonblocking() {
                    if let Some(work) = self.pool.try_pop_async_task() {
                        work();
                    } else {
                        break;
                    }
                }
                // Görev çalışıyorsa varsa alt-partilere yardım et, ardından `done_cv` üzerinde bekle.
                while !task.is_done_nonblocking() {
                    if !self.pool.help_active_batch() {
                        let mut guard = task.result.lock();
                        while guard.is_none() {
                            guard = task.done_cv.wait(guard);
                        }
                        break;
                    }
                }
                let res = task
                    .result
                    .lock()
                    .take()
                    .unwrap_or(Err(Error::InvalidState("uçuş görevi sonuç üretmedi")));
                self.allocation_peak = self.allocation_peak.max(self.pipeline_butce.tepe());
                match res {
                    Ok(completed) => {
                        let has_limit = self.config.allocation_limit.is_some();
                        self.decoded_frames += 1;
                        self.last_tools = Some(completed.decoded_tools);
                        if self.collect_diagnostics {
                            self.decoded_tools.push(completed.decoded_tools);
                        }
                        let mut evicted: [Option<Arc<RefFrame>>; 8] = [const { None }; 8];
                        if let Some(rf_arc) = completed.ref_frame {
                            for (slot, reference) in self.pixel_refs.iter_mut().enumerate() {
                                if task.refresh_frame_flags & (1 << slot) != 0
                                    && self.slot_promises[slot].is_some()
                                    && let Some(old) = reference.replace(Arc::clone(&rf_arc))
                                {
                                    evicted[slot] = Some(old);
                                }
                            }
                            if task.refresh_frame_flags == 0 {
                                task.out_slot.clear_ready();
                            }
                            if let Ok(rf) = Arc::try_unwrap(rf_arc) {
                                self.pool.buf_pool().recycle_ref_frame(rf, has_limit);
                            }
                        } else {
                            task.out_slot.clear_ready();
                        }
                        for old in evicted.into_iter().flatten() {
                            if let Ok(rf) = Arc::try_unwrap(old) {
                                self.pool.buf_pool().recycle_ref_frame(rf, has_limit);
                            }
                        }
                        if let Some((frame, tools, capi_meta)) = completed.output {
                            self.outputs.push_back(OutputEntry::Frame(frame));
                            self.output_errors.push_back(None);
                            self.output_tools.push_back(Some(tools));
                            self.output_capi_meta.push_back(capi_meta);
                        }
                        drop(completed._butce_tutucu);
                        self.pipeline_butce.taban_guncelle(self.kalici_bayt());
                        self.allocation_peak = self.allocation_peak.max(self.pipeline_butce.tepe());
                        Ok(true)
                    }
                    Err(err) => {
                        for slot in 0..8 {
                            if task.refresh_frame_flags & (1 << slot) != 0 {
                                self.pixel_refs[slot] = None;
                                if self.slot_promises[slot]
                                    .as_ref()
                                    .is_some_and(|s| Arc::ptr_eq(s, &task.out_slot))
                                {
                                    self.slot_promises[slot] = None;
                                    self.parse.header_state.clear_slots(1 << slot);
                                }
                            }
                        }
                        self.unsupported_frames += 1;
                        self.last_decode_error = Some(format!("{err}"));
                        self.last_error_packet_tag = Some(task.packet_tag);
                        self.pipeline_butce.taban_guncelle(self.kalici_bayt());
                        Err(err)
                    }
                }
            }
        }
    }

    /// `reset()` ve `drop()` için uçuştaki tüm görevleri iptal edip temizce boşaltır.
    fn cancel_and_drain_in_flight(&mut self) {
        if self.in_flight.is_empty() {
            return;
        }
        self.cancel_flag.store(true, Ordering::Release);
        for promise in self.slot_promises.iter().flatten() {
            promise.mark_failed();
        }
        for entry in &self.in_flight {
            if let InFlightEntry::Decode(task) = entry {
                task.out_slot.mark_failed();
            }
        }
        while let Some(work) = self.pool.try_pop_async_task() {
            work();
        }
        while let Some(entry) = self.in_flight.pop_front() {
            if let InFlightEntry::Decode(task) = entry {
                let mut guard = task.result.lock();
                while guard.is_none() {
                    guard = task.done_cv.wait(guard);
                }
            }
        }
        self.allocation_peak = self.allocation_peak.max(self.pipeline_butce.tepe());
        self.cancel_flag = Arc::new(AtomicBool::new(false));
    }

    /// `show_existing_frame` (dav1d `src/obu.c:1527-1641`).
    ///
    /// Yuva başlığı ya da resmi yoksa (ya da katı uyumda gösterilemiyorsa)
    /// hata döner ve başlık **bekler** (`dangling_show_existing`).
    fn show_existing(
        &mut self,
        slot: usize,
        opts: ParseOpts,
        prealloc: Option<&mut PreallocCb<'_>>,
    ) -> Result<()> {
        if self.config.effective_frame_delay() > 1 && !self.parse_only {
            return self.show_existing_pipeline(slot, opts, prealloc);
        }
        let fail = |what: &'static str| -> Result<()> {
            Err(BitstreamError::new(BitstreamErrorKind::InvalidFrameHeader(what), 0).into())
        };
        let Some(ref_hdr) = self.parse.header_state.reference(slot) else {
            return fail("show_existing_frame: referans yuva boş (obu.c 1529)");
        };
        if self.should_skip_show_existing(ref_hdr.frame_type) {
            self.parse.dangling_show_existing = None;
            return Ok(());
        }
        if !self.parse_only && self.pixel_refs.get(slot).and_then(Option::as_ref).is_none() {
            return fail("show_existing_frame: referans resmi yok (obu.c 1544)");
        }
        // dav1d `src/obu.c:1545-1548`: katı uyumda (`Config::strict_std_compliance`)
        // **gösterilemeyen** kareyi gösteren `show_existing` hata verir.
        // `showable` yuva durumudur; anahtar kare bir kez gösterilince 0'lanır
        // (`src/obu.c:1621`).
        if opts.strict && !self.parse.header_state.showable(slot) {
            return fail(
                "show_existing_frame: gösterilemeyen kare (strict_std_compliance; obu.c 1545-1548)",
            );
        }
        let prealloc_id = if !self.parse_only
            && let Some(cb) = prealloc
            && let Some(seq) = self.parse.sequence_header.as_ref()
        {
            let (chroma, bd) = frame_format(seq)?;
            let req = PreallocRequest {
                width: ref_hdr.width[1] as usize,
                height: ref_hdr.height as usize,
                chroma,
                bit_depth: bd,
                seq_hdr: Some(seq),
                frame_hdr: ref_hdr,
                content_light: self.parse.content_light,
                mastering_display: self.parse.mastering_display,
                itut_t35: &self.parse.itut_t35,
            };
            match cb(&req) {
                Ok(id) => id,
                Err(err) => {
                    self.parse.dangling_show_existing = None;
                    self.last_decode_error = Some(format!("{err}"));
                    self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                    return Err(err);
                }
            }
        } else {
            0
        };
        // P38: çıktı kopyası durum değişmeden **önce** ayrılır; bütçe ya da
        // ayırma hatasında başlık bekler (yukarıdaki hatalar gibi), yuvalar
        // değişmez.
        //
        // `show_existing` yeni blok çözmez: kendi çözülen karesi yoktur,
        // sayaçlar sıfırdır (§5.8); yalnız çıktıya uygulanan film grain
        // sayaçları dolar. Grain parametreleri gösterilen resmin kendi
        // başlığındandır (seed dâhil; dav1d `c->out` = yuvadaki resim,
        // `src/obu.c:1549-1551`).
        let cikti = if self.parse_only {
            None
        } else {
            let Some(entry) = self.pixel_refs.get(slot).and_then(Clone::clone) else {
                return fail("show_existing_frame: referans resmi yok (obu.c 1544)");
            };
            let mut tools = crate::decode::intra::ToolCounts::default();
            let butce = self.butce_defteri(0);
            let frame = build_output_frame_from_ref(
                self.config.apply_grain,
                &self.dsp,
                &self.pool,
                &entry,
                self.outputs_pushed,
                &mut tools,
                &butce,
            );
            self.tepe_kat(&butce);
            Some((entry, frame?, tools))
        };
        self.parse.dangling_show_existing = None;
        let Some(displayed) = self.parse.header_state.show_existing(slot) else {
            return fail("show_existing_frame: referans yuva boş (obu.c 1529)");
        };
        // **Tuzak:** başlık çıktısı **gösterilen** karenin başlığıdır
        // (framehdr differential'ı bunu bekler); dav1d `show_existing`'i her
        // zaman `c->out`a verir (`src/obu.c:1550-1560`).
        if self.collect_diagnostics {
            self.frame_outputs.push(displayed.clone());
            self.frame_output_flags.push(true);
        }
        self.parse.frames_completed += 1;
        if self.parse_only {
            self.outputs
                .push_back(OutputEntry::Unsupported("yalnız ayrıştırma"));
            self.outputs_pushed += 1;
            self.output_errors
                .push_back(Some("yalnız ayrıştırma kipi".into()));
            self.output_tools.push_back(None);
            self.output_capi_meta.push_back(OutputCapiMeta::default());
            return Ok(());
        }
        let Some((entry, mut frame, tools)) = cikti else {
            return fail("show_existing_frame: referans resmi yok (obu.c 1544)");
        };
        if self.parse.current_packet_timestamp.is_some() {
            frame.metadata_mut().timestamp = self.parse.current_packet_timestamp;
        }
        // dav1d `src/obu.c:1619-1640`: gösterilen kare KEY ise diğer yedi yuva
        // da bu kareye yenilenir; piksel yuvaları başlık durumuyla
        // (`HeaderReferenceState::show_existing`) aynı kalmalıdır.
        if displayed.frame_type == FrameType::Key {
            for (i, reference) in self.pixel_refs.iter_mut().enumerate() {
                if i != slot {
                    *reference = Some(Arc::clone(&entry));
                    self.slot_promises[i] = Some(FrameSlot::ready(Arc::clone(&entry)));
                }
            }
        }
        let capi_meta = OutputCapiMeta {
            seq_hdr: self.parse.sequence_header_arc.clone(),
            frame_hdr: Some(Arc::new(displayed)),
            content_light: self.parse.content_light,
            mastering_display: self.parse.mastering_display,
            itut_t35: core::mem::take(&mut self.parse.itut_t35),
            event_flags: core::mem::take(&mut self.parse.pending_event_flags),
            packet_tag: self.parse.current_packet_tag,
            new_temporal_unit: core::mem::take(&mut self.parse.pending_new_temporal_unit),
            prealloc_id,
        };
        self.outputs.push_back(OutputEntry::Frame(frame));
        self.outputs_pushed += 1;
        self.output_errors.push_back(None);
        self.output_tools.push_back(Some(tools));
        self.output_capi_meta.push_back(capi_meta);
        Ok(())
    }

    /// `n_fc >= 2` boru hattı için `show_existing_frame` işlemi.
    fn show_existing_pipeline(
        &mut self,
        slot: usize,
        opts: ParseOpts,
        prealloc: Option<&mut PreallocCb<'_>>,
    ) -> Result<()> {
        let fail = |what: &'static str| -> Result<()> {
            Err(BitstreamError::new(BitstreamErrorKind::InvalidFrameHeader(what), 0).into())
        };
        let Some(ref_hdr) = self.parse.header_state.reference(slot) else {
            return fail("show_existing_frame: referans yuva boş (obu.c 1529)");
        };
        if self.should_skip_show_existing(ref_hdr.frame_type) {
            self.parse.dangling_show_existing = None;
            return Ok(());
        }
        let Some(promise) = self.slot_promises.get(slot).and_then(Clone::clone) else {
            return fail("show_existing_frame: referans resmi yok (obu.c 1544)");
        };
        if promise.is_failed_nonblocking() {
            while !self.in_flight.is_empty() {
                self.retire_front_in_flight(true)?;
            }
            return fail("show_existing_frame: referans resmi yok (obu.c 1544)");
        }
        if opts.strict && !self.parse.header_state.showable(slot) {
            return fail(
                "show_existing_frame: gösterilemeyen kare (strict_std_compliance; obu.c 1545-1548)",
            );
        }
        let prealloc_id = if let Some(cb) = prealloc
            && let Some(seq) = self.parse.sequence_header.as_ref()
        {
            let (chroma, bd) = frame_format(seq)?;
            let req = PreallocRequest {
                width: ref_hdr.width[1] as usize,
                height: ref_hdr.height as usize,
                chroma,
                bit_depth: bd,
                seq_hdr: Some(seq),
                frame_hdr: ref_hdr,
                content_light: self.parse.content_light,
                mastering_display: self.parse.mastering_display,
                itut_t35: &self.parse.itut_t35,
            };
            match cb(&req) {
                Ok(id) => id,
                Err(err) => {
                    self.parse.dangling_show_existing = None;
                    self.last_decode_error = Some(format!("{err}"));
                    self.last_error_packet_tag = Some(self.parse.current_packet_tag);
                    return Err(err);
                }
            }
        } else {
            0
        };

        // Bütçe sınırı (`allocation_limit`) açıksa veya uçuş kuyruğu boşsa
        // senkron doğrula; aksi hâlde uçuşta çözülen karenin bitmesini
        // beklemeden `InFlightEntry::ShowExisting` olarak kuyruğa ekle.
        if self.config.allocation_limit.is_some() || self.in_flight.is_empty() {
            while !promise.is_done_nonblocking() {
                if let Some(work) = self.pool.try_pop_async_task() {
                    work();
                } else if !self.pool.help_active_batch() {
                    break;
                }
            }
            let Some(entry) = promise.wait(&self.cancel_flag) else {
                while !self.in_flight.is_empty() {
                    self.retire_front_in_flight(true)?;
                }
                return fail("show_existing_frame: referans resmi yok (obu.c 1544)");
            };

            self.pipeline_butce.taban_guncelle(self.kalici_bayt());
            let out_butce = Butce::alt_defter(&self.pipeline_butce);
            let mut tools = ToolCounts::default();
            let fg_pool = self.pool.single_thread_handle();
            let frame_res = build_output_frame_from_ref(
                self.config.apply_grain,
                &self.dsp,
                &fg_pool,
                &entry,
                self.outputs_pushed,
                &mut tools,
                &out_butce,
            );
            self.allocation_peak = self.allocation_peak.max(self.pipeline_butce.tepe());
            let mut frame = frame_res?;
            if self.parse.current_packet_timestamp.is_some() {
                frame.metadata_mut().timestamp = self.parse.current_packet_timestamp;
            }
            drop(out_butce);

            self.parse.dangling_show_existing = None;
            let Some(displayed) = self.parse.header_state.show_existing(slot) else {
                return fail("show_existing_frame: referans yuva boş (obu.c 1529)");
            };
            if self.collect_diagnostics {
                self.frame_outputs.push(displayed.clone());
                self.frame_output_flags.push(true);
            }
            self.parse.frames_completed += 1;
            let output_seq = self.outputs_pushed;
            self.outputs_pushed += 1;

            let refresh_all_key = displayed.frame_type == FrameType::Key;
            if refresh_all_key {
                let ready_slot = FrameSlot::ready(Arc::clone(&entry));
                for (i, slot_promise) in self.slot_promises.iter_mut().enumerate() {
                    if i != slot {
                        *slot_promise = Some(Arc::clone(&ready_slot));
                    }
                }
            }

            let capi_meta = OutputCapiMeta {
                seq_hdr: self.parse.sequence_header_arc.clone(),
                frame_hdr: Some(Arc::new(displayed)),
                content_light: self.parse.content_light,
                mastering_display: self.parse.mastering_display,
                itut_t35: core::mem::take(&mut self.parse.itut_t35),
                event_flags: core::mem::take(&mut self.parse.pending_event_flags),
                packet_tag: self.parse.current_packet_tag,
                new_temporal_unit: core::mem::take(&mut self.parse.pending_new_temporal_unit),
                prealloc_id,
            };

            if self.in_flight.is_empty() {
                if refresh_all_key {
                    for reference in &mut self.pixel_refs {
                        *reference = Some(Arc::clone(&entry));
                    }
                }
                self.outputs.push_back(OutputEntry::Frame(frame));
                self.output_errors.push_back(None);
                self.output_tools.push_back(Some(tools));
                self.output_capi_meta.push_back(capi_meta);
                self.pipeline_butce.taban_guncelle(self.kalici_bayt());
            } else {
                drop(frame);
                self.in_flight.push_back(InFlightEntry::ShowExisting {
                    promise,
                    output_seq,
                    refresh_all_key,
                    capi_meta,
                    packet_timestamp: self.parse.current_packet_timestamp,
                });
            }
            return Ok(());
        }

        self.parse.dangling_show_existing = None;
        let Some(displayed) = self.parse.header_state.show_existing(slot) else {
            return fail("show_existing_frame: referans yuva boş (obu.c 1529)");
        };
        if self.collect_diagnostics {
            self.frame_outputs.push(displayed.clone());
            self.frame_output_flags.push(true);
        }
        self.parse.frames_completed += 1;
        let output_seq = self.outputs_pushed;
        self.outputs_pushed += 1;

        let refresh_all_key = displayed.frame_type == FrameType::Key;
        if refresh_all_key {
            for (i, slot_promise) in self.slot_promises.iter_mut().enumerate() {
                if i != slot {
                    *slot_promise = Some(Arc::clone(&promise));
                }
            }
        }
        let capi_meta = OutputCapiMeta {
            seq_hdr: self.parse.sequence_header_arc.clone(),
            frame_hdr: Some(Arc::new(displayed)),
            content_light: self.parse.content_light,
            mastering_display: self.parse.mastering_display,
            itut_t35: core::mem::take(&mut self.parse.itut_t35),
            event_flags: core::mem::take(&mut self.parse.pending_event_flags),
            packet_tag: self.parse.current_packet_tag,
            new_temporal_unit: core::mem::take(&mut self.parse.pending_new_temporal_unit),
            prealloc_id,
        };
        self.in_flight.push_back(InFlightEntry::ShowExisting {
            promise,
            output_seq,
            refresh_all_key,
            capi_meta,
            packet_timestamp: self.parse.current_packet_timestamp,
        });
        Ok(())
    }

    /// Sıradaki çıktı karesini verir (`Ok(None)` = şu an çıktı yok; dav1d
    /// `dav1d_get_picture` `EAGAIN`).
    ///
    /// - [`DecoderState::Ready`] durumunda: emekliye ayrılmış `outputs` kuyruğunu
    ///   verir; ayrıca uçuş kuyruğunun (`in_flight`) başındaki görev zaten
    ///   tamamlanmışsa bloklamadan emekliye ayırıp verir; başındaki görev hâlâ
    ///   çalışıyorsa boru hattını tıkamamak için `Ok(None)` döner.
    /// - [`DecoderState::Draining`] (`flush()` sonrası) durumunda: uçuş
    ///   kuyruğunun başındaki görevi tamamlanana kadar bekleyip sırayla verir;
    ///   tüm uçuş kuyruğu ve `outputs` boşaldığında `Ok(None)` döner.
    ///
    /// # Hatalar
    ///
    /// [`Error::Unsupported`]: yalnız tanılama kipinde ([`Decoder::set_parse_only`])
    /// her kare için "yalnız ayrıştırma" girdisi veya uçuşta geç hata.
    pub fn receive_frame(&mut self) -> Result<Option<Frame>> {
        let block = self.state == DecoderState::Draining;
        Ok(self.receive_frame_capi(block)?.map(|(frame, _)| frame))
    }

    /// C-ABI (`dav1d_get_picture`) için sıradaki çıktı karesini ve eşlik eden
    /// C-ABI metaverisini döndürür (`block == true` ise uçuş kuyruğunun başını bekler).
    pub(crate) fn receive_frame_capi(
        &mut self,
        block: bool,
    ) -> Result<Option<(Frame, OutputCapiMeta)>> {
        loop {
            if let Some(entry) = self.outputs.pop_front() {
                let capi_meta = self.output_capi_meta.pop_front().unwrap_or_default();
                let out = match entry {
                    OutputEntry::Frame(frame) => {
                        let _ = self.output_errors.pop_front();
                        // Tanılama kuralı: sayaçlar **bu çıktı karesinin
                        // kendi** çözülen karesine bağlıdır; `show_existing` sıfır alır.
                        self.last_output_tools = self.output_tools.pop_front().flatten();
                        Ok(Some((frame, capi_meta)))
                    }
                    OutputEntry::Unsupported(reason) => {
                        if let Some(Some(detail)) = self.output_errors.pop_front() {
                            self.last_decode_error = Some(detail);
                        }
                        let _ = self.output_tools.pop_front();
                        self.last_output_tools = None;
                        self.last_error_packet_tag = Some(capi_meta.packet_tag);
                        Err(Error::Unsupported(reason))
                    }
                };
                self.sync_shared_kalici();
                return out;
            }
            if !self.retire_front_in_flight(block)? {
                return Ok(None);
            }
        }
    }

    /// Son çözme hatasını üreten paketin C-ABI kimlik etiketi.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    pub(crate) const fn last_error_packet_tag(&self) -> Option<u64> {
        self.last_error_packet_tag
    }

    /// Verilen C-ABI paket etiketine (`packet_tag`) sahip bekleyen çıktı veya uçuş görevi var mı?
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    pub(crate) fn has_pending_packet_tag(&self, tag: u64) -> bool {
        self.output_capi_meta.iter().any(|m| m.packet_tag == tag)
            || self.in_flight.iter().any(|e| match e {
                InFlightEntry::Decode(t) => t.packet_tag == tag,
                InFlightEntry::ShowExisting { capi_meta, .. } => capi_meta.packet_tag == tag,
            })
    }

    /// Verilen ön-tahsis kimliğine (`prealloc_id`) sahip bekleyen çıktı veya uçuş görevi var mı?
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    pub(crate) fn has_pending_prealloc_id(&self, id: u64) -> bool {
        if id == 0 {
            return false;
        }
        self.output_capi_meta.iter().any(|m| m.prealloc_id == id)
            || self.in_flight.iter().any(|e| match e {
                InFlightEntry::Decode(t) => t.prealloc_id == id,
                InFlightEntry::ShowExisting { capi_meta, .. } => capi_meta.prealloc_id == id,
            })
    }

    /// C-ABI (`dav1d_apply_grain`) için verilen kareye film grain uygular.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    pub(crate) fn apply_grain_capi(
        &self,
        frame: &Frame,
        fg: &crate::bitstream::frame_header::FilmGrainHeader,
        mc_identity: bool,
    ) -> Result<Frame> {
        let butce = self.butce_defteri(0);
        let mut tools = ToolCounts::default();
        build_output_frame(
            true,
            &self.dsp,
            &self.pool,
            frame,
            fg,
            mc_identity,
            self.outputs_pushed,
            &mut tools,
            &butce,
        )
    }

    /// Çıktı sırasıyla biriken kare başlıklarını **boşaltır** (yalnızca
    /// [`Decoder::set_collect_diagnostics`]`(true)` etkinken doldurulur;
    /// varsayılan olarak boş döner).
    ///
    /// Sıra dav1d `dav1d_get_picture` sırasıyla aynıdır: tek iş parçacıklı
    /// çözücüde kodlama sırası; `show_existing_frame` durumunda **gösterilen
    /// referans karenin** başlığı (dav1d `src/obu.c` 1581–1641). Görünmez
    /// kareler de (`show_frame = 0`) kare tamamlandığında buraya eklenir;
    /// dav1d oracle'ı `output_invisible_frames = 1` ile aynı davranışı üretir.
    #[must_use]
    pub fn drain_frame_headers(&mut self) -> Vec<FrameHeader> {
        core::mem::take(&mut self.frame_outputs)
    }

    /// `drain_frame_headers()` ile **hizalı** gösterim bayrakları (yalnızca
    /// [`Decoder::set_collect_diagnostics`]`(true)` etkinken doldurulur).
    ///
    /// `true` ise çıktı karesi `show_frame`/`show_existing_frame` ile
    /// gösterilir (dav1d varsayılan çıktı kümesi).
    pub fn drain_frame_output_flags(&mut self) -> Vec<bool> {
        core::mem::take(&mut self.frame_output_flags)
    }

    /// Bekleyen çıktı kare başlığı sayısı (yalnızca
    /// [`Decoder::set_collect_diagnostics`]`(true)` etkinken büyür).
    #[must_use]
    pub fn pending_frame_headers(&self) -> usize {
        self.frame_outputs.len()
    }

    /// Bekleyen (tile'ları henüz tamamlanmamış) kare başlığı.
    #[must_use]
    pub fn pending_frame_header(&self) -> Option<&FrameHeader> {
        self.parse.header_state.current()
    }

    /// Referans yuvasındaki kare başlığı (0–7).
    #[must_use]
    pub fn reference_header(&self, slot: usize) -> Option<&FrameHeader> {
        self.parse.header_state.reference(slot)
    }

    /// Bekleyen karenin şimdiye kadar aldığı tile sayısı.
    #[must_use]
    pub fn pending_tile_count(&self) -> usize {
        self.parse.header_state.n_tiles()
    }

    /// Ayrıştırılan frame header sayısı (redundant dâhil).
    #[must_use]
    pub const fn frame_header_parses(&self) -> u64 {
        self.parse.frame_header_parses
    }

    /// Tamamlanan (bütün tile'ları gelen) kare sayısı; `show_existing_frame`
    /// gösterimleri dâhil.
    #[must_use]
    pub const fn frames_completed(&self) -> u64 {
        self.parse.frames_completed
    }

    /// İşlenen tile group OBU sayısı.
    #[must_use]
    pub const fn tile_groups_processed(&self) -> u64 {
        self.parse.tile_groups_processed
    }

    /// Sınırlarıyla atlanan tile verisi (bayt).
    #[must_use]
    pub const fn tile_bytes_skipped(&self) -> u64 {
        self.parse.tile_bytes_skipped
    }

    /// Son tile group'un başlangıç/bitiş ve tile boyutları.
    #[must_use]
    pub fn last_tile_group(&self) -> Option<&TileGroupInfo> {
        self.parse.last_tile_group.as_ref()
    }

    /// Bekleyen paketlerin işlenmesini ve karelerin teslimini ister.
    ///
    /// Durumu [`DecoderState::Draining`] yapar ve **idempotenttir**: arka arkaya
    /// çağrılabilir. Bu durumda `send_packet` [`Error::InvalidState`] döner.
    pub fn flush(&mut self) {
        self.state = DecoderState::Draining;
    }

    /// Çözücüyü yeni bir akışa hazırlar (dav1d `dav1d_flush`,
    /// `src/lib.c:526-557`).
    ///
    /// Durum [`DecoderState::Ready`]; etkin sequence header, bekleyen kare,
    /// referans yuvaları (başlık + piksel + CDF + segment haritası), çıktı
    /// kuyruğu ve sayaçlar **sıfırlanır** (GUV-01: bozuk akış sonrası tanımlı
    /// kurtarma noktası). Yapılandırma, CPU/DSP seçimi ve dav1d gibi seçili
    /// operating point maskesi (`operating_point_idc`, `max_spatial_id`;
    /// bir sonraki dizi başlığına kadar katman süzgeci) **korunur**.
    pub fn reset(&mut self) {
        self.cancel_and_drain_in_flight();
        self.state = DecoderState::Ready;
        self.parse = ParseState {
            operating_point_idc: self.parse.operating_point_idc,
            max_spatial_id: self.parse.max_spatial_id,
            ..ParseState::default()
        };
        self.frame_outputs.clear();
        self.frame_output_flags.clear();
        self.outputs.clear();
        self.output_errors.clear();
        self.output_tools.clear();
        self.output_capi_meta.clear();
        self.last_error_packet_tag = None;
        self.last_output_tools = None;
        self.slot_promises = [None, None, None, None, None, None, None, None];
        let has_limit = self.config.allocation_limit.is_some();
        for slot in &mut self.pixel_refs {
            if let Some(arc) = slot.take()
                && let Ok(rf) = Arc::try_unwrap(arc)
            {
                self.pool.buf_pool().recycle_ref_frame(rf, has_limit);
            }
        }
        self.pipeline_butce = match self.shared_butce {
            Some(ref ust) => Arc::new(Butce::alt_defter(ust)),
            None => Arc::new(Butce::yeni(self.config.allocation_limit, 0)),
        };
        self.decoded_frames = 0;
        self.unsupported_frames = 0;
        self.last_decode_error = None;
        self.outputs_pushed = 0;
        // P38: kare başına büyüyen tanılama listesi de sıfırlanır (bütçe
        // defterinin tabanında sayılır; `decoded_frames` ile hizalı).
        self.decoded_tools.clear();
    }

    /// Son çözülen karenin araç kapsamı sayaçları (tanılama).
    ///
    /// Sayaçlar üretim API'sinin **anlamını değiştirmez**; yalnız gözlem
    /// sağlar ve bağımlılık eklemez (tanılama sayacı).
    #[must_use]
    pub fn last_tool_counts(&self) -> Option<&crate::decode::intra::ToolCounts> {
        self.last_output_tools.as_ref()
    }

    /// Çözülen **her** karenin araç sayaçları (çözüm sırasıyla; yalnızca
    /// [`Decoder::set_collect_diagnostics`]`(true)` etkinken doldurulur).
    #[must_use]
    pub fn decoded_tool_counts(&self) -> &[crate::decode::intra::ToolCounts] {
        &self.decoded_tools
    }

    /// Son çözüm hatası (tanılama).
    #[must_use]
    pub fn last_decode_error(&self) -> Option<&str> {
        self.last_decode_error.as_deref()
    }

    /// Başarıyla çözülen kare sayısı.
    #[must_use]
    pub const fn decoded_frames(&self) -> u64 {
        self.decoded_frames
    }

    /// Kapsam dışı olduğu için çözülemeyen çıktı kare sayısı.
    #[must_use]
    pub const fn unsupported_frames(&self) -> u64 {
        self.unsupported_frames
    }

    /// Geçerli durum.
    #[must_use]
    pub const fn state(&self) -> DecoderState {
        self.state
    }

    /// Etkin (son başarıyla ayrıştırılan) sequence header.
    #[must_use]
    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.parse.sequence_header.as_ref()
    }

    /// Seçili operating point'in `operating_point_idc` maskesi (12 bit).
    #[must_use]
    pub const fn operating_point_idc(&self) -> u16 {
        self.parse.operating_point_idc
    }

    /// Seçili operating point'in en büyük spatial katman kimliği.
    #[must_use]
    pub const fn max_spatial_id(&self) -> u8 {
        self.parse.max_spatial_id
    }

    /// Başarıyla uygulanan sequence header sayısı.
    #[must_use]
    pub const fn sequence_header_updates(&self) -> u64 {
        self.parse.sequence_header_updates
    }

    /// Operating point katman süzgeciyle atlanan OBU sayısı.
    #[must_use]
    pub const fn layer_obus_skipped(&self) -> u64 {
        self.parse.layer_obus_skipped
    }

    /// Kurulumda kullanılan yapılandırma.
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Etkin CPU bayrakları (kullanıcı maskesi düşülmüş hâli).
    #[must_use]
    pub const fn cpu_flags(&self) -> CpuFlags {
        self.cpu_flags
    }

    /// Kurulumda **bir kez** seçilen DSP tablosu (değişmez).
    #[must_use]
    pub const fn dsp(&self) -> &DspDispatch {
        &self.dsp
    }
    /// Metadata OBU ayrıştırma ve denetimi (`dav1d` `src/obu.c:1356-1515`):
    /// `metadata_type` leb128 hatası; HDR CLL (32 bit), HDR MDCV (192 bit) ve
    /// ITU-T T.35 yüklerini ayrıştırıp saklar.
    fn process_metadata_obu(&mut self, obu: &Obu<'_>, strict: bool) -> Result<()> {
        let base = obu.payload_offset_bits();
        let mut reader = BitReader::new(obu.payload);
        let meta_type = reader
            .read_uleb128()
            .map_err(|err| rebase_payload_error(err, base))?;
        match meta_type {
            1 => {
                // OBU_META_HDR_CLL (32 bit)
                let max_cll = reader
                    .read_bits(16)
                    .map_err(|err| rebase_payload_error(err, base))?
                    as u16;
                let max_fall = reader
                    .read_bits(16)
                    .map_err(|err| rebase_payload_error(err, base))?
                    as u16;
                reader
                    .read_trailing_bits(strict)
                    .map_err(|err| rebase_payload_error(err, base))?;
                self.parse.content_light = Some((max_cll, max_fall));
                Ok(())
            }
            2 => {
                // OBU_META_HDR_MDCV (192 bit)
                let mut primaries = [[0u16; 2]; 3];
                for p in &mut primaries {
                    p[0] = reader
                        .read_bits(16)
                        .map_err(|err| rebase_payload_error(err, base))?
                        as u16;
                    p[1] = reader
                        .read_bits(16)
                        .map_err(|err| rebase_payload_error(err, base))?
                        as u16;
                }
                let wp0 = reader
                    .read_bits(16)
                    .map_err(|err| rebase_payload_error(err, base))?
                    as u16;
                let wp1 = reader
                    .read_bits(16)
                    .map_err(|err| rebase_payload_error(err, base))?
                    as u16;
                let max_hi = reader
                    .read_bits(16)
                    .map_err(|err| rebase_payload_error(err, base))?;
                let max_lo = reader
                    .read_bits(16)
                    .map_err(|err| rebase_payload_error(err, base))?;
                let min_hi = reader
                    .read_bits(16)
                    .map_err(|err| rebase_payload_error(err, base))?;
                let min_lo = reader
                    .read_bits(16)
                    .map_err(|err| rebase_payload_error(err, base))?;
                reader
                    .read_trailing_bits(strict)
                    .map_err(|err| rebase_payload_error(err, base))?;
                self.parse.mastering_display = Some(MasteringDisplayMeta {
                    primaries,
                    white_point: [wp0, wp1],
                    max_luminance: (max_hi << 16) | max_lo,
                    min_luminance: (min_hi << 16) | min_lo,
                });
                Ok(())
            }
            4 => {
                // OBU_META_ITUT_T35 (`src/obu.c:1445-1501`)
                let start = reader.byte_position();
                if let Some(rest) = obu.payload.get(start..) {
                    let Some(end) = memchr::memrchr(0x80, rest) else {
                        return Ok(());
                    };
                    if end == 0 || rest[end + 1..].iter().any(|&b| b != 0) {
                        return Ok(());
                    }
                    let country_code = rest[0];
                    let (ext, payload_start) = if country_code == 0xFF {
                        if end < 2 {
                            return Ok(());
                        }
                        (rest[1], 2usize)
                    } else {
                        (0u8, 1usize)
                    };
                    if payload_start < end {
                        self.parse.itut_t35.push(ItutT35Meta {
                            country_code,
                            country_code_extension_byte: ext,
                            payload: rest[payload_start..end].to_vec(),
                        });
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
/// `n_fc >= 2` boru hattı için referans yuvası vaatlerini (`FrameSlot`) başlık
/// düzeyinde doğrular (`dav1d` `decode.c:3446-3478`).
fn validate_frame_refs_slots(
    hdr: &FrameHeader,
    chroma: ChromaFormat,
    bd: BitDepth,
    slots: &[Option<Arc<FrameSlot>>; 8],
) -> Result<()> {
    let is_inter = matches!(hdr.frame_type, FrameType::Inter | FrameType::Switch);
    if is_inter {
        for i in 0..7 {
            let slot = usize::from(hdr.refidx[i] as u8);
            let Some(r) = slots.get(slot).and_then(Option::as_ref) else {
                return Err(Error::Unsupported("referans çözülmedi: inter referans"));
            };
            if r.is_failed_nonblocking() {
                return Err(Error::Unsupported("referans çözülmedi: inter referans"));
            }
            let cur_w = hdr.width[0] as usize;
            let cur_h = hdr.height as usize;
            if cur_w * 2 < r.width
                || cur_h * 2 < r.height
                || cur_w > r.width * 16
                || cur_h > r.height * 16
            {
                return Err(Error::InvalidData("ölçekli referans oranı aralık dışında"));
            }
            if r.chroma != chroma || r.bd != bd {
                return Err(Error::InvalidData("referans piksel biçimi uyuşmuyor"));
            }
        }
        if hdr.primary_ref_frame != 7 {
            let pr = usize::from(hdr.primary_ref_frame);
            let slot = usize::from(hdr.refidx[pr] as u8);
            let Some(r) = slots.get(slot).and_then(Option::as_ref) else {
                return Err(Error::Unsupported("referans çözülmedi: primary_ref_frame"));
            };
            if r.is_failed_nonblocking() {
                return Err(Error::Unsupported("referans çözülmedi: primary_ref_frame"));
            }
        }
    }
    Ok(())
}

fn fill_fg_tools(fg: &crate::bitstream::frame_header::FilmGrainHeader, tools: &mut ToolCounts) {
    let d = &fg.data;
    tools.fg_applied = 1;
    tools.fg_update0 = u64::from(!fg.update);
    tools.fg_overlap = u64::from(d.overlap_flag);
    tools.fg_csfl = u64::from(d.chroma_scaling_from_luma);
    tools.fg_clip = u64::from(d.clip_to_restricted_range);
    tools.fg_ar_lag1 = u64::try_from(d.ar_coeff_lag).unwrap_or(0) + 1;
    tools.fg_y = u64::from(d.num_y_points != 0);
    tools.fg_uv = u64::from(d.num_uv_points[0] != 0 || d.num_uv_points[1] != 0);
}

/// Referans olarak saklanmayan (`refresh_frame_flags == 0`) karelerde `PicBuf`'u
/// yerinde [`Frame`]'e sıkıştırır ve varsa şerit bazlı Film Grain uygular (`4.A.1`).
#[allow(clippy::too_many_arguments)]
fn build_output_frame_in_place(
    apply_grain_cfg: bool,
    dsp: &DspDispatch,
    pool: &WorkerPool,
    pic: crate::decode::intra::PicBuf,
    chroma: ChromaFormat,
    bd: BitDepth,
    fg: &crate::bitstream::frame_header::FilmGrainHeader,
    mc_identity: bool,
    output_seq: u64,
    tools: &mut ToolCounts,
    butce: &Butce,
) -> Result<Frame> {
    use crate::decode::filmgrain;
    let (_, total) = Frame::plane_byte_lens_and_total(pic.width, pic.height, chroma, bd)?;
    butce.ayir(total)?;
    let mut frame = Frame::from_owned_pic(
        pic.width,
        pic.height,
        chroma,
        bd,
        pic.data,
        pic.stride,
        !butce.has_limit(),
    )
    .inspect_err(|_| butce.birak(total))?;
    let d = &fg.data;
    if !apply_grain_cfg || !filmgrain::has_grain(d) {
        return Ok(frame);
    }
    fill_fg_tools(fg, tools);
    crate::dsp::profil_bd!(bd);
    filmgrain::apply_grain_in_place(
        &mut frame,
        d,
        mc_identity,
        dsp.for_bit_depth(bd).fg(),
        filmgrain::trace_picture(output_seq),
        butce,
        pool,
    )
    .inspect_err(|_| butce.birak(total))?;
    Ok(frame)
}

/// Referans olarak saklanacak (`refresh_frame_flags != 0`, `n_fc == 1`) karenin
/// `&mut PicBuf` düzlemlerinden tek geçişte çıktı [`Frame`]'ini üretir (`4.A.1`).
#[allow(clippy::too_many_arguments)]
fn build_output_frame_from_mut_pic(
    apply_grain_cfg: bool,
    dsp: &DspDispatch,
    pool: &WorkerPool,
    pic: &mut crate::decode::intra::PicBuf,
    chroma: ChromaFormat,
    bd: BitDepth,
    fg: &crate::bitstream::frame_header::FilmGrainHeader,
    mc_identity: bool,
    output_seq: u64,
    tools: &mut ToolCounts,
    butce: &Butce,
) -> Result<Frame> {
    use crate::decode::filmgrain;
    let (lens, _) = Frame::plane_byte_lens_and_total(pic.width, pic.height, chroma, bd)?;
    let bufs = pool.buf_pool().take_u8_triple(lens, butce.has_limit());
    let d = &fg.data;
    if !apply_grain_cfg || !filmgrain::has_grain(d) {
        return Frame::new_from_pic_in(
            pic.width,
            pic.height,
            chroma,
            bd,
            &mut pic.data,
            pic.stride,
            bufs,
            butce,
            pool,
        );
    }
    fill_fg_tools(fg, tools);
    crate::dsp::profil_bd!(bd);
    filmgrain::apply_grain_from_pic_mut(
        pic,
        chroma,
        bd,
        d,
        mc_identity,
        dsp.for_bit_depth(bd).fg(),
        filmgrain::trace_picture(output_seq),
        bufs,
        butce,
        pool,
    )
}

/// Paylaşılan `Arc<RefFrame>` içindeki `PicBuf` düzlemlerinden tek geçişte çıktı
/// [`Frame`]'ini üretir (`show_existing_frame` ve boru hattı referans yolu; `4.A.1`).
fn build_output_frame_from_ref(
    apply_grain_cfg: bool,
    dsp: &DspDispatch,
    pool: &WorkerPool,
    rf: &Arc<RefFrame>,
    output_seq: u64,
    tools: &mut ToolCounts,
    butce: &Butce,
) -> Result<Frame> {
    use crate::decode::filmgrain;
    let (lens, _) =
        Frame::plane_byte_lens_and_total(rf.pic.width, rf.pic.height, rf.chroma, rf.bd)?;
    let bufs = pool.buf_pool().take_u8_triple(lens, butce.has_limit());
    let fg = &rf.hdr.film_grain;
    let d = &fg.data;
    if !apply_grain_cfg || !filmgrain::has_grain(d) {
        return Frame::new_from_shared_pic_in(
            rf.pic.width,
            rf.pic.height,
            rf.chroma,
            rf.bd,
            rf,
            |r, idx| (r.pic.data[idx].as_slice(), r.pic.stride[idx]),
            bufs,
            butce,
            pool,
        );
    }
    fill_fg_tools(fg, tools);
    crate::dsp::profil_bd!(rf.bd);
    filmgrain::apply_grain_from_ref_pic(
        rf,
        d,
        rf.mc_identity,
        dsp.for_bit_depth(rf.bd).fg(),
        filmgrain::trace_picture(output_seq),
        bufs,
        butce,
        pool,
    )
}

/// Çıktı karesini (varsa Film Grain uygulayarak) üretir (`dav1d_apply_grain` C-ABI yolu).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn build_output_frame(
    apply_grain_cfg: bool,
    dsp: &DspDispatch,
    pool: &WorkerPool,
    frame: &Frame,
    fg: &crate::bitstream::frame_header::FilmGrainHeader,
    mc_identity: bool,
    output_seq: u64,
    tools: &mut ToolCounts,
    butce: &Butce,
) -> Result<Frame> {
    use crate::decode::filmgrain;
    let d = &fg.data;
    if !apply_grain_cfg || !filmgrain::has_grain(d) {
        let lens = [
            frame
                .plane(crate::frame::PlaneType::Y)
                .map_or(0, |p| p.allocation_bytes()),
            frame
                .plane(crate::frame::PlaneType::U)
                .map_or(0, |p| p.allocation_bytes()),
            frame
                .plane(crate::frame::PlaneType::V)
                .map_or(0, |p| p.allocation_bytes()),
        ];
        let bufs = pool.buf_pool().take_u8_triple(lens, butce.has_limit());
        return frame.try_clone_with_bufs(bufs, butce);
    }
    fill_fg_tools(fg, tools);
    crate::dsp::profil_bd!(frame.bit_depth());
    filmgrain::apply_grain(
        frame,
        d,
        mc_identity,
        dsp.for_bit_depth(frame.bit_depth()).fg(),
        filmgrain::trace_picture(output_seq),
        butce,
        pool,
    )
}

/// Sequence header hatasını OBU yükü tabanından paket tabanına taşır.
fn rebase_payload_error(err: Error, base_bits: usize) -> Error {
    match err {
        Error::Bitstream(e) => Error::Bitstream(BitstreamError::new(
            e.kind,
            e.bit_offset.saturating_add(base_bits),
        )),
        other => other,
    }
}

/// OBU yüküne göre konumlu bit akışı hatası.
fn payload_error(obu: &Obu<'_>, kind: BitstreamErrorKind) -> Error {
    Error::Bitstream(BitstreamError::new(kind, obu.payload_offset_bits()))
}

/// Bir frame header / frame / tile group OBU'sunu işler.
/// Ayrıştırma ayarları (`Config`'den; paket başına kopyalanır).
#[derive(Clone, Copy, Debug)]
struct ParseOpts {
    /// dav1d `frame_size_limit` (0 = sınırsız).
    frame_size_limit: u64,
    /// dav1d `strict_std_compliance`.
    strict: bool,
}

/// dav1d `src/obu.c:1300-1304` kare alanı sınırı.
///
/// `limit == 0` **sınırsızdır** (dav1d varsayılanı, `src/lib.c:82`). Sıfırdan
/// farklıysa `width[1] * height > limit` olan kare başlığı reddedilir
/// (`DAV1D_ERR(ERANGE)`); çarpım `u64`'te taşmaz (kenar boşlukları `saturating`).
fn frame_size_exceeds(width: i32, height: i32, limit: u64) -> bool {
    if limit == 0 {
        return false;
    }
    u64::from(width.unsigned_abs()).saturating_mul(u64::from(height.unsigned_abs())) > limit
}

/// Bekleyen karenin tile sayısı (`cols × rows`; bekleyen kare yoksa 0).
fn current_tiles(state: &HeaderReferenceState) -> u16 {
    state
        .current()
        .map_or(0, |h| u16::from(h.tiling.cols) * u16::from(h.tiling.rows))
}

/// `parse_tile_hdr()` + bayt hizası (dav1d `src/obu.c:1154-1167`, `1333-1336`):
/// `(tile_start, tile_end, veri_ofseti)`.
fn tile_group_header(payload: &[u8], header: &FrameHeader) -> Result<(u16, u16, usize)> {
    let n_tiles = u32::from(header.tiling.cols) * u32::from(header.tiling.rows);
    let mut reader = BitReader::new(payload);
    let have_tile_pos = n_tiles > 1 && reader.read_bit()? != 0;
    let (start, end) = if have_tile_pos {
        let n_bits = u32::from(header.tiling.log2_cols) + u32::from(header.tiling.log2_rows);
        let start = reader.read_bits(n_bits)? as u16;
        let end = reader.read_bits(n_bits)? as u16;
        (start, end)
    } else {
        (0, n_tiles.saturating_sub(1) as u16)
    };
    reader.align_to_byte();
    let data_offset = reader.byte_position();
    if data_offset > payload.len() {
        return Err(BitstreamError::new(
            BitstreamErrorKind::InvalidTileGroup("tile group başlığı OBU'yu aştı"),
            reader.bit_position(),
        )
        .into());
    }
    Ok((start, end, data_offset))
}

/// Tile group'ları tile'lara böler (dav1d `dav1d_decode_frame_init_cdf`,
/// `src/decode.c:3159-3189`); boyut hatası **geç** hatadır. Tile verisi
/// kopyalanmaz (P38): dilimler bekleyen tile group verisine bakar.
fn split_tile_groups<'g>(
    groups: &'g [TileGroupRaw],
    header: &FrameHeader,
) -> Result<Vec<&'g [u8]>> {
    let n = usize::from(header.tiling.cols) * usize::from(header.tiling.rows);
    let n_bytes = usize::from(header.tiling.n_bytes);
    let mut tiles: Vec<&[u8]> = vec![&[]; n];
    for group in groups {
        let data = &group.data[..];
        let mut pos = 0usize;
        let mut size = data.len();
        for j in group.start..=group.end {
            let tile_sz = if j == group.end {
                size
            } else {
                if n_bytes > size {
                    return Err(Error::InvalidData(
                        "tile boyut alanı tile grubunu aşıyor (decode.c 3168)",
                    ));
                }
                let mut sz = 0usize;
                for k in 0..n_bytes {
                    sz |= usize::from(data[pos + k]) << (8 * k);
                }
                pos += n_bytes;
                size -= n_bytes;
                sz += 1;
                if sz > size {
                    return Err(Error::InvalidData(
                        "tile boyutu tile grubunu aşıyor (decode.c 3174)",
                    ));
                }
                sz
            };
            let Some(slot) = tiles.get_mut(usize::from(j)) else {
                return Err(Error::InvalidData("tile indeksi aralık dışında"));
            };
            *slot = &data[pos..pos + tile_sz];
            pos += tile_sz;
            size -= tile_sz;
        }
    }
    Ok(tiles)
}

/// Metadata OBU denetimleri (birim test yardımcısı):
/// `metadata_type` leb128 hatası; HDR CLL (32 bit) ve HDR MDCV (192 bit)
/// yükünün taşması ya da trailing bits hatası.
#[cfg(test)]
fn validate_metadata(obu: &Obu<'_>, strict: bool) -> Result<()> {
    let base = obu.payload_offset_bits();
    let mut reader = BitReader::new(obu.payload);
    let meta_type = reader
        .read_uleb128()
        .map_err(|err| rebase_payload_error(err, base))?;
    let bits = match meta_type {
        1 => 32,  // OBU_META_HDR_CLL
        2 => 192, // OBU_META_HDR_MDCV
        _ => return Ok(()),
    };
    for _ in 0..bits / 16 {
        reader
            .read_bits(16)
            .map_err(|err| rebase_payload_error(err, base))?;
    }
    reader
        .read_trailing_bits(strict)
        .map_err(|err| rebase_payload_error(err, base))
}

/// Dizi başlığının piksel biçimi (`decode_frame` ile aynı eşleme).
fn frame_format(
    seq: &SequenceHeader,
) -> Result<(crate::frame::ChromaFormat, crate::frame::BitDepth)> {
    use crate::frame::{BitDepth, ChromaFormat};
    let chroma = match seq.layout.raw() {
        0 => ChromaFormat::Monochrome,
        1 => ChromaFormat::C420,
        2 => ChromaFormat::C422,
        _ => ChromaFormat::C444,
    };
    let bd = match seq.hbd {
        0 => BitDepth::Eight,
        1 => BitDepth::Ten,
        2 => BitDepth::Twelve,
        _ => return Err(Error::Unsupported("geçersiz seq hbd")),
    };
    Ok((chroma, bd))
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{Decoder, DecoderState};
    use crate::api::{Config, Packet};
    use crate::cpu::CpuFlags;
    use crate::dsp::DspDispatch;
    use crate::error::{Error, Result};

    /// Testler için MSB-first bit yazıcı.
    struct BitWriter {
        bits: Vec<u8>,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }

        fn bit(&mut self, value: u32) {
            self.bits.push((value & 1) as u8);
        }

        fn bits(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bit((value >> i) & 1);
            }
        }

        fn bytes(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, bit) in self.bits.iter().enumerate() {
                if *bit != 0 {
                    out[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            out
        }
    }

    /// Reduced still-picture sequence header yükü (S2a.1 asgari başlık).
    fn reduced_seq_header_payload() -> Vec<u8> {
        let mut w = BitWriter::new();
        w.bits(0, 3); // profile 0
        w.bit(1); // still_picture
        w.bit(1); // reduced_still_picture_header
        w.bits(0, 3); // major_level 0
        w.bits(0, 2); // minor_level 0
        w.bits(0, 4); // width_n_bits 1
        w.bits(0, 4); // height_n_bits 1
        w.bit(1); // max_width = 2
        w.bit(1); // max_height = 2
        w.bit(0); // sb128
        w.bit(0); // filter_intra
        w.bit(0); // intra_edge_filter
        w.bit(0); // super_res
        w.bit(0); // cdef
        w.bit(0); // restoration
        w.bit(0); // high_bitdepth
        w.bit(0); // mono_chrome
        w.bit(0); // color_description_present
        w.bit(0); // color_range
        w.bits(0, 2); // chroma_sample_position
        w.bit(0); // separate_uv_delta_q
        w.bit(0); // film_grain_present
        w.bit(1); // trailing_one_bit
        while !w.bits.len().is_multiple_of(8) {
            w.bit(0);
        }
        w.bytes()
    }

    /// Non-reduced, tek operating point'li, istenen idc'yi taşıyan başlık yükü.
    fn seq_header_payload_with_idc(idc: u16) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.bits(0, 3); // profile 0
        w.bit(0); // still
        w.bit(0); // reduced
        w.bit(0); // timing_info_present
        w.bit(0); // display_model_info_present
        w.bits(0, 5); // 1 operating point
        w.bits(u32::from(idc), 12);
        w.bits(0, 3); // major 2
        w.bits(0, 2); // minor
        w.bits(0, 4); // width_n_bits 1
        w.bits(0, 4); // height_n_bits 1
        w.bit(0); // max_width = 1
        w.bit(0); // max_height = 1
        w.bit(0); // frame_id_numbers_present
        w.bit(0); // sb128
        w.bit(0); // filter_intra
        w.bit(0); // intra_edge_filter
        w.bit(0); // inter_intra
        w.bit(0); // masked_compound
        w.bit(0); // warped_motion
        w.bit(0); // dual_filter
        w.bit(0); // order_hint
        w.bit(0); // screen_content_tools ilk
        w.bit(0); // ikinci -> OFF
        // force_integer_mv okunmaz (OFF)
        w.bit(0); // super_res
        w.bit(0); // cdef
        w.bit(0); // restoration
        w.bit(0); // high_bitdepth
        w.bit(0); // mono
        w.bit(0); // color_description_present
        w.bit(0); // color_range
        w.bits(0, 2); // chr
        w.bit(0); // separate_uv_delta_q
        w.bit(0); // film_grain_present
        w.bit(1); // trailing_one_bit
        while !w.bits.len().is_multiple_of(8) {
            w.bit(0);
        }
        w.bytes()
    }

    /// Sequence header OBU'su (boyut alanı = 1, yük = `payload`).
    fn seq_header_obu(payload: &[u8]) -> Vec<u8> {
        let mut packet = vec![0x0A];
        write_uleb128(payload.len() as u64, &mut packet);
        packet.extend_from_slice(payload);
        packet
    }

    fn write_uleb128(mut value: u64, out: &mut Vec<u8>) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    #[test]
    fn frame_size_limit_sinir_semantigi() {
        use super::frame_size_exceeds;
        // 0 = sınırsız (dav1d varsayılanı).
        assert!(!frame_size_exceeds(65536, 65536, 0));
        // Tam sınır kabul, bir fazlası ret (4096*4096).
        assert!(!frame_size_exceeds(4096, 4096, 4096 * 4096));
        assert!(frame_size_exceeds(4097, 4096, 4096 * 4096));
        // Ölçülen 22 zaman aşımı karesi (test7740 paket 4) sınırı aşar.
        assert!(frame_size_exceeds(13312, 3040, 4096 * 4096));
    }

    /// P38 taşma sınıfı: kare alanı `u64`'te doygun çarpımla hesaplanır;
    /// AV1 sınırı (65 536²) ve `i32` uçları taşmaz.
    #[test]
    fn tasma_kare_alani_ve_tile_boyutu_tasmaz() {
        use super::{TileGroupRaw, frame_size_exceeds, split_tile_groups};
        use crate::bitstream::frame_header::FrameHeader;
        assert!(frame_size_exceeds(65536, 65536, 4096 * 4096));
        // (2^31 − 1)² ≈ 2^62: `u64`'e sığar.
        assert!(frame_size_exceeds(i32::MAX, i32::MAX, 1 << 61));
        assert!(!frame_size_exceeds(i32::MAX, i32::MAX, u64::MAX));
        assert!(!frame_size_exceeds(i32::MIN, 0, 1));
        // 4 baytlık tile boyutu alanının en büyük değeri (2^32 bayt) taşmaz,
        // kalan veriyi aştığı için hata olur.
        let mut h = FrameHeader::default();
        h.tiling.cols = 2;
        h.tiling.rows = 1;
        h.tiling.n_bytes = 4;
        let groups = [TileGroupRaw {
            start: 0,
            end: 1,
            data: vec![0xFF, 0xFF, 0xFF, 0xFF, 0xA0],
        }];
        assert!(split_tile_groups(&groups, &h).is_err());
    }

    #[test]
    fn frame_size_limit_yapilandirmada_korunur() {
        let config = Config {
            frame_size_limit: 4096 * 4096,
            ..Config::default()
        };
        assert_eq!(config.validate(), Ok(()));
        assert_eq!(config.frame_size_limit, 16_777_216);
        assert_eq!(Config::default().frame_size_limit, 0);
    }

    #[test]
    fn kurulum_yapilandirmayi_dogrular() {
        assert!(Decoder::new(Config::default()).is_ok());

        let bad_budget = Config {
            allocation_limit: Some(0),
            ..Config::default()
        };
        assert!(matches!(
            Decoder::new(bad_budget),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn kurulumda_cpu_bayraklari_ve_dsp_bir_kez_belirlenir() {
        let decoder = Decoder::new(Config {
            cpu_mask: CpuFlags::AVX2,
            ..Config::default()
        })
        .expect("kurulum başarılı olmalı");

        assert!(
            !decoder.cpu_flags().contains(CpuFlags::AVX2),
            "kullanıcı maskesi AVX2 yolunu kapatmalı"
        );
        assert_eq!(
            decoder.cpu_flags(),
            CpuFlags::detect().difference(CpuFlags::AVX2)
        );
        // P30: AVX2 kapalıyken hiçbir yuva AVX2 gövdesi tutmaz; diğer
        // düzeylerin (SSSE3, SSE4.1, AVX512ICL) atamaları planla aynıdır.
        let states = decoder.dsp().slot_states();
        assert!(
            states.iter().all(|s| s.implementation != "avx2"),
            "AVX2 kapalıyken AVX2 yuvası olmamalı"
        );
        assert_eq!(
            states,
            crate::dsp::DspDispatch::planned_slot_states(decoder.cpu_flags())
        );
    }

    /// F3a: dav1d `--cpumask` düzeyleri çözücü kurulumunda uygulanır; `none`
    /// düzeyinde hiçbir ISA yuvası atanmaz ve etkin çekirdekler skaler
    /// gövdelerdir (zorlanmış skaler kip).
    #[test]
    fn cpu_duzeyi_kurulumda_uygulanir() {
        use crate::cpu::CpuLevel;
        let detected = CpuFlags::detect();
        for level in CpuLevel::ALL {
            let decoder = Decoder::new(Config {
                cpu_mask: level.cpu_mask(),
                ..Config::default()
            })
            .expect("kurulum başarılı olmalı");
            assert_eq!(decoder.cpu_flags(), detected.intersection(level.allowed()));
            assert_eq!(
                decoder.dsp(),
                &DspDispatch::select(detected.intersection(level.allowed()))
            );
        }
        let skaler = Decoder::new(Config {
            cpu_mask: CpuLevel::None.cpu_mask(),
            ..Config::default()
        })
        .expect("kurulum başarılı olmalı");
        assert!(skaler.cpu_flags().is_empty());
        assert_eq!(skaler.dsp().assigned_count(), 0);
        // P30: yuva kimliği seçim kaydıyla (işlev adresi karşılaştırılmaz).
        for field in ["dir", "fb[0]", "fb[1]", "fb[2]"] {
            assert_eq!(
                skaler
                    .dsp()
                    .slot_implementation(crate::frame::BitDepth::Eight, "cdef", field),
                "skaler"
            );
        }
        let tam = Decoder::new(Config::default()).expect("kurulum başarılı olmalı");
        assert_eq!(tam.cpu_flags(), detected);
        // P30: yuva başına en çok bir aday atanır (düzeyler üst üste yazar).
        assert_eq!(
            tam.dsp().assigned_count(),
            crate::dsp::DspDispatch::planned_slot_states(detected)
                .iter()
                .filter(|s| s.is_assigned())
                .count()
        );
    }

    #[test]
    fn durum_makinesi_flush_ve_reset() {
        let mut decoder = Decoder::new(Config::default()).expect("kurulum başarılı olmalı");
        assert_eq!(decoder.state(), DecoderState::Ready);

        decoder.flush();
        assert_eq!(decoder.state(), DecoderState::Draining);
        decoder.flush();
        assert_eq!(decoder.state(), DecoderState::Draining, "flush idempotent");

        assert_eq!(
            decoder.send_packet(&Packet::new(&[0x0A, 0x0B])),
            Err(Error::InvalidState(
                "flush sonrası send_packet: önce reset() çağrılmalı"
            ))
        );

        decoder.reset();
        assert_eq!(decoder.state(), DecoderState::Ready);

        // reset sonrası hata artık durum hatası değil, bit akışı hatasıdır.
        assert!(matches!(
            decoder.send_packet(&Packet::new(&[0x0A, 0x0B])),
            Err(Error::Bitstream(_))
        ));
    }

    #[test]
    fn sequence_header_paketten_ayristirilip_saklanir() {
        let payload = reduced_seq_header_payload();
        let mut decoder = Decoder::new(Config::default()).expect("kurulum");
        assert!(decoder.sequence_header().is_none());
        decoder
            .send_packet(&Packet::new(&seq_header_obu(&payload)))
            .expect("sequence header kabul edilmeli");
        let header = decoder.sequence_header().expect("saklandı");
        assert_eq!(header.profile, 0);
        assert!(header.still_picture);
        assert_eq!(header.max_width, 2);
        assert_eq!(header.max_height, 2);
        assert_eq!(decoder.sequence_header_updates(), 1);
        assert_eq!(decoder.operating_point_idc(), 0);
        // Bu paket kare tamamlamaz: çıktı yok.
        assert!(matches!(decoder.receive_frame(), Ok(None)));
    }

    #[test]
    fn obu_has_size_field_sifir_ve_coklu_obu() {
        let payload = reduced_seq_header_payload();
        // TD + seq header (boyut alanı YOK: paketin kalanını kaplar).
        let mut packet = vec![0x12, 0x00, 0x08];
        packet.extend_from_slice(&payload);
        let mut decoder = Decoder::new(Config::default()).expect("kurulum");
        decoder
            .send_packet(&Packet::new(&packet))
            .expect("boyut alanı olmayan OBU");
        assert!(decoder.sequence_header().is_some());

        // Aynı pakette iki sequence header: ikisi de uygulanır, sonuncusu
        // etkin olur (dav1d aynı davranış: her sequence header işlenir).
        let mut packet = seq_header_obu(&payload);
        packet.extend_from_slice(&seq_header_obu(&payload));
        decoder
            .send_packet(&Packet::new(&packet))
            .expect("iki sequence header");
        assert_eq!(decoder.sequence_header_updates(), 3);
    }

    #[test]
    fn operating_point_katman_suzgeci_calisir() {
        // idc = 0x101: yalnız temporal 0 + spatial 0 katmanları seçili.
        let header = seq_header_obu(&seq_header_payload_with_idc(0x101));
        let mut decoder = Decoder::new(Config::default()).expect("kurulum");
        decoder
            .send_packet(&Packet::new(&header))
            .expect("sequence header");
        assert_eq!(decoder.operating_point_idc(), 0x101);

        // Uzantılı frame OBU (tip 6), spatial_id=1 -> süzülür.
        // 0x34 = 0 0110 1 0 0: frame, extension var, boyut alanı yok.
        // Uzantı baytı 0b0000_1000: temporal_id=0, spatial_id=1.
        let filtered = [0x34, 0b0000_1000];
        decoder
            .send_packet(&Packet::new(&filtered))
            .expect("süzme hatasız");
        assert_eq!(decoder.layer_obus_skipped(), 1);

        // temporal_id=0, spatial_id=0 -> katmanda; atlanmaz. Metadata OBU
        // (tip 5): `metadata_type` 6 (kayıtsız) dav1d gibi yok sayılır.
        // 0x2C = 0 0101 1 0 0: metadata, extension var, boyut alanı yok.
        let kept = [0x2C, 0b0000_0000, 0x06];
        decoder.send_packet(&Packet::new(&kept)).expect("katmanda");
        assert_eq!(decoder.layer_obus_skipped(), 1);
    }

    /// dav1d `gen_picture` (`src/lib.c:409-417`): hatalı OBU'dan **önceki**
    /// OBU'ların etkisi kalır, paketin kalanı atılır.
    #[test]
    fn bozuk_paket_hataya_kadar_uygulanir() {
        let payload = reduced_seq_header_payload();
        let mut decoder = Decoder::new(Config::default()).expect("kurulum");
        decoder
            .send_packet(&Packet::new(&seq_header_obu(&payload)))
            .expect("sequence header");
        let before = decoder.sequence_header().cloned();

        // İkinci sequence header + sonunda boyutu taşan OBU: ilk OBU
        // uygulanır, hata döner.
        let mut packet = seq_header_obu(&payload);
        packet.extend_from_slice(&[0x0A, 0x05]); // boyut 5 > kalan 0
        assert!(matches!(
            decoder.send_packet(&Packet::new(&packet)),
            Err(Error::Bitstream(_))
        ));
        assert_eq!(decoder.sequence_header(), before.as_ref());
        assert_eq!(decoder.sequence_header_updates(), 2);

        // Yasak bit (`obu_forbidden_bit`) dav1d kütüphane varsayılanında
        // (`strict_std_compliance = 0`) yok sayılır; katı uyumda hatadır.
        assert_eq!(decoder.send_packet(&Packet::new(&[0x80])), Ok(()));
        let mut strict = Decoder::new(Config {
            strict_std_compliance: true,
            ..Config::default()
        })
        .expect("kurulum");
        assert!(matches!(
            strict.send_packet(&Packet::new(&[0x80])),
            Err(Error::Bitstream(_))
        ));
    }

    /// `Clone` bağımsız kopya verir: kopyaya gönderilen paket aslını etkilemez.
    #[test]
    fn klon_bagimsiz_durum_tasir() {
        let payload = reduced_seq_header_payload();
        let mut decoder = Decoder::new(Config::default()).expect("kurulum");
        decoder
            .send_packet(&Packet::new(&seq_header_obu(&payload)))
            .expect("sequence header");
        let mut fork = decoder.clone();
        assert_eq!(fork.sequence_header(), decoder.sequence_header());
        fork.send_packet(&Packet::new(&seq_header_obu(&payload)))
            .expect("ikinci sequence header");
        assert_eq!(fork.sequence_header_updates(), 2);
        assert_eq!(decoder.sequence_header_updates(), 1);
        fork.reset();
        assert!(fork.sequence_header().is_none());
        assert!(decoder.sequence_header().is_some());
    }

    #[test]
    fn reset_ayristirma_durumunu_temizler() {
        let payload = reduced_seq_header_payload();
        let mut decoder = Decoder::new(Config::default()).expect("kurulum");
        decoder
            .send_packet(&Packet::new(&seq_header_obu(&payload)))
            .expect("sequence header");
        decoder.reset();
        assert!(decoder.sequence_header().is_none());
        assert_eq!(decoder.sequence_header_updates(), 0);
        assert_eq!(decoder.operating_point_idc(), 0);
        assert_eq!(decoder.layer_obus_skipped(), 0);
        assert_eq!(decoder.state(), DecoderState::Ready);
    }

    #[test]
    fn bos_paket_ve_yalniz_td_paketi_hatasizdir() {
        let mut decoder = Decoder::new(Config::default()).expect("kurulum");
        decoder.send_packet(&Packet::new(&[])).expect("boş paket");
        decoder
            .send_packet(&Packet::new(&[0x12, 0x00]))
            .expect("TD");
        assert!(decoder.sequence_header().is_none());
        assert_eq!(decoder.sequence_header_updates(), 0);
    }

    #[test]
    fn cozme_yollari_panik_uretmez() {
        let mut decoder = Decoder::new(Config::default()).expect("kurulum başarılı olmalı");
        // Sequence header OBU'su, boyut alanı kalan bayttan büyük.
        let result: Result<()> = decoder.send_packet(&Packet::new(&[0x0A, 0x05]));
        assert!(matches!(result, Err(Error::Bitstream(_))));
        assert!(matches!(decoder.receive_frame(), Ok(None)));

        decoder.flush();
        assert!(matches!(decoder.receive_frame(), Ok(None)));
    }

    /// dav1d `dav1d_decode_frame_init_cdf` (`src/decode.c:3159-3189`): tile
    /// boyutları çözme aşamasında ayrıştırılır; taşma **geç** hatadır.
    #[test]
    fn tile_gruplari_dav1d_gibi_bolunur() {
        use super::{TileGroupRaw, split_tile_groups};
        use crate::bitstream::frame_header::FrameHeader;
        let mut h = FrameHeader::default();
        h.tiling.cols = 2;
        h.tiling.rows = 1;
        h.tiling.n_bytes = 1;
        let group = TileGroupRaw {
            start: 0,
            end: 1,
            data: vec![0x01, 0xA0, 0xA1, 0xB0],
        };
        let groups = [group];
        let tiles = split_tile_groups(&groups, &h).expect("bölünür");
        assert_eq!(tiles, vec![&[0xA0u8, 0xA1][..], &[0xB0][..]]);
        let over = TileGroupRaw {
            start: 0,
            end: 1,
            data: vec![0x05, 0xA0],
        };
        assert!(split_tile_groups(&[over], &h).is_err());
        let short = TileGroupRaw {
            start: 0,
            end: 1,
            data: Vec::new(),
        };
        assert!(split_tile_groups(&[short], &h).is_err());
    }

    /// dav1d `src/obu.c:1356-1515`: metadata türü leb128'i ve HDR CLL/MDCV
    /// yükü denetlenir; bilinmeyen türler yok sayılır.
    #[test]
    fn metadata_dav1d_gibi_denetlenir() {
        use super::validate_metadata;
        use crate::bitstream::split_next;
        let check = |payload: &[u8]| {
            let mut packet = vec![0x2A, payload.len() as u8];
            packet.extend_from_slice(payload);
            let mut offset = 0;
            let obu = split_next(&packet, &mut offset)
                .expect("OBU")
                .expect("OBU var");
            validate_metadata(&obu, false)
        };
        assert!(check(&[0x01, 0, 0, 0, 0, 0x80]).is_ok());
        assert!(check(&[0x01, 0, 0]).is_err());
        assert!(check(&[0x06]).is_ok());
        assert!(check(&[]).is_err());
        let mut mdcv = vec![0x02];
        mdcv.extend_from_slice(&[0u8; 24]);
        mdcv.push(0x80);
        assert!(check(&mdcv).is_ok());
        assert!(check(&mdcv[..20]).is_err());
    }

    #[test]
    fn decoder_send_ve_clone_sozlesmesini_saglar() {
        fn assert_send<T: Send>() {}
        assert_send::<Decoder>();
        let d = Decoder::new(Config {
            n_threads: 2,
            ..Config::default()
        })
        .expect("2 iş parçacıklı kurulum");
        let d2 = d.clone();
        assert_eq!(d2.config().n_threads, 2);
    }

    #[test]
    fn uzun_omurlu_decoder_varsayilan_kipte_sabit_bellek_tutar() {
        // 1x1 8-bit 4:2:0 AV1 anahtar kare (TD + Sequence Header + Keyframe).
        const PKT_1X1: &[u8] = &[
            0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32,
            0x17, 0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68,
            0xD2, 0x16, 0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
        ];

        // 1. Varsayılan kipte (`collect_diagnostics == false`): 1000 karelik döngüde
        //    `kalici_bayt()` ve `allocation_peak()` sabit kalır.
        let mut probe = Decoder::new(Config {
            n_threads: 1,
            max_frame_delay: 1,
            ..Config::default()
        })
        .unwrap();
        assert!(!probe.collect_diagnostics());
        for _ in 0..10 {
            probe.send_packet(&Packet::new(PKT_1X1)).unwrap();
            let _ = probe.receive_frame().unwrap().unwrap();
        }
        let steady_peak = probe.allocation_peak();
        let steady_kalici = probe.kalici_bayt();

        let mut dec = Decoder::new(Config {
            n_threads: 1,
            max_frame_delay: 1,
            allocation_limit: Some(steady_peak),
            ..Config::default()
        })
        .unwrap();
        for _ in 0..1_000 {
            dec.send_packet(&Packet::new(PKT_1X1))
                .expect("sabit bütçe altında 1000 kare çözülmeli");
            let frame = dec.receive_frame().unwrap().expect("kare üretilmeli");
            assert_eq!((frame.width(), frame.height()), (1, 1));
        }
        assert_eq!(dec.decoded_frames(), 1_000);
        assert_eq!(dec.frames_completed(), 1_000);
        assert!(dec.last_tool_counts().is_some());
        assert_eq!(dec.pending_frame_headers(), 0);
        assert!(dec.decoded_tool_counts().is_empty());
        assert_eq!(dec.kalici_bayt(), steady_kalici);
        assert!(dec.allocation_peak() <= steady_peak);

        // 2. `set_collect_diagnostics(true)` açıldığında geçmiş kayıtları çalışır.
        let mut dec_diag = Decoder::new(Config {
            n_threads: 1,
            max_frame_delay: 1,
            ..Config::default()
        })
        .unwrap();
        dec_diag.set_collect_diagnostics(true);
        dec_diag.send_packet(&Packet::new(PKT_1X1)).unwrap();
        let _ = dec_diag.receive_frame().unwrap().unwrap();
        assert_eq!(dec_diag.pending_frame_headers(), 1);
        assert_eq!(dec_diag.drain_frame_headers().len(), 1);
        assert_eq!(dec_diag.drain_frame_output_flags(), vec![true]);
        assert_eq!(dec_diag.decoded_tool_counts().len(), 1);
    }
}
