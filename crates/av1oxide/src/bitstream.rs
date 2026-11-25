//! Bit akışı: bit okuyucu, OBU katmanı, sequence ve frame header (F2a / S2a.2).
//!
//! ## Sorumluluk
//!
//! - **Bit okuyucu** ([`reader`]): `f(n)`, `s(n)`, `leb128()`, `uvlc()`,
//!   `ns(max)`, `su(ref, n)`, bayt hizalama ve `trailing_bits`; tüm okumalar
//!   sınır denetimlidir ([`BitReader`]).
//! - **OBU katmanı** ([`obu`]): başlık + uzantı (temporal/spatial id) + boyut
//!   alanı; `obu_has_size_field = 0` dâhil. Tanınan tipler: temporal delimiter,
//!   sequence header, padding; metadata/tile list **yalnız sınırlarıyla atlanır**;
//!   frame header/frame/redundant frame header/tile group ayrıştırıcıya verilir.
//! - **Sequence header** ([`seq_header`]): `Dav1dSequenceHeader`'ın bütün
//!   anlamsal alanları, operating point'ler, timing/decoder model, renk
//!   yapılandırması, araç bayrakları ve film grain varlığı. Dav1d'nin reddettiği
//!   değerler aynı kurallarla reddedilir (kaynak satırı: [`seq_header`] doc).
//! - **Frame header** ([`frame_header`]): `Dav1dFrameHeader`'ın bütün alanları,
//!   tile bilgisi, global motion, film grain ve tile group başlığı/boyutları
//!   (tile verisi çözülmez). Referans başlıkları `crate::decode::refs` içinde
//!   tutulur.
//! - **Paket → OBU → durum** bağlanması [`crate::api::Decoder::send_packet`]
//!   içindedir: sequence header saklanır, seçilen operating point'e göre OBU
//!   süzülür, sonraki adıma ait OBU'lar sınırlarıyla atlanır.
//!
//! ## Upstream karşılığı (dav1d `99f3354`, salt-okunur)
//!
//! | Modül | dav1d kaynağı | AV1 spesifikasyonu |
//! |---|---|---|
//! | [`reader`] | `src/getbits.c`, `src/getbits.h` | §4.10 (temel ayrıştırma süreçleri), §5.3.5 |
//! | [`obu`] | `src/obu.c` (`dav1d_parse_obus` 1169+, `dav1d_parse_sequence_header` 302+) | §5.3 (OBU), §6.4 (OBU anlambilimi) |
//! | [`seq_header`] | `src/obu.c` (`parse_seq_hdr` 72–300), `include/dav1d/headers.h` (194–290) | §5.5 (sequence header OBU), §6.5 |
//! | [`frame_header`] | `src/obu.c` (`parse_frame_hdr` 409–1166, `parse_tile_hdr` 1157–1166), `include/dav1d/headers.h` (335–434) | §5.9–§5.10, §6.8–§6.10 |
//!
//! Bu modül C kodunu **çağırmaz**; davranış ve doğruluk hedeflenir (SRC-11).
//! Bit akışı **güvenilmeyen girdidir** (GUV-01): hiçbir okuma `unsafe`, panik
//! veya sonsuz döngü üretmez; her ayrıştırıcı adımı `Result` döner.
//!
//! ## Bu adımın sınırı (S2a.2)
//!
//! MSAC/CDF (S2a.3), tile/blok çözme ve kare üretimi (F2b) bu modülde
//! **yoktur**; tile verisi yalnız sınırlarıyla atlanır. `Decoder::receive_frame`
//! bu adımda kare üretmez ([`crate::api::Decoder`] dokümanı).

#![forbid(unsafe_code)]

pub mod frame_header;
pub mod obu;
pub mod reader;
pub mod seq_header;

pub use frame_header::{
    CdefParams, DEFAULT_MODE_REF_DELTAS, DEFAULT_WM_PARAMS, DeltaParams, FilmGrainData,
    FilmGrainHeader, FilterMode, FrameHeader, FrameHeaderContext, FrameHeaderOperatingPoint,
    FrameType, LoopFilterParams, LoopfilterModeRefDeltas, MAX_CDEF_STRENGTHS, MAX_SEGMENTS,
    MAX_TILE_COLS, MAX_TILE_ROWS, NUM_REF_FRAMES, PRIMARY_REF_NONE, QuantizationParams,
    REFS_PER_FRAME, RestorationParams, RestorationType, SegmentationData, SegmentationDataSet,
    SegmentationParams, SuperResParams, TOTAL_REFS_PER_FRAME, TileGroupInfo, TilingInfo, TxfmMode,
    WarpedMotionParams, WarpedMotionType, get_poc_diff, parse_tile_group,
};
pub use obu::{
    Obu, ObuExtension, ObuHeader, ObuType, parse_sequence_header_obus, split_next, split_next_with,
};
pub use reader::BitReader;
pub use seq_header::{
    AdaptiveBoolean, COLOR_PRI_BT709, COLOR_PRI_UNKNOWN, ChromaSamplePosition,
    MAX_OPERATING_POINTS, MC_IDENTITY, MC_UNKNOWN, OperatingParameterInfo, OperatingPoint,
    PixelLayout, SequenceHeader, SequenceHeaderParseOptions, TRC_SRGB, TRC_UNKNOWN,
};
