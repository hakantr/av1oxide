//! Rust ↔ C (`Dav1d*`) başlık, ayar ve hata dönüşümleri.

#![forbid(unsafe_code)]

use core::ffi::{c_int, c_uint};

use crate::api::{Config, DecodeFrameType, InloopFilters};
use crate::bitstream::frame_header::{FilmGrainData, FilmGrainHeader, FrameHeader};
use crate::bitstream::{ObuType, SequenceHeader, SequenceHeaderParseOptions, split_next_with};
use crate::capi::types::{
    DAV1D_DECODEFRAMETYPE_ALL, DAV1D_DECODEFRAMETYPE_INTRA, DAV1D_DECODEFRAMETYPE_KEY,
    DAV1D_DECODEFRAMETYPE_REFERENCE, DAV1D_INLOOPFILTER_ALL, DAV1D_INLOOPFILTER_CDEF,
    DAV1D_INLOOPFILTER_DEBLOCK, DAV1D_INLOOPFILTER_RESTORATION, DAV1D_MAX_CDEF_STRENGTHS,
    DAV1D_MAX_FRAME_DELAY, DAV1D_MAX_OPERATING_POINTS, DAV1D_MAX_SEGMENTS, DAV1D_MAX_THREADS,
    DAV1D_MAX_TILE_COLS, DAV1D_MAX_TILE_ROWS, DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420,
    DAV1D_PIXEL_LAYOUT_I422, DAV1D_PIXEL_LAYOUT_I444, DAV1D_REFS_PER_FRAME, Dav1dDecodeFrameType,
    Dav1dFilmGrainData, Dav1dFrameHeader, Dav1dFrameHeaderCdef, Dav1dFrameHeaderDelta,
    Dav1dFrameHeaderDeltaLf, Dav1dFrameHeaderDeltaQ, Dav1dFrameHeaderFilmGrain,
    Dav1dFrameHeaderLoopfilter, Dav1dFrameHeaderOperatingPoint, Dav1dFrameHeaderQuant,
    Dav1dFrameHeaderRestoration, Dav1dFrameHeaderSegmentation, Dav1dFrameHeaderSuperRes,
    Dav1dFrameHeaderTiling, Dav1dInloopFilterType, Dav1dLoopfilterModeRefDeltas, Dav1dPixelLayout,
    Dav1dSegmentationData, Dav1dSegmentationDataSet, Dav1dSequenceHeader,
    Dav1dSequenceHeaderOperatingParameterInfo, Dav1dSequenceHeaderOperatingPoint, Dav1dSettings,
    Dav1dWarpedMotionParams, EINVAL, ENOENT, ENOMEM, ERANGE, dav1d_err,
};
use crate::decode::warpv::get_shear_params;
use crate::error::{BitstreamErrorKind, Error};
use crate::frame::{BitDepth, ChromaFormat};

/// `dav1d_get_frame_delay` için `Dav1dSettings` doğrulaması ve etkin kare gecikmesi hesabı.
pub(crate) fn frame_delay_from_settings(s: &Dav1dSettings) -> Result<c_int, c_int> {
    if s.n_threads < 0
        || s.n_threads > DAV1D_MAX_THREADS
        || s.max_frame_delay < 0
        || s.max_frame_delay > DAV1D_MAX_FRAME_DELAY
    {
        return Err(dav1d_err(EINVAL));
    }
    let cfg = Config {
        n_threads: s.n_threads as usize,
        max_frame_delay: s.max_frame_delay as usize,
        ..Config::default()
    };
    Ok(cfg.effective_frame_delay() as c_int)
}

/// `Dav1dInloopFilterType` → [`InloopFilters`].
pub(crate) const fn inloop_filters_from_c(mask: Dav1dInloopFilterType) -> InloopFilters {
    let mut bits = 0u8;
    if (mask & DAV1D_INLOOPFILTER_DEBLOCK) != 0 {
        bits |= InloopFilters::DEBLOCK.bits();
    }
    if (mask & DAV1D_INLOOPFILTER_CDEF) != 0 {
        bits |= InloopFilters::CDEF.bits();
    }
    if (mask & DAV1D_INLOOPFILTER_RESTORATION) != 0 {
        bits |= InloopFilters::RESTORATION.bits();
    }
    InloopFilters::from_bits_truncate(bits)
}

/// `Dav1dDecodeFrameType` → [`DecodeFrameType`].
pub(crate) const fn decode_frame_type_from_c(ft: Dav1dDecodeFrameType) -> Option<DecodeFrameType> {
    match ft {
        DAV1D_DECODEFRAMETYPE_ALL => Some(DecodeFrameType::All),
        DAV1D_DECODEFRAMETYPE_REFERENCE => Some(DecodeFrameType::Reference),
        DAV1D_DECODEFRAMETYPE_INTRA => Some(DecodeFrameType::Intra),
        DAV1D_DECODEFRAMETYPE_KEY => Some(DecodeFrameType::Key),
        _ => None,
    }
}

/// `Dav1dSettings` yapısını doğrular ve `av1oxide::Config`'e dönüştürür.
pub(crate) fn settings_to_config(s: &Dav1dSettings) -> Result<Config, c_int> {
    if s.n_threads < 0
        || s.n_threads > DAV1D_MAX_THREADS
        || s.max_frame_delay < 0
        || s.max_frame_delay > DAV1D_MAX_FRAME_DELAY
        || s.operating_point < 0
        || s.operating_point > 31
        || (s.inloop_filters & !DAV1D_INLOOPFILTER_ALL) != 0
        || decode_frame_type_from_c(s.decode_frame_type).is_none()
    {
        return Err(dav1d_err(EINVAL));
    }
    let has_alloc = s.allocator.alloc_picture_callback.is_some();
    let has_release = s.allocator.release_picture_callback.is_some();
    if has_alloc != has_release {
        return Err(dav1d_err(EINVAL));
    }
    if !has_alloc && !s.allocator.cookie.is_null() {
        return Err(dav1d_err(EINVAL));
    }

    let config = Config {
        n_threads: s.n_threads as usize,
        max_frame_delay: s.max_frame_delay as usize,
        apply_grain: s.apply_grain != 0,
        operating_point: s.operating_point as u8,
        frame_size_limit: u64::from(s.frame_size_limit),
        strict_std_compliance: s.strict_std_compliance != 0,
        output_invisible_frames: s.output_invisible_frames != 0,
        ..Config::default()
    };
    config.validate().map_err(|_| dav1d_err(EINVAL))?;
    Ok(config)
}

/// `av1oxide::Error` değerini negatif `DAV1D_ERR(...)` C hata koduna dönüştürür.
pub(crate) fn error_to_dav1d(err: &Error) -> c_int {
    match err {
        Error::AllocationFailed { .. }
        | Error::BudgetExceeded { .. }
        | Error::SizeOverflow { .. } => dav1d_err(ENOMEM),
        Error::Bitstream(b) => match b.kind {
            BitstreamErrorKind::InvalidFrameHeader(msg) if msg.contains("frame_size_limit") => {
                dav1d_err(ERANGE)
            }
            _ => dav1d_err(EINVAL),
        },
        Error::InvalidArgument(_)
        | Error::InvalidData(_)
        | Error::InvalidState(_)
        | Error::Unsupported(_) => dav1d_err(EINVAL),
    }
}

/// `ChromaFormat` → `Dav1dPixelLayout`.
pub(crate) const fn chroma_to_pixel_layout(chroma: ChromaFormat) -> Dav1dPixelLayout {
    match chroma {
        ChromaFormat::Monochrome => DAV1D_PIXEL_LAYOUT_I400,
        ChromaFormat::C420 => DAV1D_PIXEL_LAYOUT_I420,
        ChromaFormat::C422 => DAV1D_PIXEL_LAYOUT_I422,
        ChromaFormat::C444 => DAV1D_PIXEL_LAYOUT_I444,
    }
}

/// `Dav1dPixelLayout` → `ChromaFormat`.
pub(crate) const fn pixel_layout_to_chroma(layout: Dav1dPixelLayout) -> Option<ChromaFormat> {
    match layout {
        DAV1D_PIXEL_LAYOUT_I400 => Some(ChromaFormat::Monochrome),
        DAV1D_PIXEL_LAYOUT_I420 => Some(ChromaFormat::C420),
        DAV1D_PIXEL_LAYOUT_I422 => Some(ChromaFormat::C422),
        DAV1D_PIXEL_LAYOUT_I444 => Some(ChromaFormat::C444),
        _ => None,
    }
}

/// `bpc` (`8`, `10`, `12`) → `BitDepth`.
pub(crate) const fn bpc_to_bit_depth(bpc: c_int) -> Option<BitDepth> {
    match bpc {
        8 => Some(BitDepth::Eight),
        10 => Some(BitDepth::Ten),
        12 => Some(BitDepth::Twelve),
        _ => None,
    }
}

/// `SequenceHeader` → `Dav1dSequenceHeader`.
pub(crate) fn seq_header_to_c(src: &SequenceHeader) -> Dav1dSequenceHeader {
    let mut operating_points =
        [Dav1dSequenceHeaderOperatingPoint::default(); DAV1D_MAX_OPERATING_POINTS];
    for (dst, op) in operating_points.iter_mut().zip(src.operating_points.iter()) {
        *dst = Dav1dSequenceHeaderOperatingPoint {
            major_level: op.major_level,
            minor_level: op.minor_level,
            initial_display_delay: op.initial_display_delay,
            idc: op.idc,
            tier: op.tier,
            decoder_model_param_present: u8::from(op.decoder_model_param_present),
            display_model_param_present: u8::from(op.display_model_param_present),
        };
    }

    let mut operating_parameter_info =
        [Dav1dSequenceHeaderOperatingParameterInfo::default(); DAV1D_MAX_OPERATING_POINTS];
    for (dst, info) in operating_parameter_info
        .iter_mut()
        .zip(src.operating_parameter_info.iter())
    {
        *dst = Dav1dSequenceHeaderOperatingParameterInfo {
            decoder_buffer_delay: info.decoder_buffer_delay,
            encoder_buffer_delay: info.encoder_buffer_delay,
            low_delay_mode: u8::from(info.low_delay_mode),
        };
    }

    Dav1dSequenceHeader {
        profile: src.profile,
        max_width: src.max_width,
        max_height: src.max_height,
        layout: c_uint::from(src.layout.raw()),
        pri: c_uint::from(src.pri),
        trc: c_uint::from(src.trc),
        mtrx: c_uint::from(src.mtrx),
        chr: c_uint::from(src.chr.raw()),
        hbd: src.hbd,
        color_range: src.color_range,
        num_operating_points: src.num_operating_points,
        operating_points,
        still_picture: u8::from(src.still_picture),
        reduced_still_picture_header: u8::from(src.reduced_still_picture_header),
        timing_info_present: u8::from(src.timing_info_present),
        num_units_in_tick: src.num_units_in_tick,
        time_scale: src.time_scale,
        equal_picture_interval: u8::from(src.equal_picture_interval),
        num_ticks_per_picture: src.num_ticks_per_picture,
        decoder_model_info_present: u8::from(src.decoder_model_info_present),
        encoder_decoder_buffer_delay_length: src.encoder_decoder_buffer_delay_length,
        num_units_in_decoding_tick: src.num_units_in_decoding_tick,
        buffer_removal_delay_length: src.buffer_removal_delay_length,
        frame_presentation_delay_length: src.frame_presentation_delay_length,
        display_model_info_present: u8::from(src.display_model_info_present),
        width_n_bits: src.width_n_bits,
        height_n_bits: src.height_n_bits,
        frame_id_numbers_present: u8::from(src.frame_id_numbers_present),
        delta_frame_id_n_bits: src.delta_frame_id_n_bits,
        frame_id_n_bits: src.frame_id_n_bits,
        sb128: u8::from(src.sb128),
        filter_intra: u8::from(src.filter_intra),
        intra_edge_filter: u8::from(src.intra_edge_filter),
        inter_intra: u8::from(src.inter_intra),
        masked_compound: u8::from(src.masked_compound),
        warped_motion: u8::from(src.warped_motion),
        dual_filter: u8::from(src.dual_filter),
        order_hint: u8::from(src.order_hint),
        jnt_comp: u8::from(src.jnt_comp),
        ref_frame_mvs: u8::from(src.ref_frame_mvs),
        screen_content_tools: c_uint::from(src.screen_content_tools.raw()),
        force_integer_mv: c_uint::from(src.force_integer_mv.raw()),
        order_hint_n_bits: src.order_hint_n_bits,
        super_res: u8::from(src.super_res),
        cdef: u8::from(src.cdef),
        restoration: u8::from(src.restoration),
        ss_hor: src.ss_hor,
        ss_ver: src.ss_ver,
        monochrome: u8::from(src.monochrome),
        color_description_present: u8::from(src.color_description_present),
        separate_uv_delta_q: u8::from(src.separate_uv_delta_q),
        film_grain_present: u8::from(src.film_grain_present),
        operating_parameter_info,
    }
}

/// `FilmGrainHeader` → `Dav1dFrameHeaderFilmGrain`.
pub(crate) fn film_grain_to_c(src: &FilmGrainHeader) -> Dav1dFrameHeaderFilmGrain {
    let d = &src.data;
    Dav1dFrameHeaderFilmGrain {
        data: Dav1dFilmGrainData {
            seed: d.seed,
            num_y_points: d.num_y_points,
            y_points: d.y_points,
            chroma_scaling_from_luma: c_int::from(d.chroma_scaling_from_luma),
            num_uv_points: d.num_uv_points,
            uv_points: d.uv_points,
            scaling_shift: d.scaling_shift,
            ar_coeff_lag: d.ar_coeff_lag,
            ar_coeffs_y: d.ar_coeffs_y,
            ar_coeffs_uv: d.ar_coeffs_uv,
            ar_coeff_shift: d.ar_coeff_shift,
            grain_scale_shift: d.grain_scale_shift,
            uv_mult: d.uv_mult,
            uv_luma_mult: d.uv_luma_mult,
            uv_offset: d.uv_offset,
            overlap_flag: c_int::from(d.overlap_flag),
            clip_to_restricted_range: c_int::from(d.clip_to_restricted_range),
        },
        present: u8::from(src.present),
        update: u8::from(src.update),
    }
}

/// `Dav1dFrameHeaderFilmGrain` → `FilmGrainHeader`.
pub(crate) fn film_grain_from_c(src: &Dav1dFrameHeaderFilmGrain) -> FilmGrainHeader {
    let d = &src.data;
    FilmGrainHeader {
        present: src.present != 0,
        update: src.update != 0,
        data: FilmGrainData {
            seed: d.seed,
            num_y_points: d.num_y_points,
            y_points: d.y_points,
            chroma_scaling_from_luma: d.chroma_scaling_from_luma != 0,
            num_uv_points: d.num_uv_points,
            uv_points: d.uv_points,
            scaling_shift: d.scaling_shift,
            ar_coeff_lag: d.ar_coeff_lag,
            ar_coeffs_y: d.ar_coeffs_y,
            ar_coeffs_uv: d.ar_coeffs_uv,
            ar_coeff_shift: d.ar_coeff_shift,
            grain_scale_shift: d.grain_scale_shift,
            uv_mult: d.uv_mult,
            uv_luma_mult: d.uv_luma_mult,
            uv_offset: d.uv_offset,
            overlap_flag: d.overlap_flag != 0,
            clip_to_restricted_range: d.clip_to_restricted_range != 0,
        },
    }
}

/// `dav1d` `has_grain()` kontrolü (`src/lib.c:303-309`).
pub(crate) const fn has_grain_c(fg: &Dav1dFilmGrainData) -> bool {
    fg.num_y_points != 0
        || fg.num_uv_points[0] != 0
        || fg.num_uv_points[1] != 0
        || (fg.clip_to_restricted_range != 0 && fg.chroma_scaling_from_luma != 0)
}

/// `FrameHeader` → `Dav1dFrameHeader`.
pub(crate) fn frame_header_to_c(src: &FrameHeader) -> Dav1dFrameHeader {
    let mut operating_points =
        [Dav1dFrameHeaderOperatingPoint::default(); DAV1D_MAX_OPERATING_POINTS];
    for (dst, op) in operating_points.iter_mut().zip(src.operating_points.iter()) {
        dst.buffer_removal_time = op.buffer_removal_time;
    }

    let mut seg_d = [Dav1dSegmentationData::default(); DAV1D_MAX_SEGMENTS];
    for (dst, s) in seg_d.iter_mut().zip(src.segmentation.seg_data.d.iter()) {
        *dst = Dav1dSegmentationData {
            delta_q: s.delta_q,
            delta_lf_y_v: s.delta_lf_y_v,
            delta_lf_y_h: s.delta_lf_y_h,
            delta_lf_u: s.delta_lf_u,
            delta_lf_v: s.delta_lf_v,
            r#ref: s.reference,
            skip: u8::from(s.skip),
            globalmv: u8::from(s.globalmv),
        };
    }
    let mut lossless = [0u8; DAV1D_MAX_SEGMENTS];
    for (dst, &flag) in lossless.iter_mut().zip(src.segmentation.lossless.iter()) {
        *dst = u8::from(flag);
    }

    let mut gmv = [Dav1dWarpedMotionParams::default(); DAV1D_REFS_PER_FRAME];
    for (dst, wm) in gmv.iter_mut().zip(src.gmv.iter()) {
        *dst = Dav1dWarpedMotionParams {
            r#type: c_uint::from(wm.kind.raw()),
            matrix: wm.matrix,
            abcd: get_shear_params(wm).unwrap_or([0; 4]),
        };
    }

    let mut col_start_sb = [0u16; DAV1D_MAX_TILE_COLS + 1];
    col_start_sb.copy_from_slice(&src.tiling.col_start_sb);
    let mut row_start_sb = [0u16; DAV1D_MAX_TILE_ROWS + 1];
    row_start_sb.copy_from_slice(&src.tiling.row_start_sb);

    let mut y_strength = [0u8; DAV1D_MAX_CDEF_STRENGTHS];
    y_strength.copy_from_slice(&src.cdef.y_strength);
    let mut uv_strength = [0u8; DAV1D_MAX_CDEF_STRENGTHS];
    uv_strength.copy_from_slice(&src.cdef.uv_strength);

    Dav1dFrameHeader {
        film_grain: film_grain_to_c(&src.film_grain),
        frame_type: c_uint::from(src.frame_type.raw()),
        width: src.width,
        height: src.height,
        frame_offset: src.frame_offset,
        temporal_id: src.temporal_id,
        spatial_id: src.spatial_id,
        show_existing_frame: u8::from(src.show_existing_frame),
        existing_frame_idx: src.existing_frame_idx,
        frame_id: src.frame_id,
        frame_presentation_delay: src.frame_presentation_delay,
        show_frame: u8::from(src.show_frame),
        showable_frame: u8::from(src.showable_frame),
        error_resilient_mode: u8::from(src.error_resilient_mode),
        disable_cdf_update: u8::from(src.disable_cdf_update),
        allow_screen_content_tools: u8::from(src.allow_screen_content_tools),
        force_integer_mv: u8::from(src.force_integer_mv),
        frame_size_override: u8::from(src.frame_size_override),
        primary_ref_frame: src.primary_ref_frame,
        buffer_removal_time_present: u8::from(src.buffer_removal_time_present),
        operating_points,
        refresh_frame_flags: src.refresh_frame_flags,
        render_width: src.render_width,
        render_height: src.render_height,
        super_res: Dav1dFrameHeaderSuperRes {
            width_scale_denominator: src.super_res.width_scale_denominator,
            enabled: u8::from(src.super_res.enabled),
        },
        have_render_size: u8::from(src.have_render_size),
        allow_intrabc: u8::from(src.allow_intrabc),
        frame_ref_short_signaling: u8::from(src.frame_ref_short_signaling),
        refidx: src.refidx,
        hp: u8::from(src.hp),
        subpel_filter_mode: c_uint::from(src.subpel_filter_mode.raw()),
        switchable_motion_mode: u8::from(src.switchable_motion_mode),
        use_ref_frame_mvs: u8::from(src.use_ref_frame_mvs),
        refresh_context: u8::from(src.refresh_context),
        tiling: Dav1dFrameHeaderTiling {
            uniform: u8::from(src.tiling.uniform),
            n_bytes: src.tiling.n_bytes,
            min_log2_cols: src.tiling.min_log2_cols,
            max_log2_cols: src.tiling.max_log2_cols,
            log2_cols: src.tiling.log2_cols,
            cols: src.tiling.cols,
            min_log2_rows: src.tiling.min_log2_rows,
            max_log2_rows: src.tiling.max_log2_rows,
            log2_rows: src.tiling.log2_rows,
            rows: src.tiling.rows,
            col_start_sb,
            row_start_sb,
            update: src.tiling.update,
        },
        quant: Dav1dFrameHeaderQuant {
            yac: src.quant.yac,
            ydc_delta: src.quant.ydc_delta,
            udc_delta: src.quant.udc_delta,
            uac_delta: src.quant.uac_delta,
            vdc_delta: src.quant.vdc_delta,
            vac_delta: src.quant.vac_delta,
            qm: u8::from(src.quant.qm),
            qm_y: src.quant.qm_y,
            qm_u: src.quant.qm_u,
            qm_v: src.quant.qm_v,
        },
        segmentation: Dav1dFrameHeaderSegmentation {
            enabled: u8::from(src.segmentation.enabled),
            update_map: u8::from(src.segmentation.update_map),
            temporal: u8::from(src.segmentation.temporal),
            update_data: u8::from(src.segmentation.update_data),
            seg_data: Dav1dSegmentationDataSet {
                d: seg_d,
                preskip: u8::from(src.segmentation.seg_data.preskip),
                last_active_segid: src.segmentation.seg_data.last_active_segid,
            },
            lossless,
            qidx: src.segmentation.qidx,
        },
        delta: Dav1dFrameHeaderDelta {
            q: Dav1dFrameHeaderDeltaQ {
                present: u8::from(src.delta.q_present),
                res_log2: src.delta.q_res_log2,
            },
            lf: Dav1dFrameHeaderDeltaLf {
                present: u8::from(src.delta.lf_present),
                res_log2: src.delta.lf_res_log2,
                multi: u8::from(src.delta.lf_multi),
            },
        },
        all_lossless: u8::from(src.all_lossless),
        loopfilter: Dav1dFrameHeaderLoopfilter {
            level_y: src.loopfilter.level_y,
            level_u: src.loopfilter.level_u,
            level_v: src.loopfilter.level_v,
            mode_ref_delta_enabled: u8::from(src.loopfilter.mode_ref_delta_enabled),
            mode_ref_delta_update: u8::from(src.loopfilter.mode_ref_delta_update),
            mode_ref_deltas: Dav1dLoopfilterModeRefDeltas {
                mode_delta: src.loopfilter.mode_ref_deltas.mode_delta,
                ref_delta: src.loopfilter.mode_ref_deltas.ref_delta,
            },
            sharpness: src.loopfilter.sharpness,
        },
        cdef: Dav1dFrameHeaderCdef {
            damping: src.cdef.damping,
            n_bits: src.cdef.n_bits,
            y_strength,
            uv_strength,
        },
        restoration: Dav1dFrameHeaderRestoration {
            r#type: [
                c_uint::from(src.restoration.kind[0].raw()),
                c_uint::from(src.restoration.kind[1].raw()),
                c_uint::from(src.restoration.kind[2].raw()),
            ],
            unit_size: src.restoration.unit_size,
        },
        txfm_mode: c_uint::from(src.txfm_mode.raw()),
        switchable_comp_refs: u8::from(src.switchable_comp_refs),
        skip_mode_allowed: u8::from(src.skip_mode_allowed),
        skip_mode_enabled: u8::from(src.skip_mode_enabled),
        skip_mode_refs: src.skip_mode_refs,
        warp_motion: u8::from(src.warp_motion),
        reduced_txtp_set: u8::from(src.reduced_txtp_set),
        gmv,
    }
}

/// `dav1d_parse_sequence_header` için bayt diliminden Sequence Header OBU ayrıştırır.
///
/// Birden çok Sequence Header OBU'su varsa `dav1d` (`src/obu.c:302-339`) gibi
/// sonuncusunu döndürür; hiç yoksa `-ENOENT`, bozuksa `-EINVAL` döner.
pub(crate) fn parse_sequence_header_bytes(buf: &[u8]) -> Result<Dav1dSequenceHeader, c_int> {
    if buf.is_empty() || buf.len() > usize::MAX / 2 {
        return Err(dav1d_err(EINVAL));
    }
    let mut offset = 0usize;
    let mut last_seq = None;
    while offset < buf.len() {
        let Some(obu) = split_next_with(buf, &mut offset, false).map_err(|_| dav1d_err(EINVAL))?
        else {
            break;
        };
        if obu.header.obu_type == ObuType::SequenceHeader {
            let parsed = SequenceHeader::parse_with(
                obu.payload,
                SequenceHeaderParseOptions {
                    strict_std_compliance: false,
                },
            )
            .map_err(|_| dav1d_err(EINVAL))?;
            last_seq = Some(seq_header_to_c(&parsed));
        }
    }
    last_seq.ok_or(dav1d_err(ENOENT))
}
