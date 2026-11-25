//! **ÜRETİLMİŞ DOSYA — elle düzenlenmez.**
//!
//! dav1d `src/cdf.h` yapılarının hücre (u16) cinsinden alan
//! ofsetleri. Üretici: `scripts/oracle/decode_tables_oracle.c`
//! (`offsetof`). Üretim crate'i ham `CdfContext` hücrelerini bu
//! ofsetlerle indeksler; S2a.3 ham yerleşimi bayt-bayt doğrular.

#![forbid(unsafe_code)]
#![allow(dead_code, non_upper_case_globals)]

/// `sizeof(CdfContext)` hücre sayısı.
pub(crate) const CDF_CONTEXT_CELLS: usize = 6736;
/// `sizeof(CdfCoefContext)` hücre sayısı.
pub(crate) const COEF_CONTEXT_CELLS: usize = 3056;

pub(crate) const COEF_eob_bin_16: usize = 0;
pub(crate) const COEF_eob_bin_32: usize = 32;
pub(crate) const COEF_eob_bin_64: usize = 64;
pub(crate) const COEF_eob_bin_128: usize = 96;
pub(crate) const COEF_eob_bin_256: usize = 128;
pub(crate) const COEF_eob_bin_512: usize = 192;
pub(crate) const COEF_eob_bin_1024: usize = 224;
pub(crate) const COEF_eob_base_tok: usize = 256;
pub(crate) const COEF_base_tok: usize = 416;
pub(crate) const COEF_br_tok: usize = 2056;
pub(crate) const COEF_eob_hi_bit: usize = 2728;
pub(crate) const COEF_skip: usize = 2908;
pub(crate) const COEF_dc_sign: usize = 3038;

pub(crate) const MODE_uv_mode: usize = 0;
pub(crate) const MODE_partition: usize = 416;
pub(crate) const MODE_cfl_alpha: usize = 736;
pub(crate) const MODE_txtp_inter1: usize = 832;
pub(crate) const MODE_txtp_inter2: usize = 864;
pub(crate) const MODE_txtp_inter3: usize = 2256;
pub(crate) const MODE_txtp_intra1: usize = 880;
pub(crate) const MODE_txtp_intra2: usize = 1088;
pub(crate) const MODE_cfl_sign: usize = 1400;
pub(crate) const MODE_angle_delta: usize = 1408;
pub(crate) const MODE_filter_intra: usize = 1472;
pub(crate) const MODE_seg_id: usize = 1480;
pub(crate) const MODE_pal_sz: usize = 1504;
pub(crate) const MODE_color_map: usize = 1616;
pub(crate) const MODE_txsz: usize = 2176;
pub(crate) const MODE_delta_q: usize = 2224;
pub(crate) const MODE_delta_lf: usize = 2228;
pub(crate) const MODE_restore_switchable: usize = 2248;
pub(crate) const MODE_restore_wiener: usize = 2252;
pub(crate) const MODE_restore_sgrproj: usize = 2254;
pub(crate) const MODE_use_filter_intra: usize = 2264;
pub(crate) const MODE_txpart: usize = 2308;
pub(crate) const MODE_skip: usize = 2350;
pub(crate) const MODE_pal_y: usize = 2356;
pub(crate) const MODE_pal_uv: usize = 2398;
pub(crate) const MODE_intrabc: usize = 2402;
pub(crate) const MODE_y_mode: usize = 2416;
pub(crate) const MODE_wedge_idx: usize = 2480;
pub(crate) const MODE_comp_inter_mode: usize = 2624;
pub(crate) const MODE_filter: usize = 2688;
pub(crate) const MODE_interintra_mode: usize = 2752;
pub(crate) const MODE_motion_mode: usize = 2768;
pub(crate) const MODE_skip_mode: usize = 2856;
pub(crate) const MODE_newmv_mode: usize = 2862;
pub(crate) const MODE_globalmv_mode: usize = 2874;
pub(crate) const MODE_refmv_mode: usize = 2878;
pub(crate) const MODE_drl_bit: usize = 2890;
pub(crate) const MODE_intra: usize = 2896;
pub(crate) const MODE_comp: usize = 2904;
pub(crate) const MODE_comp_dir: usize = 2914;
pub(crate) const MODE_jnt_comp: usize = 2924;
pub(crate) const MODE_mask_comp: usize = 2936;
pub(crate) const MODE_wedge_comp: usize = 2948;
pub(crate) const MODE_ref: usize = 2966;
pub(crate) const MODE_comp_fwd_ref: usize = 3002;
pub(crate) const MODE_comp_bwd_ref: usize = 3020;
pub(crate) const MODE_comp_uni_ref: usize = 3032;
pub(crate) const MODE_seg_pred: usize = 3050;
pub(crate) const MODE_interintra: usize = 3056;
pub(crate) const MODE_interintra_wedge: usize = 3070;
pub(crate) const MODE_obmc: usize = 3084;

pub(crate) const CDF_m: usize = 3056;
pub(crate) const CDF_kfym: usize = 6336;

pub(crate) const MVC_classes: usize = 0;
pub(crate) const MVC_sign: usize = 16;
pub(crate) const MVC_class0: usize = 18;
pub(crate) const MVC_class0_fp: usize = 20;
pub(crate) const MVC_class0_hp: usize = 28;
pub(crate) const MVC_classN: usize = 30;
pub(crate) const MVC_classN_fp: usize = 52;
pub(crate) const MVC_classN_hp: usize = 56;
pub(crate) const MV_comp: usize = 0;
pub(crate) const MV_joint: usize = 128;
