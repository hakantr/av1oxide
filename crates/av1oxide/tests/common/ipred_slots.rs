//! F3c: intra tahmin yuvaları üzerinden skaler girişler.
//!
//! Oracle ve sentinel testleri skaler gövdeleri çalışma zamanı bit
//! derinliğiyle çağırır; F3c'de girişler `const` bit derinliğiyle monomorf
//! olduğu için bu kısayollar girdinin bit derinliği tablosundaki yuvayı
//! (`IntraPredDsp::scalar_for`) çağırır. Böylece testler çözme hattının
//! kullandığı tablo girişlerini sınar.

#![allow(clippy::too_many_arguments)]

use av1oxide::dsp::ipred::{Edge, IntraPredDsp, IntraPredMode};
use av1oxide::frame::BitDepth;

/// `intra_pred[mode]`.
pub fn intra_pred(
    mode: IntraPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let f = IntraPredDsp::scalar_for(bit_depth).intra_pred[mode.index()];
    f(
        dst,
        dst_stride,
        edge,
        width,
        height,
        angle,
        max_width,
        max_height,
        bit_depth,
        bitdepth_max,
    );
}

macro_rules! cfl_ac {
    ($name:ident, $i:expr) => {
        /// `cfl_ac[layout - 1]`.
        pub fn $name(
            ac: &mut [i16],
            y: &[u8],
            y_stride: usize,
            w_pad: usize,
            h_pad: usize,
            cw: usize,
            ch: usize,
            bit_depth: BitDepth,
        ) {
            let f = IntraPredDsp::scalar_for(bit_depth).cfl_ac[$i];
            f(ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth);
        }
    };
}

cfl_ac!(cfl_ac_420, 0);
cfl_ac!(cfl_ac_422, 1);
cfl_ac!(cfl_ac_444, 2);

macro_rules! cfl_pred {
    ($name:ident, $i:expr) => {
        /// `cfl_pred[mode]`.
        pub fn $name(
            dst: &mut [u8],
            dst_stride: usize,
            edge: Edge<'_>,
            width: usize,
            height: usize,
            ac: &[i16],
            alpha: i32,
            bit_depth: BitDepth,
            bitdepth_max: u16,
        ) {
            let f = IntraPredDsp::scalar_for(bit_depth).cfl_pred[$i];
            f(
                dst,
                dst_stride,
                edge,
                width,
                height,
                ac,
                alpha,
                bit_depth,
                bitdepth_max,
            );
        }
    };
}

cfl_pred!(cfl_pred_dc, 0);
cfl_pred!(cfl_pred_left_dc, 1);
cfl_pred!(cfl_pred_top_dc, 2);
cfl_pred!(cfl_pred_dc_128, 3);

/// `pal_pred`.
pub fn pal_pred(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) {
    let f = IntraPredDsp::scalar_for(bit_depth).pal_pred;
    f(dst, dst_stride, pal, idx, w, h, bit_depth);
}
