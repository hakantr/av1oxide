//! Yerel/global warp parametreleri — dav1d `src/warpmv.c` birebir karşılığı.
//!
//! ## Sorumluluk
//!
//! `dav1d_get_shear_params` (shear parametreleri + geçerlilik) ve
//! `dav1d_find_affine_int` (en küçük kareler affine çözümü) ile
//! `dav1d_set_affine_mv2d`. Warp çekirdekleri (`warp8x8`/`warp8x8t`) P13'te
//! DSP katmanında doğrulandı; bu modül yalnız **parametre** üretir.
//!
//! ## Upstream karşılığı (dav1d @ `99f3354`, salt-okunur)
//!
//! | Rust | dav1d satırı |
//! |---|---|
//! | [`DIV_LUT`] | `src/warpmv.c:43-68` |
//! | [`iclip_wmp`] | `src/warpmv.c:70-74` |
//! | [`resolve_divisor_32`] | `src/warpmv.c:76-85` |
//! | [`get_shear_params`] | `src/warpmv.c:87-108` |
//! | [`resolve_divisor_64`] | `src/warpmv.c:110-121` |
//! | [`find_affine_int`] | `src/warpmv.c:157-230` |
//!
//! ## Neden ayrı modül
//!
//! Fonksiyonlar `Dec` durumuna bağlı değildir; `intra.rs` içinde tutmak
//! dosyayı daha da büyütür ve test edilebilirliği azaltırdı.

#![forbid(unsafe_code)]

use crate::bitstream::frame_header::WarpedMotionParams;

/// `div_lut` (`src/warpmv.c:43-68`) — 257 girişli bölme çarpanı tablosu.
pub(crate) const DIV_LUT: [u16; 257] = [
    16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768, 15709, 15650,
    15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142, 15087, 15033, 14980, 14926,
    14873, 14821, 14769, 14717, 14665, 14614, 14564, 14513, 14463, 14413, 14364, 14315, 14266,
    14218, 14170, 14122, 14075, 14028, 13981, 13935, 13888, 13843, 13797, 13752, 13707, 13662,
    13618, 13574, 13530, 13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148, 13107,
    13066, 13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633, 12596,
    12558, 12520, 12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228, 12193, 12157, 12122,
    12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848, 11815, 11782, 11749, 11716, 11683,
    11651, 11619, 11586, 11555, 11523, 11491, 11460, 11429, 11398, 11367, 11336, 11305, 11275,
    11245, 11215, 11185, 11155, 11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894,
    10866, 10838, 10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538,
    10512, 10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255, 10230, 10205,
    10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986, 9963, 9939, 9916, 9892, 9869,
    9846, 9823, 9800, 9777, 9754, 9732, 9709, 9687, 9664, 9642, 9620, 9598, 9576, 9554, 9533, 9511,
    9489, 9468, 9447, 9425, 9404, 9383, 9362, 9341, 9321, 9300, 9279, 9259, 9239, 9218, 9198, 9178,
    9158, 9138, 9118, 9098, 9079, 9059, 9039, 9020, 9001, 8981, 8962, 8943, 8924, 8905, 8886, 8867,
    8849, 8830, 8812, 8793, 8775, 8756, 8738, 8720, 8702, 8684, 8666, 8648, 8630, 8613, 8595, 8577,
    8560, 8542, 8525, 8508, 8490, 8473, 8456, 8439, 8422, 8405, 8389, 8372, 8355, 8339, 8322, 8306,
    8289, 8273, 8257, 8240, 8224, 8208, 8192,
];

/// `iclip_wmp` (`src/warpmv.c:70-74`).
fn iclip_wmp(v: i32) -> i16 {
    let cv = v.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    let mag = ((cv.abs() + 32) >> 6) * (1 << 6);
    if cv < 0 { -mag as i16 } else { mag as i16 }
}

/// `resolve_divisor_32` (`src/warpmv.c:76-85`).
fn resolve_divisor_32(d: u32, shift: &mut i32) -> i32 {
    *shift = 31 - d.leading_zeros() as i32; // ulog2
    let e = d as i64 - (1i64 << *shift);
    let f = if *shift > 8 {
        ((e + (1i64 << (*shift - 9))) >> (*shift - 8)) as usize
    } else {
        (e << (8 - *shift)) as usize
    };
    *shift += 14;
    i32::from(DIV_LUT[f.min(256)])
}

/// `dav1d_get_shear_params` (`src/warpmv.c:87-108`).
///
/// Geçerli (warp uygulanabilir) ise `Some((alpha, beta, gamma, delta))`.
pub(crate) fn get_shear_params(wm: &WarpedMotionParams) -> Option<[i16; 4]> {
    let mat = &wm.matrix;
    if mat[2] <= 0 {
        return None;
    }
    let alpha = iclip_wmp(mat[2] - 0x10000);
    let beta = iclip_wmp(mat[3]);

    let mut shift = 0i32;
    let y0 = resolve_divisor_32(mat[2].unsigned_abs(), &mut shift);
    let y = if mat[2] < 0 { -y0 } else { y0 };
    let v1 = i64::from(mat[4]) * 0x10000 * i64::from(y);
    let rnd = (1i64 << shift) >> 1;
    let gamma = iclip_wmp(apply_sign64(((v1.abs() + rnd) >> shift) as i32, v1));
    let v2 = i64::from(mat[3]) * i64::from(mat[4]) * i64::from(y);
    let delta = iclip_wmp(
        mat[5].wrapping_sub(apply_sign64(((v2.abs() + rnd) >> shift) as i32, v2)) - 0x10000,
    );

    // Geçersizlik koşulu (`src/warpmv.c:104-107`).
    let bad = 4 * i32::from(alpha.abs()) + 7 * i32::from(beta.abs()) >= 0x10000
        || 4 * i32::from(gamma.abs()) + 4 * i32::from(delta.abs()) >= 0x10000;
    if bad {
        None
    } else {
        Some([alpha, beta, gamma, delta])
    }
}

/// `apply_sign64`.
#[inline]
fn apply_sign64(v: i32, s: i64) -> i32 {
    if s < 0 { -v } else { v }
}

/// `resolve_divisor_64` (`src/warpmv.c:110-121`).
fn resolve_divisor_64(d: u64, shift: &mut i32) -> i32 {
    *shift = 63 - d.leading_zeros() as i32; // u64log2
    let e = d as i128 - (1i128 << *shift);
    let f = if *shift > 8 {
        ((e + (1i128 << (*shift - 9))) >> (*shift - 8)) as usize
    } else {
        (e << (8 - *shift)) as usize
    };
    *shift += 14;
    i32::from(DIV_LUT[f.min(256)])
}

/// `get_mult_shift_ndiag` (`src/warpmv.c:125-134`).
fn get_mult_shift_ndiag(px: i64, idet: i32, shift: i32) -> i32 {
    let v1 = px * i64::from(idet);
    let v2 = apply_sign64(((v1.abs() + ((1i64 << shift) >> 1)) >> shift) as i32, v1);
    v2.clamp(-0x1fff, 0x1fff)
}

/// `get_mult_shift_diag` (`src/warpmv.c:136-145`).
fn get_mult_shift_diag(px: i64, idet: i32, shift: i32) -> i32 {
    let v1 = px * i64::from(idet);
    let v2 = apply_sign64(((v1.abs() + ((1i64 << shift) >> 1)) >> shift) as i32, v1);
    v2.clamp(0xe001, 0x11fff)
}

/// `dav1d_find_affine_int` (`src/warpmv.c:157-230`).
///
/// `pts[i] = [[kaynak x, kaynak y], [hedef x, hedef y]]`. Başarılıysa
/// matris `wm.matrix` içine yazılır ve `true` döner.
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_affine_int(
    pts: &[[[i32; 2]; 2]],
    np: usize,
    bw4: usize,
    bh4: usize,
    mv: (i32, i32),
    wm: &mut WarpedMotionParams,
    bx4: usize,
    by4: usize,
) -> bool {
    let mut a = [[0i32; 2]; 2];
    let mut bx = [0i32; 2];
    let mut by = [0i32; 2];
    let rsuy = 2 * bh4 as i32 - 1;
    let rsux = 2 * bw4 as i32 - 1;
    let suy = rsuy * 8;
    let sux = rsux * 8;
    let duy = suy + mv.1;
    let dux = sux + mv.0;
    let isuy = by4 as i32 * 4 + rsuy;
    let isux = bx4 as i32 * 4 + rsux;

    for p in pts.iter().take(np) {
        let dx = p[1][0] - dux;
        let dy = p[1][1] - duy;
        let sx = p[0][0] - sux;
        let sy = p[0][1] - suy;
        if (sx - dx).abs() < 256 && (sy - dy).abs() < 256 {
            a[0][0] += ((sx * sx) >> 2) + sx * 2 + 8;
            a[0][1] += ((sx * sy) >> 2) + sx + sy + 4;
            a[1][1] += ((sy * sy) >> 2) + sy * 2 + 8;
            bx[0] += ((sx * dx) >> 2) + sx + dx + 8;
            bx[1] += ((sy * dx) >> 2) + sy + dx + 4;
            by[0] += ((sx * dy) >> 2) + sx + dy + 4;
            by[1] += ((sy * dy) >> 2) + sy + dy + 8;
        }
    }

    let det = i64::from(a[0][0]) * i64::from(a[1][1]) - i64::from(a[0][1]) * i64::from(a[0][1]);
    if det == 0 {
        return false;
    }
    let mut shift = 0i32;
    let mut idet = resolve_divisor_64(det.unsigned_abs(), &mut shift);
    if det < 0 {
        idet = -idet;
    }
    shift -= 16;
    if shift < 0 {
        idet <<= -shift;
        shift = 0;
    }

    wm.matrix[2] = get_mult_shift_diag(
        i64::from(a[1][1]) * i64::from(bx[0]) - i64::from(a[0][1]) * i64::from(bx[1]),
        idet,
        shift,
    );
    wm.matrix[3] = get_mult_shift_ndiag(
        i64::from(a[0][0]) * i64::from(bx[1]) - i64::from(a[0][1]) * i64::from(bx[0]),
        idet,
        shift,
    );
    wm.matrix[4] = get_mult_shift_ndiag(
        i64::from(a[1][1]) * i64::from(by[0]) - i64::from(a[0][1]) * i64::from(by[1]),
        idet,
        shift,
    );
    wm.matrix[5] = get_mult_shift_diag(
        i64::from(a[0][0]) * i64::from(by[1]) - i64::from(a[0][1]) * i64::from(by[0]),
        idet,
        shift,
    );

    wm.matrix[0] = (mv.0 * 0x2000 - (isux * (wm.matrix[2] - 0x10000) + isuy * wm.matrix[3]))
        .clamp(-0x800000, 0x7fffff);
    wm.matrix[1] = (mv.1 * 0x2000 - (isux * wm.matrix[4] + isuy * (wm.matrix[5] - 0x10000)))
        .clamp(-0x800000, 0x7fffff);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_lut_uclu_ve_azalan() {
        assert_eq!(DIV_LUT.len(), 257);
        assert_eq!(DIV_LUT[0], 16384);
        assert_eq!(DIV_LUT[256], 8192);
        for w in DIV_LUT.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn translation_shear_gecerli() {
        let wm = WarpedMotionParams {
            kind: crate::bitstream::frame_header::WarpedMotionType::Translation,
            matrix: [1 << 16, 0, 1 << 16, 0, 0, 1 << 16],
        };
        let p = get_shear_params(&wm).expect("sıfır matris geçerli");
        assert_eq!(p, [0, 0, 0, 0]);
    }
}
