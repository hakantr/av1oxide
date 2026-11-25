//! Superres büyütme — dav1d `resize_c` (`src/mc_tmpl.c:918-944`) skaler gövdesi.
//!
//! Sözleşme: [`crate::dsp::resize`]. 8/10/12 bit tek gövde; piksel okuma ve
//! yazma bayt dilimleri üzerindendir (16bpc küçük sonlu).

#![forbid(unsafe_code)]

use crate::dsp::resize::ResizeBitDepth;
use crate::dsp::resize_tables::RESIZE_FILTER;

#[inline]
fn load(buf: &[u8], idx: usize, bpp: usize) -> i32 {
    if bpp == 1 {
        i32::from(buf[idx])
    } else {
        i32::from(u16::from_le_bytes([buf[2 * idx], buf[2 * idx + 1]]))
    }
}

#[inline]
fn store(buf: &mut [u8], idx: usize, bpp: usize, v: i32) {
    if bpp == 1 {
        buf[idx] = v as u8;
    } else {
        buf[2 * idx..2 * idx + 2].copy_from_slice(&(v as u16).to_le_bytes());
    }
}

/// `resize_c` (`src/mc_tmpl.c:918-944`).
///
/// Her satırda `mx = mx0`, `src_x = −1` ile başlar; çıkış pikseli `x` için
/// 8 dokunuşlu filtre `RESIZE_FILTER[mx >> 8]` kaynak `src_x − 3 .. src_x + 4`
/// (her biri `[0, src_w − 1]`'e kırpılmış) üzerinde uygulanır ve
/// `(−Σ + 64) >> 7` `[0, bitdepth_max]`'a kırpılır; sonra
/// `mx += dx; src_x += mx >> 14; mx &= 0x3fff`.
#[allow(clippy::too_many_arguments)]
pub fn resize<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    dst_w: usize,
    h: usize,
    src_w: usize,
    dx: i32,
    mx0: i32,
    bd: ResizeBitDepth,
) {
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    let bd = match BPC {
        8 => ResizeBitDepth::B8,
        10 => ResizeBitDepth::B10,
        _ => ResizeBitDepth::B12,
    };
    let bpp = bd.bytes();
    let max = bd.bitdepth_max();
    let ok = h > 0
        && dst_w > 0
        && src_w > 0
        && (0..=0x3fff).contains(&mx0)
        && dx > 0
        && src.len() >= (h - 1) * src_stride + src_w * bpp
        && dst.len() >= (h - 1) * dst_stride + dst_w * bpp;
    debug_assert!(ok, "resize sözleşme dışı girdi");
    if !ok {
        return;
    }
    let last = src_w as i64 - 1;
    for y in 0..h {
        let s = &src[y * src_stride..];
        let d = &mut dst[y * dst_stride..];
        let mut mx = mx0;
        let mut src_x: i64 = -1;
        for x in 0..dst_w {
            let f = &RESIZE_FILTER[(mx >> 8) as usize];
            let mut sum = 0i32;
            for (k, &tap) in f.iter().enumerate() {
                let sx = (src_x + k as i64 - 3).clamp(0, last) as usize;
                sum += i32::from(tap) * load(s, sx, bpp);
            }
            store(d, x, bpp, ((-sum + 64) >> 7).clamp(0, max));
            mx += dx;
            src_x += i64::from(mx >> 14);
            mx &= 0x3fff;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test kısayolu: bit derinliğinin skaler yuvası.
    #[allow(clippy::too_many_arguments)]
    fn resize(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        dst_w: usize,
        h: usize,
        src_w: usize,
        dx: i32,
        mx0: i32,
        bd: ResizeBitDepth,
    ) {
        (crate::dsp::resize::scalar_for(bd))(
            dst, dst_stride, src, src_stride, dst_w, h, src_w, dx, mx0, bd,
        );
    }

    /// `dx = 1 << 14` ve `mx0 = 0`: filtre 0 (`[0,0,0,−128,0,0,0,0]`) kaynak
    /// `src_x` pikselini (−1'den başlar, kırpılır) aynen kopyalar.
    #[test]
    fn birim_adim_kopya() {
        let src: Vec<u8> = (0..16u8).map(|v| v * 3).collect();
        let mut dst = vec![0u8; 16];
        resize(
            &mut dst,
            16,
            &src,
            16,
            16,
            1,
            16,
            1 << 14,
            0,
            ResizeBitDepth::B8,
        );
        // x = 0'da src_x = −1 → kırpılmış 0; sonra her adımda bir ilerler.
        let mut want = vec![src[0]];
        want.extend_from_slice(&src[..15]);
        assert_eq!(dst, want);
    }

    /// Kaynak kenar ötesi okunmaz: `src` tam `src_w` uzunluğunda olabilir.
    #[test]
    fn kenar_kirpmasi_sinir_dilim() {
        for bd in ResizeBitDepth::ALL {
            let bpp = bd.bytes();
            let src = vec![0x55u8; 7 * bpp];
            let mut dst = vec![0u8; 13 * bpp];
            resize(&mut dst, 13 * bpp, &src, 7 * bpp, 13, 1, 7, 8820, 1234, bd);
            assert!(dst.iter().any(|&b| b != 0));
        }
    }

    /// 16bpc çıkış `bitdepth_max`'a kırpılır.
    #[test]
    fn yuksek_bit_kirpma() {
        let max = ResizeBitDepth::B10.bitdepth_max() as u16;
        let mut src = Vec::new();
        for x in 0..32u16 {
            src.extend_from_slice(&(if x % 2 == 0 { max } else { 0 }).to_le_bytes());
        }
        let mut dst = vec![0u8; 64];
        resize(
            &mut dst,
            64,
            &src,
            64,
            32,
            1,
            32,
            9000,
            77,
            ResizeBitDepth::B10,
        );
        for c in dst.chunks(2) {
            assert!(u16::from_le_bytes([c[0], c[1]]) <= max);
        }
    }
}
