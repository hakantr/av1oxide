//! Skaler `pal_idx_finish` çekirdeği — S2b.2.
//!
//! dav1d `src/pal.c:37-57` gövdesinin birebir portu. Sözleşme ve kaynak
//! eşlemesi [`crate::dsp::pal`] belgesindedir.
//!
//! ## Yuvarlama / aritmetik
//!
//! Yuvarlama yoktur: paketleme `src[2x] | (src[2x+1] << 4)`, görünmez kenar
//! dolgusu `src[w-1] * 0x11` (aynı nibble iki yarıda) ve son satırın kopyasıdır
//! (`src/pal.c:46-55`). Tüm değerler `u8`; taşma yoktur.
//!
//! ## Bellek sözleşmesi
//!
//! `dst`, `(bw/2)·bh` baytın **tamamına** yazılır (görünür bölge + sağ/alt
//! görünmez kenar). `src` `bw·bh` bayttır; yalnız görünür `w·h` bölgesi ve
//! dolgu için `src[w-1]` okunur. Hizalama varsayımı yoktur.

#![forbid(unsafe_code)]

/// `pal_idx_finish` skaler çekirdeği (dav1d `pal_idx_finish_c`).
///
/// # Panikler
///
/// `bw`/`bh` 4..64 ve 2'nin kuvveti değilse, `w`/`h` 4'ün katı değilse ya da
/// dilimler kısaşa panikler.
pub fn pal_idx_finish(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    assert!(
        (4..=64).contains(&bw) && bw.is_power_of_two(),
        "pal_idx_finish bw 4..=64 ve 2'nin kuvveti olmalı: {bw}"
    );
    assert!(
        (4..=64).contains(&bh) && bh.is_power_of_two(),
        "pal_idx_finish bh 4..=64 ve 2'nin kuvveti olmalı: {bh}"
    );
    assert!(
        (4..=w).contains(&w) && w <= bw && w.is_multiple_of(4),
        "pal_idx_finish w 4..=bw ve 4'ün katı olmalı: {w}"
    );
    assert!(
        (4..=h).contains(&h) && h <= bh && h.is_multiple_of(4),
        "pal_idx_finish h 4..=bh ve 4'ün katı olmalı: {h}"
    );
    assert!(
        src.len() >= bw * bh,
        "pal_idx_finish src dilimi kısa: {} < {}",
        src.len(),
        bw * bh
    );
    let dst_width = w / 2;
    let dst_bw = bw / 2;
    assert!(
        dst.len() >= dst_bw * bh,
        "pal_idx_finish dst dilimi kısa: {} < {}",
        dst.len(),
        dst_bw * bh
    );

    for y in 0..h {
        let src_row = y * bw;
        let dst_row = y * dst_bw;
        for x in 0..dst_width {
            // `pal.c:51`: C'de `int` OR + `uint8_t` daraltma; Rust'ta `u8`
            // işlemleri aynı alt 8 biti verir (`<< 4` üst bitleri atar).
            dst[dst_row + x] = src[src_row + x * 2] | (src[src_row + x * 2 + 1] << 4);
        }
        if dst_width < dst_bw {
            // `pal.c:53`: `src[w-1] * 0x11`. İndeksler `read_pal_indices`'ten
            // 0..=7 gelir (`decode_symbol_adapt` < `pal_sz ≤ 8`), yani
            // `7 * 17 = 119 < 256` ⇒ `u8` taşması **oluşamaz**; C'deki `int`
            // çarpımı da aynı sonucu verir.
            let fill = src[src_row + w - 1] * 0x11;
            for x in dst_width..dst_bw {
                dst[dst_row + x] = fill;
            }
        }
    }

    if h < bh {
        let last_row = (h - 1) * dst_bw;
        for y in h..bh {
            let dst_row = y * dst_bw;
            for x in 0..dst_bw {
                dst[dst_row + x] = dst[last_row + x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pal_idx_finish;

    #[test]
    fn gorunur_bolge_paketlenir() {
        let mut src = [0u8; 16];
        for (index, value) in src.iter_mut().enumerate() {
            *value = (index % 8) as u8;
        }
        let mut dst = [0u8; 8];
        pal_idx_finish(&mut dst, &src, 4, 4, 4, 4);
        assert_eq!(dst[0], 0x10);
        assert_eq!(dst[1], 0x32);
        assert_eq!(dst[2], 0x54);
        assert_eq!(dst[3], 0x76);
    }

    #[test]
    fn gorunmez_kenar_doldurulur() {
        let mut src = [0u8; 16 * 16];
        for (index, value) in src.iter_mut().enumerate() {
            *value = ((index / 16 + index % 16) % 8) as u8;
        }
        let mut dst = [0u8; 8 * 16];
        pal_idx_finish(&mut dst, &src, 16, 16, 8, 8);
        // Görünür bölge 4 bayt; sağdaki 4 bayt o satırın son görünür
        // indeksinin (`src[y*16 + 7]`) 0x11 katıdır.
        for y in 0..8 {
            let last = src[y * 16 + 7];
            for x in 4..8 {
                assert_eq!(dst[y * 8 + x], last * 0x11, "y={y} x={x}");
            }
        }
        // Alt satırlar son görünür satırın kopyası.
        for y in 8..16 {
            for x in 0..8 {
                assert_eq!(dst[y * 8 + x], dst[7 * 8 + x]);
            }
        }
    }
}
