//! Loop restoration tabloları — dav1d `src/tables.c:415-442` (S2c.2b).
//!
//! Değerler dav1d kaynağından **birebir** kopyalanmıştır; oracle testi
//! (`tests/lr_scalar_oracle.rs`, `lr_tablolar_dav1d_ile_bayt_esit`) dav1d
//! derlemesinin ham bayt dökümüyle karşılaştırır.

#![forbid(unsafe_code)]

/// `dav1d_sgr_params[16][2]` (`src/tables.c:415-420`): `[s0 (5×5), s1 (3×3)]`.
///
/// `sgr_idx = lr->type - DAV1D_RESTORATION_SGRPROJ` ile seçilir
/// (`src/lr_apply_tmpl.c:75-78`); `0` değeri ilgili geçişin **yok** olduğunu
/// gösterir (10..13 yalnız 3×3, 14..15 yalnız 5×5).
#[rustfmt::skip]
pub const SGR_PARAMS: [[u16; 2]; 16] = [
    [140, 3236], [112, 2158], [93, 1618], [80, 1438],
    [70, 1295], [58, 1177], [47, 1079], [37, 996],
    [30, 925], [25, 863], [0, 2589], [0, 1618],
    [0, 1177], [0, 925], [56, 0], [22, 0],
];

/// `dav1d_sgr_x_by_x[256]` (`src/tables.c:423-442`): `z → x` ters çevirme
/// tablosu (`sgr_calc_row_ab`, `src/looprestoration_tmpl.c:511`).
#[rustfmt::skip]
pub const SGR_X_BY_X: [u8; 256] = [
    255, 128,  85,  64,  51,  43,  37,  32,  28,  26,  23,  21,  20,  18,  17,  16,
     15,  14,  13,  13,  12,  12,  11,  11,  10,  10,   9,   9,   9,   9,   8,   8,
      8,   8,   7,   7,   7,   7,   7,   6,   6,   6,   6,   6,   6,   6,   5,   5,
      5,   5,   5,   5,   5,   5,   5,   5,   4,   4,   4,   4,   4,   4,   4,   4,
      4,   4,   4,   4,   4,   4,   4,   4,   4,   3,   3,   3,   3,   3,   3,   3,
      3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
      3,   3,   3,   3,   3,   3,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
      2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
      2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
      2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
      2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
      1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,
];

#[cfg(test)]
mod tests {
    use super::{SGR_PARAMS, SGR_X_BY_X};

    #[test]
    fn sgr_tablolari_sinir_degerleri() {
        assert_eq!(SGR_PARAMS[0], [140, 3236]);
        assert_eq!(SGR_PARAMS[10], [0, 2589]);
        assert_eq!(SGR_PARAMS[15], [22, 0]);
        // 10..13 yalnız 3×3, 14..15 yalnız 5×5; hiçbir girdide ikisi birden 0 değil.
        for (i, p) in SGR_PARAMS.iter().enumerate() {
            assert!(p[0] != 0 || p[1] != 0, "idx {i}");
            assert_eq!(p[0] == 0, (10..14).contains(&i), "idx {i}");
            assert_eq!(p[1] == 0, i >= 14, "idx {i}");
        }
        assert_eq!(SGR_X_BY_X[0], 255);
        assert_eq!(SGR_X_BY_X[1], 128);
        assert_eq!(SGR_X_BY_X[255], 0);
        // Tablo artmayandır (z büyüdükçe x küçülür).
        assert!(SGR_X_BY_X.windows(2).all(|p| p[0] >= p[1]));
    }
}
