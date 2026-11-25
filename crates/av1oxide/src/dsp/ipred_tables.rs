//! dav1d intra tahmin tabloları — S2b.2.
//!
//! **Bayt-eşitlik:** aşağıdaki üç tablonun baytları, dav1d `src/tables.c`
//! derlemesinden **doğrudan** dökülmüştür (`ARCH_X86=1`, `HAVE_ASM=0`;
//! referans derleme `.cache/dav1d-build-noasm`). SHA-256:
//! `84eb0bf4385137709ef925d54a2a427a39c789f66b54b389c52c7267c0fb0536` (536 bayt).
//!
//! | Tablo | Kaynak | Boyut |
//! |---|---|---|
//! | `dav1d_sm_weights` | `src/tables.c:690-717` | 128 bayt |
//! | `dav1d_dr_intra_derivative` | `src/tables.c:719-758` | 44 × u16 = 88 bayt |
//! | `dav1d_filter_intra_taps` | `src/tables.c:765-829` | 5 × 64 × i8 = 320 bayt |
//!
//! `dav1d_filter_intra_taps` dav1d'de `ARCH_X86` koşuluna bağlı bir `F()`
//! makrosuyla **serpiştirilmiş** yerleşimde tanımlanır (`src/tables.c:760-772`);
//! C yolu da aynı koşulla `FILTER`/`FLT_INCR` seçer (`src/ipred_tmpl.c:586-601`).
//! Referans derleme x86_64 olduğu için buradaki baytlar **x86 yerleşimidir**:
//! 8 çıktı için katsayı çiftleri `[2k], [2k+1], [16+2k], [17+2k], …`. Skaler
//! çekirdek bu yerleşimi `FLT_INCR = 2` ile okur; semantik olarak kaynak
//! tablodaki `F(idx, f0..f6)` girişleriyle birebir aynıdır.

#![forbid(unsafe_code)]

/// `dav1d_sm_weights[128]` — smooth tahmin ağırlıkları (`src/tables.c:690`).
pub const SM_WEIGHTS: [u8; 128] = [
    0, 0, 255, 128, 255, 149, 85, 64, 255, 197, 146, 105, 73, 50, 37, 32, 255, 225, 196, 170, 145,
    123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16, 255, 240, 225, 210, 196, 182, 169, 157, 145, 133,
    122, 111, 101, 92, 83, 74, 66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8, 255,
    248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138, 133, 127,
    121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44, 41, 38, 35,
    32, 29, 27, 25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

/// `dav1d_dr_intra_derivative[44]` — yönlü tahmin türevleri (`src/tables.c:719`).
pub const DR_INTRA_DERIVATIVE: [u16; 44] = [
    0, 1023, 0, 547, 372, 0, 0, 273, 215, 0, 178, 151, 0, 132, 116, 0, 102, 0, 90, 80, 0, 71, 64,
    0, 57, 51, 0, 45, 0, 40, 35, 0, 31, 27, 0, 23, 19, 0, 15, 0, 11, 0, 7, 3,
];

/// `dav1d_filter_intra_taps[5][64]` — filtre intra katsayıları (`src/tables.c:765`),
/// x86 yerleşimi (bkz. modül belgesi).
pub const FILTER_INTRA_TAPS: [[i8; 64]; 5] = [
    [
        -6, 10, -5, 2, -3, 1, -3, 1, -4, 6, -3, 2, -3, 2, -3, 1, 0, 0, 10, 0, 1, 10, 1, 2, 0, 0, 6,
        0, 2, 6, 2, 2, 0, 12, 0, 9, 0, 7, 10, 5, 0, 2, 0, 2, 0, 2, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0,
        12, 0, 9, 0, 7, 0, 5, 0,
    ],
    [
        -10, 16, -6, 0, -4, 0, -2, 0, -10, 16, -6, 0, -4, 0, -2, 0, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0,
        16, 0, 0, 16, 0, 0, 0, 10, 0, 6, 0, 4, 16, 2, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10, 0, 6, 0, 4, 0, 2, 0,
    ],
    [
        -8, 8, -8, 0, -8, 0, -8, 0, -4, 4, -4, 0, -4, 0, -4, 0, 0, 0, 8, 0, 0, 8, 0, 0, 0, 0, 4, 0,
        0, 4, 0, 0, 0, 16, 0, 16, 0, 16, 8, 16, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16,
        0, 16, 0, 16, 0, 16, 0,
    ],
    [
        -2, 8, -1, 3, -1, 2, 0, 1, -1, 4, -1, 3, -1, 2, -1, 2, 0, 0, 8, 0, 3, 8, 2, 3, 0, 0, 4, 0,
        3, 4, 2, 3, 0, 10, 0, 6, 0, 4, 8, 2, 0, 3, 0, 4, 0, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0,
        6, 0, 4, 0, 3, 0,
    ],
    [
        -12, 14, -10, 0, -9, 0, -8, 0, -10, 12, -9, 1, -8, 0, -7, 0, 0, 0, 14, 0, 0, 14, 0, 0, 0,
        0, 12, 0, 0, 12, 0, 1, 0, 14, 0, 12, 0, 11, 14, 10, 0, 0, 0, 0, 0, 1, 12, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 14, 0, 12, 0, 11, 0, 9, 0,
    ],
];
