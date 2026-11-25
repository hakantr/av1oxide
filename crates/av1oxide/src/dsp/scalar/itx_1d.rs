//! Ters dönüşüm 1 boyutlu çekirdekleri — dav1d `src/itx_1d.c` **mekanik portu**.
//!
//! ## Kaynak ve lisans
//!
//! Kaynak: dav1d `99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`, `src/itx_1d.c`
//! (BSD-2-Clause; bkz. `UPSTREAM.md`). Her işlevin başında kaynak satır aralığı
//! yazılıdır. Çelişki hâlinde dav1d kaynağı esastır.
//!
//! ## Aritmetik sözleşmesi (tanımlı davranış)
//!
//! dav1d C'si `int` (32-bit) ara değerler kullanır; geçerli AV1 katsayı
//! aralığında taşma **oluşmaz** (dav1d `src/itx_1d.c:37-63` yorumu, 19+sign
//! bit katsayı için ters çevrilmiş katsayı hilesi), ancak C'de işaretli taşma
//! yine de UB'dir. Bu port **her ara değeri `i64`** tutar; böylece:
//!
//! - geçerli aralıkta sonuç C ile **bit-aynıdır** (taşma yoksa genişletme
//!   sonucu değiştirmez),
//! - geçersiz/bozuk girdide bile Rust tarafında taşma/panik yoktur (sonuç
//!   tanımlıdır; bu girdiler AV1 uyumlu değildir).
//!
//! `>>` Rust'ta `i64` için **aritmetiktir** (C'de GCC'nin davranışıyla aynı).
//! `clip` sınırları çağıran tarafından verilir (`min`/`max`); dav1d'deki
//! `CLIP(a) = iclip(a, min, max, min, max)` ile birebir aynıdır.
//!
//! ## 64 noktalı dönüşümlerde sıfır bölgeleri
//!
//! 64 noktalı DCT yalnız **ilk 32 girişi** okur: `inv_dct64` → `inv_dct32`
//! çağrısı `stride << 1` ile yapılır ve ardından yalnız `in1..in31` (tek
//! indeksler) okunur; bileşik stride'lar nedeniyle erişilen en büyük indeks
//! 31'dir (dav1d `src/itx_1d.c:436-781`). AV1'de 64 noktalı dönüşümün üst yarı
//! katsayıları zaten sıfırdır; `inv_txfm_add_c` bu yüzden `sw = min(w,32)`,
//! `sh = min(h,32)` ile yalnız sol üst 32×32 bölgeyi okur
//! (dav1d `src/itx_tmpl.c:102,109-127`).
//!
//! ## FLIPADST
//!
//! dav1d flipADST'yi aynı ADST gövdesini **negatif çıkış stride'ı** ile
//! çağırarak yapar (`src/itx_1d.c:965-979`). Negatif stride yerine burada
//! `flip: bool` ve indeks eşlemesi kullanılır:
//! `out[k] = c[(sz-1-k)*stride]`; okunan ve yazılan kümeler aynıdır.

#![forbid(unsafe_code)]

/// Tek boyutlu ters dönüşüm işlevi: `c` dizisini `stride` adımla yerinde
/// dönüştürür; ara değerler `[min, max]` aralığına kırpılır (dav1d `CLIP`).
pub type Itx1dFn = fn(&mut [i64], usize, i64, i64);

/// dav1d `include/common/intops.h:45-47` (`iclip`) portu.
#[inline]
fn clip(v: i64, min: i64, max: i64) -> i64 {
    v.clamp(min, max)
}

/// 1B dönüşüm türü — dav1d `src/itx_1d.h:40-46` sırası **korunur**
/// (`DCT=0, ADST=1, IDENTITY=2, FLIPADST=3`); `TX1D_FNS` bu indeksle okunur.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Tx1dType {
    /// DCT-II ters dönüşümü.
    Dct = 0,
    /// ADST ters dönüşümü.
    Adst = 1,
    /// Özdeşlik (ölçekli) dönüşümü.
    Identity = 2,
    /// FlipADST (ADST'nin ters çıktı sırası).
    FlipAdst = 3,
}

/// dav1d `src/itx_1d.c:66-90` (`inv_dct4_1d_internal_c`) — mekanik port.
fn inv_dct4_1d_internal(c: &mut [i64], stride: usize, min: i64, max: i64, tx64: bool) {
    let in0: i64 = c[0];
    let in1: i64 = c[stride];

    let t0: i64;
    let t1: i64;
    let t2: i64;
    let t3: i64;
    if tx64 {
        t0 = (in0 * 181 + 128) >> 8;
        t1 = (in0 * 181 + 128) >> 8;
        t2 = (in1 * 1567 + 2048) >> 12;
        t3 = (in1 * 3784 + 2048) >> 12;
    } else {
        let in2: i64 = c[2 * stride];
        let in3: i64 = c[3 * stride];

        t0 = ((in0 + in2) * 181 + 128) >> 8;
        t1 = ((in0 - in2) * 181 + 128) >> 8;
        t2 = ((in1 * 1567 - in3 * (3784 - 4096) + 2048) >> 12) - in3;
        t3 = ((in1 * (3784 - 4096) + in3 * 1567 + 2048) >> 12) + in1;
    }

    c[0] = clip(t0 + t3, min, max);
    c[stride] = clip(t1 + t2, min, max);
    c[2 * stride] = clip(t1 - t2, min, max);
    c[3 * stride] = clip(t0 - t3, min, max);
}

/// dav1d `src/itx_1d.c:99-143` (`inv_dct8_1d_internal_c`) — mekanik port.
fn inv_dct8_1d_internal(c: &mut [i64], stride: usize, min: i64, max: i64, tx64: bool) {
    inv_dct4_1d_internal(c, stride << 1, min, max, tx64);

    let in1: i64 = c[stride];
    let in3: i64 = c[3 * stride];

    let t4a: i64;
    let mut t5a: i64;
    let mut t6a: i64;
    let t7a: i64;
    if tx64 {
        t4a = (in1 * 799 + 2048) >> 12;
        t5a = (in3 * -2276 + 2048) >> 12;
        t6a = (in3 * 3406 + 2048) >> 12;
        t7a = (in1 * 4017 + 2048) >> 12;
    } else {
        let in5: i64 = c[5 * stride];
        let in7: i64 = c[7 * stride];

        t4a = ((in1 * 799 - in7 * (4017 - 4096) + 2048) >> 12) - in7;
        t5a = (in5 * 1703 - in3 * 1138 + 1024) >> 11;
        t6a = (in5 * 1138 + in3 * 1703 + 1024) >> 11;
        t7a = ((in1 * (4017 - 4096) + in7 * 799 + 2048) >> 12) + in1;
    }

    let t4: i64 = clip(t4a + t5a, min, max);
    t5a = clip(t4a - t5a, min, max);
    let t7: i64 = clip(t7a + t6a, min, max);
    t6a = clip(t7a - t6a, min, max);

    let t5: i64 = ((t6a - t5a) * 181 + 128) >> 8;
    let t6: i64 = ((t6a + t5a) * 181 + 128) >> 8;

    let t0: i64 = c[0];
    let t1: i64 = c[2 * stride];
    let t2: i64 = c[4 * stride];
    let t3: i64 = c[6 * stride];

    c[0] = clip(t0 + t7, min, max);
    c[stride] = clip(t1 + t6, min, max);
    c[2 * stride] = clip(t2 + t5, min, max);
    c[3 * stride] = clip(t3 + t4, min, max);
    c[4 * stride] = clip(t3 - t4, min, max);
    c[5 * stride] = clip(t2 - t5, min, max);
    c[6 * stride] = clip(t1 - t6, min, max);
    c[7 * stride] = clip(t0 - t7, min, max);
}

/// dav1d `src/itx_1d.c:152-238` (`inv_dct16_1d_internal_c`) — mekanik port.
fn inv_dct16_1d_internal(c: &mut [i64], stride: usize, min: i64, max: i64, tx64: bool) {
    inv_dct8_1d_internal(c, stride << 1, min, max, tx64);

    let in1: i64 = c[stride];
    let in3: i64 = c[3 * stride];
    let in5: i64 = c[5 * stride];
    let in7: i64 = c[7 * stride];

    let mut t8a: i64;
    let mut t9a: i64;
    let mut t10a: i64;
    let mut t11a: i64;
    let mut t12a: i64;
    let mut t13a: i64;
    let mut t14a: i64;
    let mut t15a: i64;
    if tx64 {
        t8a = (in1 * 401 + 2048) >> 12;
        t9a = (in7 * -2598 + 2048) >> 12;
        t10a = (in5 * 1931 + 2048) >> 12;
        t11a = (in3 * -1189 + 2048) >> 12;
        t12a = (in3 * 3920 + 2048) >> 12;
        t13a = (in5 * 3612 + 2048) >> 12;
        t14a = (in7 * 3166 + 2048) >> 12;
        t15a = (in1 * 4076 + 2048) >> 12;
    } else {
        let in9: i64 = c[9 * stride];
        let in11: i64 = c[11 * stride];
        let in13: i64 = c[13 * stride];
        let in15: i64 = c[15 * stride];

        t8a = ((in1 * 401 - in15 * (4076 - 4096) + 2048) >> 12) - in15;
        t9a = (in9 * 1583 - in7 * 1299 + 1024) >> 11;
        t10a = ((in5 * 1931 - in11 * (3612 - 4096) + 2048) >> 12) - in11;
        t11a = ((in13 * (3920 - 4096) - in3 * 1189 + 2048) >> 12) + in13;
        t12a = ((in13 * 1189 + in3 * (3920 - 4096) + 2048) >> 12) + in3;
        t13a = ((in5 * (3612 - 4096) + in11 * 1931 + 2048) >> 12) + in5;
        t14a = (in9 * 1299 + in7 * 1583 + 1024) >> 11;
        t15a = ((in1 * (4076 - 4096) + in15 * 401 + 2048) >> 12) + in1;
    }

    let t8: i64 = clip(t8a + t9a, min, max);
    let mut t9: i64 = clip(t8a - t9a, min, max);
    let mut t10: i64 = clip(t11a - t10a, min, max);
    let mut t11: i64 = clip(t11a + t10a, min, max);
    let mut t12: i64 = clip(t12a + t13a, min, max);
    let mut t13: i64 = clip(t12a - t13a, min, max);
    let mut t14: i64 = clip(t15a - t14a, min, max);
    let t15: i64 = clip(t15a + t14a, min, max);

    t9a = ((t14 * 1567 - t9 * (3784 - 4096) + 2048) >> 12) - t9;
    t14a = ((t14 * (3784 - 4096) + t9 * 1567 + 2048) >> 12) + t14;
    t10a = ((-(t13 * (3784 - 4096) + t10 * 1567) + 2048) >> 12) - t13;
    t13a = ((t13 * 1567 - t10 * (3784 - 4096) + 2048) >> 12) - t10;

    t8a = clip(t8 + t11, min, max);
    t9 = clip(t9a + t10a, min, max);
    t10 = clip(t9a - t10a, min, max);
    t11a = clip(t8 - t11, min, max);
    t12a = clip(t15 - t12, min, max);
    t13 = clip(t14a - t13a, min, max);
    t14 = clip(t14a + t13a, min, max);
    t15a = clip(t15 + t12, min, max);

    t10a = ((t13 - t10) * 181 + 128) >> 8;
    t13a = ((t13 + t10) * 181 + 128) >> 8;
    t11 = ((t12a - t11a) * 181 + 128) >> 8;
    t12 = ((t12a + t11a) * 181 + 128) >> 8;

    let t0: i64 = c[0];
    let t1: i64 = c[2 * stride];
    let t2: i64 = c[4 * stride];
    let t3: i64 = c[6 * stride];
    let t4: i64 = c[8 * stride];
    let t5: i64 = c[10 * stride];
    let t6: i64 = c[12 * stride];
    let t7: i64 = c[14 * stride];

    c[0] = clip(t0 + t15a, min, max);
    c[stride] = clip(t1 + t14, min, max);
    c[2 * stride] = clip(t2 + t13a, min, max);
    c[3 * stride] = clip(t3 + t12, min, max);
    c[4 * stride] = clip(t4 + t11, min, max);
    c[5 * stride] = clip(t5 + t10a, min, max);
    c[6 * stride] = clip(t6 + t9, min, max);
    c[7 * stride] = clip(t7 + t8a, min, max);
    c[8 * stride] = clip(t7 - t8a, min, max);
    c[9 * stride] = clip(t6 - t9, min, max);
    c[10 * stride] = clip(t5 - t10a, min, max);
    c[11 * stride] = clip(t4 - t11, min, max);
    c[12 * stride] = clip(t3 - t12, min, max);
    c[13 * stride] = clip(t2 - t13a, min, max);
    c[14 * stride] = clip(t1 - t14, min, max);
    c[15 * stride] = clip(t0 - t15a, min, max);
}

/// dav1d `src/itx_1d.c:247-428` (`inv_dct32_1d_internal_c`) — mekanik port.
fn inv_dct32_1d_internal(c: &mut [i64], stride: usize, min: i64, max: i64, tx64: bool) {
    inv_dct16_1d_internal(c, stride << 1, min, max, tx64);

    let in1: i64 = c[stride];
    let in3: i64 = c[3 * stride];
    let in5: i64 = c[5 * stride];
    let in7: i64 = c[7 * stride];
    let in9: i64 = c[9 * stride];
    let in11: i64 = c[11 * stride];
    let in13: i64 = c[13 * stride];
    let in15: i64 = c[15 * stride];

    let mut t16a: i64;
    let mut t17a: i64;
    let mut t18a: i64;
    let mut t19a: i64;
    let mut t20a: i64;
    let mut t21a: i64;
    let mut t22a: i64;
    let mut t23a: i64;
    let mut t24a: i64;
    let mut t25a: i64;
    let mut t26a: i64;
    let mut t27a: i64;
    let mut t28a: i64;
    let mut t29a: i64;
    let mut t30a: i64;
    let mut t31a: i64;
    if tx64 {
        t16a = (in1 * 201 + 2048) >> 12;
        t17a = (in15 * -2751 + 2048) >> 12;
        t18a = (in9 * 1751 + 2048) >> 12;
        t19a = (in7 * -1380 + 2048) >> 12;
        t20a = (in5 * 995 + 2048) >> 12;
        t21a = (in11 * -2106 + 2048) >> 12;
        t22a = (in13 * 2440 + 2048) >> 12;
        t23a = (in3 * -601 + 2048) >> 12;
        t24a = (in3 * 4052 + 2048) >> 12;
        t25a = (in13 * 3290 + 2048) >> 12;
        t26a = (in11 * 3513 + 2048) >> 12;
        t27a = (in5 * 3973 + 2048) >> 12;
        t28a = (in7 * 3857 + 2048) >> 12;
        t29a = (in9 * 3703 + 2048) >> 12;
        t30a = (in15 * 3035 + 2048) >> 12;
        t31a = (in1 * 4091 + 2048) >> 12;
    } else {
        let in17: i64 = c[17 * stride];
        let in19: i64 = c[19 * stride];
        let in21: i64 = c[21 * stride];
        let in23: i64 = c[23 * stride];
        let in25: i64 = c[25 * stride];
        let in27: i64 = c[27 * stride];
        let in29: i64 = c[29 * stride];
        let in31: i64 = c[31 * stride];

        t16a = ((in1 * 201 - in31 * (4091 - 4096) + 2048) >> 12) - in31;
        t17a = ((in17 * (3035 - 4096) - in15 * 2751 + 2048) >> 12) + in17;
        t18a = ((in9 * 1751 - in23 * (3703 - 4096) + 2048) >> 12) - in23;
        t19a = ((in25 * (3857 - 4096) - in7 * 1380 + 2048) >> 12) + in25;
        t20a = ((in5 * 995 - in27 * (3973 - 4096) + 2048) >> 12) - in27;
        t21a = ((in21 * (3513 - 4096) - in11 * 2106 + 2048) >> 12) + in21;
        t22a = (in13 * 1220 - in19 * 1645 + 1024) >> 11;
        t23a = ((in29 * (4052 - 4096) - in3 * 601 + 2048) >> 12) + in29;
        t24a = ((in29 * 601 + in3 * (4052 - 4096) + 2048) >> 12) + in3;
        t25a = (in13 * 1645 + in19 * 1220 + 1024) >> 11;
        t26a = ((in21 * 2106 + in11 * (3513 - 4096) + 2048) >> 12) + in11;
        t27a = ((in5 * (3973 - 4096) + in27 * 995 + 2048) >> 12) + in5;
        t28a = ((in25 * 1380 + in7 * (3857 - 4096) + 2048) >> 12) + in7;
        t29a = ((in9 * (3703 - 4096) + in23 * 1751 + 2048) >> 12) + in9;
        t30a = ((in17 * 2751 + in15 * (3035 - 4096) + 2048) >> 12) + in15;
        t31a = ((in1 * (4091 - 4096) + in31 * 201 + 2048) >> 12) + in1;
    }

    let mut t16: i64 = clip(t16a + t17a, min, max);
    let mut t17: i64 = clip(t16a - t17a, min, max);
    let mut t18: i64 = clip(t19a - t18a, min, max);
    let mut t19: i64 = clip(t19a + t18a, min, max);
    let mut t20: i64 = clip(t20a + t21a, min, max);
    let mut t21: i64 = clip(t20a - t21a, min, max);
    let mut t22: i64 = clip(t23a - t22a, min, max);
    let mut t23: i64 = clip(t23a + t22a, min, max);
    let mut t24: i64 = clip(t24a + t25a, min, max);
    let mut t25: i64 = clip(t24a - t25a, min, max);
    let mut t26: i64 = clip(t27a - t26a, min, max);
    let mut t27: i64 = clip(t27a + t26a, min, max);
    let mut t28: i64 = clip(t28a + t29a, min, max);
    let mut t29: i64 = clip(t28a - t29a, min, max);
    let mut t30: i64 = clip(t31a - t30a, min, max);
    let mut t31: i64 = clip(t31a + t30a, min, max);

    t17a = ((t30 * 799 - t17 * (4017 - 4096) + 2048) >> 12) - t17;
    t30a = ((t30 * (4017 - 4096) + t17 * 799 + 2048) >> 12) + t30;
    t18a = ((-(t29 * (4017 - 4096) + t18 * 799) + 2048) >> 12) - t29;
    t29a = ((t29 * 799 - t18 * (4017 - 4096) + 2048) >> 12) - t18;
    t21a = (t26 * 1703 - t21 * 1138 + 1024) >> 11;
    t26a = (t26 * 1138 + t21 * 1703 + 1024) >> 11;
    t22a = (-(t25 * 1138 + t22 * 1703) + 1024) >> 11;
    t25a = (t25 * 1703 - t22 * 1138 + 1024) >> 11;

    t16a = clip(t16 + t19, min, max);
    t17 = clip(t17a + t18a, min, max);
    t18 = clip(t17a - t18a, min, max);
    t19a = clip(t16 - t19, min, max);
    t20a = clip(t23 - t20, min, max);
    t21 = clip(t22a - t21a, min, max);
    t22 = clip(t22a + t21a, min, max);
    t23a = clip(t23 + t20, min, max);
    t24a = clip(t24 + t27, min, max);
    t25 = clip(t25a + t26a, min, max);
    t26 = clip(t25a - t26a, min, max);
    t27a = clip(t24 - t27, min, max);
    t28a = clip(t31 - t28, min, max);
    t29 = clip(t30a - t29a, min, max);
    t30 = clip(t30a + t29a, min, max);
    t31a = clip(t31 + t28, min, max);

    t18a = ((t29 * 1567 - t18 * (3784 - 4096) + 2048) >> 12) - t18;
    t29a = ((t29 * (3784 - 4096) + t18 * 1567 + 2048) >> 12) + t29;
    t19 = ((t28a * 1567 - t19a * (3784 - 4096) + 2048) >> 12) - t19a;
    t28 = ((t28a * (3784 - 4096) + t19a * 1567 + 2048) >> 12) + t28a;
    t20 = ((-(t27a * (3784 - 4096) + t20a * 1567) + 2048) >> 12) - t27a;
    t27 = ((t27a * 1567 - t20a * (3784 - 4096) + 2048) >> 12) - t20a;
    t21a = ((-(t26 * (3784 - 4096) + t21 * 1567) + 2048) >> 12) - t26;
    t26a = ((t26 * 1567 - t21 * (3784 - 4096) + 2048) >> 12) - t21;

    t16 = clip(t16a + t23a, min, max);
    t17a = clip(t17 + t22, min, max);
    t18 = clip(t18a + t21a, min, max);
    t19a = clip(t19 + t20, min, max);
    t20a = clip(t19 - t20, min, max);
    t21 = clip(t18a - t21a, min, max);
    t22a = clip(t17 - t22, min, max);
    t23 = clip(t16a - t23a, min, max);
    t24 = clip(t31a - t24a, min, max);
    t25a = clip(t30 - t25, min, max);
    t26 = clip(t29a - t26a, min, max);
    t27a = clip(t28 - t27, min, max);
    t28a = clip(t28 + t27, min, max);
    t29 = clip(t29a + t26a, min, max);
    t30a = clip(t30 + t25, min, max);
    t31 = clip(t31a + t24a, min, max);

    t20 = ((t27a - t20a) * 181 + 128) >> 8;
    t27 = ((t27a + t20a) * 181 + 128) >> 8;
    t21a = ((t26 - t21) * 181 + 128) >> 8;
    t26a = ((t26 + t21) * 181 + 128) >> 8;
    t22 = ((t25a - t22a) * 181 + 128) >> 8;
    t25 = ((t25a + t22a) * 181 + 128) >> 8;
    t23a = ((t24 - t23) * 181 + 128) >> 8;
    t24a = ((t24 + t23) * 181 + 128) >> 8;

    let t0: i64 = c[0];
    let t1: i64 = c[2 * stride];
    let t2: i64 = c[4 * stride];
    let t3: i64 = c[6 * stride];
    let t4: i64 = c[8 * stride];
    let t5: i64 = c[10 * stride];
    let t6: i64 = c[12 * stride];
    let t7: i64 = c[14 * stride];
    let t8: i64 = c[16 * stride];
    let t9: i64 = c[18 * stride];
    let t10: i64 = c[20 * stride];
    let t11: i64 = c[22 * stride];
    let t12: i64 = c[24 * stride];
    let t13: i64 = c[26 * stride];
    let t14: i64 = c[28 * stride];
    let t15: i64 = c[30 * stride];

    c[0] = clip(t0 + t31, min, max);
    c[stride] = clip(t1 + t30a, min, max);
    c[2 * stride] = clip(t2 + t29, min, max);
    c[3 * stride] = clip(t3 + t28a, min, max);
    c[4 * stride] = clip(t4 + t27, min, max);
    c[5 * stride] = clip(t5 + t26a, min, max);
    c[6 * stride] = clip(t6 + t25, min, max);
    c[7 * stride] = clip(t7 + t24a, min, max);
    c[8 * stride] = clip(t8 + t23a, min, max);
    c[9 * stride] = clip(t9 + t22, min, max);
    c[10 * stride] = clip(t10 + t21a, min, max);
    c[11 * stride] = clip(t11 + t20, min, max);
    c[12 * stride] = clip(t12 + t19a, min, max);
    c[13 * stride] = clip(t13 + t18, min, max);
    c[14 * stride] = clip(t14 + t17a, min, max);
    c[15 * stride] = clip(t15 + t16, min, max);
    c[16 * stride] = clip(t15 - t16, min, max);
    c[17 * stride] = clip(t14 - t17a, min, max);
    c[18 * stride] = clip(t13 - t18, min, max);
    c[19 * stride] = clip(t12 - t19a, min, max);
    c[20 * stride] = clip(t11 - t20, min, max);
    c[21 * stride] = clip(t10 - t21a, min, max);
    c[22 * stride] = clip(t9 - t22, min, max);
    c[23 * stride] = clip(t8 - t23a, min, max);
    c[24 * stride] = clip(t7 - t24a, min, max);
    c[25 * stride] = clip(t6 - t25, min, max);
    c[26 * stride] = clip(t5 - t26a, min, max);
    c[27 * stride] = clip(t4 - t27, min, max);
    c[28 * stride] = clip(t3 - t28a, min, max);
    c[29 * stride] = clip(t2 - t29, min, max);
    c[30 * stride] = clip(t1 - t30a, min, max);
    c[31 * stride] = clip(t0 - t31, min, max);
}

/// dav1d `src/itx_1d.c:436-781` (`inv_dct64_1d_c`) — mekanik port (64 nokta; yalnız
/// ilk 32 giriş okunur, üst yarı sıfır kabul edilir).
fn inv_dct64_1d_internal(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_dct32_1d_internal(c, stride << 1, min, max, true);

    let in1: i64 = c[stride];
    let in3: i64 = c[3 * stride];
    let in5: i64 = c[5 * stride];
    let in7: i64 = c[7 * stride];
    let in9: i64 = c[9 * stride];
    let in11: i64 = c[11 * stride];
    let in13: i64 = c[13 * stride];
    let in15: i64 = c[15 * stride];
    let in17: i64 = c[17 * stride];
    let in19: i64 = c[19 * stride];
    let in21: i64 = c[21 * stride];
    let in23: i64 = c[23 * stride];
    let in25: i64 = c[25 * stride];
    let in27: i64 = c[27 * stride];
    let in29: i64 = c[29 * stride];
    let in31: i64 = c[31 * stride];

    let mut t32a: i64 = (in1 * 101 + 2048) >> 12;
    let mut t33a: i64 = (in31 * -2824 + 2048) >> 12;
    let mut t34a: i64 = (in17 * 1660 + 2048) >> 12;
    let mut t35a: i64 = (in15 * -1474 + 2048) >> 12;
    let mut t36a: i64 = (in9 * 897 + 2048) >> 12;
    let mut t37a: i64 = (in23 * -2191 + 2048) >> 12;
    let mut t38a: i64 = (in25 * 2359 + 2048) >> 12;
    let mut t39a: i64 = (in7 * -700 + 2048) >> 12;
    let mut t40a: i64 = (in5 * 501 + 2048) >> 12;
    let mut t41a: i64 = (in27 * -2520 + 2048) >> 12;
    let mut t42a: i64 = (in21 * 2019 + 2048) >> 12;
    let mut t43a: i64 = (in11 * -1092 + 2048) >> 12;
    let mut t44a: i64 = (in13 * 1285 + 2048) >> 12;
    let mut t45a: i64 = (in19 * -1842 + 2048) >> 12;
    let mut t46a: i64 = (in29 * 2675 + 2048) >> 12;
    let mut t47a: i64 = (in3 * -301 + 2048) >> 12;
    let mut t48a: i64 = (in3 * 4085 + 2048) >> 12;
    let mut t49a: i64 = (in29 * 3102 + 2048) >> 12;
    let mut t50a: i64 = (in19 * 3659 + 2048) >> 12;
    let mut t51a: i64 = (in13 * 3889 + 2048) >> 12;
    let mut t52a: i64 = (in11 * 3948 + 2048) >> 12;
    let mut t53a: i64 = (in21 * 3564 + 2048) >> 12;
    let mut t54a: i64 = (in27 * 3229 + 2048) >> 12;
    let mut t55a: i64 = (in5 * 4065 + 2048) >> 12;
    let mut t56a: i64 = (in7 * 4036 + 2048) >> 12;
    let mut t57a: i64 = (in25 * 3349 + 2048) >> 12;
    let mut t58a: i64 = (in23 * 3461 + 2048) >> 12;
    let mut t59a: i64 = (in9 * 3996 + 2048) >> 12;
    let mut t60a: i64 = (in15 * 3822 + 2048) >> 12;
    let mut t61a: i64 = (in17 * 3745 + 2048) >> 12;
    let mut t62a: i64 = (in31 * 2967 + 2048) >> 12;
    let mut t63a: i64 = (in1 * 4095 + 2048) >> 12;

    let mut t32: i64 = clip(t32a + t33a, min, max);
    let mut t33: i64 = clip(t32a - t33a, min, max);
    let mut t34: i64 = clip(t35a - t34a, min, max);
    let mut t35: i64 = clip(t35a + t34a, min, max);
    let mut t36: i64 = clip(t36a + t37a, min, max);
    let mut t37: i64 = clip(t36a - t37a, min, max);
    let mut t38: i64 = clip(t39a - t38a, min, max);
    let mut t39: i64 = clip(t39a + t38a, min, max);
    let mut t40: i64 = clip(t40a + t41a, min, max);
    let mut t41: i64 = clip(t40a - t41a, min, max);
    let mut t42: i64 = clip(t43a - t42a, min, max);
    let mut t43: i64 = clip(t43a + t42a, min, max);
    let mut t44: i64 = clip(t44a + t45a, min, max);
    let mut t45: i64 = clip(t44a - t45a, min, max);
    let mut t46: i64 = clip(t47a - t46a, min, max);
    let mut t47: i64 = clip(t47a + t46a, min, max);
    let mut t48: i64 = clip(t48a + t49a, min, max);
    let mut t49: i64 = clip(t48a - t49a, min, max);
    let mut t50: i64 = clip(t51a - t50a, min, max);
    let mut t51: i64 = clip(t51a + t50a, min, max);
    let mut t52: i64 = clip(t52a + t53a, min, max);
    let mut t53: i64 = clip(t52a - t53a, min, max);
    let mut t54: i64 = clip(t55a - t54a, min, max);
    let mut t55: i64 = clip(t55a + t54a, min, max);
    let mut t56: i64 = clip(t56a + t57a, min, max);
    let mut t57: i64 = clip(t56a - t57a, min, max);
    let mut t58: i64 = clip(t59a - t58a, min, max);
    let mut t59: i64 = clip(t59a + t58a, min, max);
    let mut t60: i64 = clip(t60a + t61a, min, max);
    let mut t61: i64 = clip(t60a - t61a, min, max);
    let mut t62: i64 = clip(t63a - t62a, min, max);
    let mut t63: i64 = clip(t63a + t62a, min, max);

    t33a = ((t33 * (4096 - 4076) + t62 * 401 + 2048) >> 12) - t33;
    t34a = ((t34 * -401 + t61 * (4096 - 4076) + 2048) >> 12) - t61;
    t37a = (t37 * -1299 + t58 * 1583 + 1024) >> 11;
    t38a = (t38 * -1583 + t57 * -1299 + 1024) >> 11;
    t41a = ((t41 * (4096 - 3612) + t54 * 1931 + 2048) >> 12) - t41;
    t42a = ((t42 * -1931 + t53 * (4096 - 3612) + 2048) >> 12) - t53;
    t45a = ((t45 * -1189 + t50 * (3920 - 4096) + 2048) >> 12) + t50;
    t46a = ((t46 * (4096 - 3920) + t49 * -1189 + 2048) >> 12) - t46;
    t49a = ((t46 * -1189 + t49 * (3920 - 4096) + 2048) >> 12) + t49;
    t50a = ((t45 * (3920 - 4096) + t50 * 1189 + 2048) >> 12) + t45;
    t53a = ((t42 * (4096 - 3612) + t53 * 1931 + 2048) >> 12) - t42;
    t54a = ((t41 * 1931 + t54 * (3612 - 4096) + 2048) >> 12) + t54;
    t57a = (t38 * -1299 + t57 * 1583 + 1024) >> 11;
    t58a = (t37 * 1583 + t58 * 1299 + 1024) >> 11;
    t61a = ((t34 * (4096 - 4076) + t61 * 401 + 2048) >> 12) - t34;
    t62a = ((t33 * 401 + t62 * (4076 - 4096) + 2048) >> 12) + t62;

    t32a = clip(t32 + t35, min, max);
    t33 = clip(t33a + t34a, min, max);
    t34 = clip(t33a - t34a, min, max);
    t35a = clip(t32 - t35, min, max);
    t36a = clip(t39 - t36, min, max);
    t37 = clip(t38a - t37a, min, max);
    t38 = clip(t38a + t37a, min, max);
    t39a = clip(t39 + t36, min, max);
    t40a = clip(t40 + t43, min, max);
    t41 = clip(t41a + t42a, min, max);
    t42 = clip(t41a - t42a, min, max);
    t43a = clip(t40 - t43, min, max);
    t44a = clip(t47 - t44, min, max);
    t45 = clip(t46a - t45a, min, max);
    t46 = clip(t46a + t45a, min, max);
    t47a = clip(t47 + t44, min, max);
    t48a = clip(t48 + t51, min, max);
    t49 = clip(t49a + t50a, min, max);
    t50 = clip(t49a - t50a, min, max);
    t51a = clip(t48 - t51, min, max);
    t52a = clip(t55 - t52, min, max);
    t53 = clip(t54a - t53a, min, max);
    t54 = clip(t54a + t53a, min, max);
    t55a = clip(t55 + t52, min, max);
    t56a = clip(t56 + t59, min, max);
    t57 = clip(t57a + t58a, min, max);
    t58 = clip(t57a - t58a, min, max);
    t59a = clip(t56 - t59, min, max);
    t60a = clip(t63 - t60, min, max);
    t61 = clip(t62a - t61a, min, max);
    t62 = clip(t62a + t61a, min, max);
    t63a = clip(t63 + t60, min, max);

    t34a = ((t34 * (4096 - 4017) + t61 * 799 + 2048) >> 12) - t34;
    t35 = ((t35a * (4096 - 4017) + t60a * 799 + 2048) >> 12) - t35a;
    t36 = ((t36a * -799 + t59a * (4096 - 4017) + 2048) >> 12) - t59a;
    t37a = ((t37 * -799 + t58 * (4096 - 4017) + 2048) >> 12) - t58;
    t42a = (t42 * -1138 + t53 * 1703 + 1024) >> 11;
    t43 = (t43a * -1138 + t52a * 1703 + 1024) >> 11;
    t44 = (t44a * -1703 + t51a * -1138 + 1024) >> 11;
    t45a = (t45 * -1703 + t50 * -1138 + 1024) >> 11;
    t50a = (t45 * -1138 + t50 * 1703 + 1024) >> 11;
    t51 = (t44a * -1138 + t51a * 1703 + 1024) >> 11;
    t52 = (t43a * 1703 + t52a * 1138 + 1024) >> 11;
    t53a = (t42 * 1703 + t53 * 1138 + 1024) >> 11;
    t58a = ((t37 * (4096 - 4017) + t58 * 799 + 2048) >> 12) - t37;
    t59 = ((t36a * (4096 - 4017) + t59a * 799 + 2048) >> 12) - t36a;
    t60 = ((t35a * 799 + t60a * (4017 - 4096) + 2048) >> 12) + t60a;
    t61a = ((t34 * 799 + t61 * (4017 - 4096) + 2048) >> 12) + t61;

    t32 = clip(t32a + t39a, min, max);
    t33a = clip(t33 + t38, min, max);
    t34 = clip(t34a + t37a, min, max);
    t35a = clip(t35 + t36, min, max);
    t36a = clip(t35 - t36, min, max);
    t37 = clip(t34a - t37a, min, max);
    t38a = clip(t33 - t38, min, max);
    t39 = clip(t32a - t39a, min, max);
    t40 = clip(t47a - t40a, min, max);
    t41a = clip(t46 - t41, min, max);
    t42 = clip(t45a - t42a, min, max);
    t43a = clip(t44 - t43, min, max);
    t44a = clip(t44 + t43, min, max);
    t45 = clip(t45a + t42a, min, max);
    t46a = clip(t46 + t41, min, max);
    t47 = clip(t47a + t40a, min, max);
    t48 = clip(t48a + t55a, min, max);
    t49a = clip(t49 + t54, min, max);
    t50 = clip(t50a + t53a, min, max);
    t51a = clip(t51 + t52, min, max);
    t52a = clip(t51 - t52, min, max);
    t53 = clip(t50a - t53a, min, max);
    t54a = clip(t49 - t54, min, max);
    t55 = clip(t48a - t55a, min, max);
    t56 = clip(t63a - t56a, min, max);
    t57a = clip(t62 - t57, min, max);
    t58 = clip(t61a - t58a, min, max);
    t59a = clip(t60 - t59, min, max);
    t60a = clip(t60 + t59, min, max);
    t61 = clip(t61a + t58a, min, max);
    t62a = clip(t62 + t57, min, max);
    t63 = clip(t63a + t56a, min, max);

    t36 = ((t36a * (4096 - 3784) + t59a * 1567 + 2048) >> 12) - t36a;
    t37a = ((t37 * (4096 - 3784) + t58 * 1567 + 2048) >> 12) - t37;
    t38 = ((t38a * (4096 - 3784) + t57a * 1567 + 2048) >> 12) - t38a;
    t39a = ((t39 * (4096 - 3784) + t56 * 1567 + 2048) >> 12) - t39;
    t40a = ((t40 * -1567 + t55 * (4096 - 3784) + 2048) >> 12) - t55;
    t41 = ((t41a * -1567 + t54a * (4096 - 3784) + 2048) >> 12) - t54a;
    t42a = ((t42 * -1567 + t53 * (4096 - 3784) + 2048) >> 12) - t53;
    t43 = ((t43a * -1567 + t52a * (4096 - 3784) + 2048) >> 12) - t52a;
    t52 = ((t43a * (4096 - 3784) + t52a * 1567 + 2048) >> 12) - t43a;
    t53a = ((t42 * (4096 - 3784) + t53 * 1567 + 2048) >> 12) - t42;
    t54 = ((t41a * (4096 - 3784) + t54a * 1567 + 2048) >> 12) - t41a;
    t55a = ((t40 * (4096 - 3784) + t55 * 1567 + 2048) >> 12) - t40;
    t56a = ((t39 * 1567 + t56 * (3784 - 4096) + 2048) >> 12) + t56;
    t57 = ((t38a * 1567 + t57a * (3784 - 4096) + 2048) >> 12) + t57a;
    t58a = ((t37 * 1567 + t58 * (3784 - 4096) + 2048) >> 12) + t58;
    t59 = ((t36a * 1567 + t59a * (3784 - 4096) + 2048) >> 12) + t59a;

    t32a = clip(t32 + t47, min, max);
    t33 = clip(t33a + t46a, min, max);
    t34a = clip(t34 + t45, min, max);
    t35 = clip(t35a + t44a, min, max);
    t36a = clip(t36 + t43, min, max);
    t37 = clip(t37a + t42a, min, max);
    t38a = clip(t38 + t41, min, max);
    t39 = clip(t39a + t40a, min, max);
    t40 = clip(t39a - t40a, min, max);
    t41a = clip(t38 - t41, min, max);
    t42 = clip(t37a - t42a, min, max);
    t43a = clip(t36 - t43, min, max);
    t44 = clip(t35a - t44a, min, max);
    t45a = clip(t34 - t45, min, max);
    t46 = clip(t33a - t46a, min, max);
    t47a = clip(t32 - t47, min, max);
    t48a = clip(t63 - t48, min, max);
    t49 = clip(t62a - t49a, min, max);
    t50a = clip(t61 - t50, min, max);
    t51 = clip(t60a - t51a, min, max);
    t52a = clip(t59 - t52, min, max);
    t53 = clip(t58a - t53a, min, max);
    t54a = clip(t57 - t54, min, max);
    t55 = clip(t56a - t55a, min, max);
    t56 = clip(t56a + t55a, min, max);
    t57a = clip(t57 + t54, min, max);
    t58 = clip(t58a + t53a, min, max);
    t59a = clip(t59 + t52, min, max);
    t60 = clip(t60a + t51a, min, max);
    t61a = clip(t61 + t50, min, max);
    t62 = clip(t62a + t49a, min, max);
    t63a = clip(t63 + t48, min, max);

    t40a = ((t55 - t40) * 181 + 128) >> 8;
    t41 = ((t54a - t41a) * 181 + 128) >> 8;
    t42a = ((t53 - t42) * 181 + 128) >> 8;
    t43 = ((t52a - t43a) * 181 + 128) >> 8;
    t44a = ((t51 - t44) * 181 + 128) >> 8;
    t45 = ((t50a - t45a) * 181 + 128) >> 8;
    t46a = ((t49 - t46) * 181 + 128) >> 8;
    t47 = ((t48a - t47a) * 181 + 128) >> 8;
    t48 = ((t47a + t48a) * 181 + 128) >> 8;
    t49a = ((t46 + t49) * 181 + 128) >> 8;
    t50 = ((t45a + t50a) * 181 + 128) >> 8;
    t51a = ((t44 + t51) * 181 + 128) >> 8;
    t52 = ((t43a + t52a) * 181 + 128) >> 8;
    t53a = ((t42 + t53) * 181 + 128) >> 8;
    t54 = ((t41a + t54a) * 181 + 128) >> 8;
    t55a = ((t40 + t55) * 181 + 128) >> 8;

    let t0: i64 = c[0];
    let t1: i64 = c[2 * stride];
    let t2: i64 = c[4 * stride];
    let t3: i64 = c[6 * stride];
    let t4: i64 = c[8 * stride];
    let t5: i64 = c[10 * stride];
    let t6: i64 = c[12 * stride];
    let t7: i64 = c[14 * stride];
    let t8: i64 = c[16 * stride];
    let t9: i64 = c[18 * stride];
    let t10: i64 = c[20 * stride];
    let t11: i64 = c[22 * stride];
    let t12: i64 = c[24 * stride];
    let t13: i64 = c[26 * stride];
    let t14: i64 = c[28 * stride];
    let t15: i64 = c[30 * stride];
    let t16: i64 = c[32 * stride];
    let t17: i64 = c[34 * stride];
    let t18: i64 = c[36 * stride];
    let t19: i64 = c[38 * stride];
    let t20: i64 = c[40 * stride];
    let t21: i64 = c[42 * stride];
    let t22: i64 = c[44 * stride];
    let t23: i64 = c[46 * stride];
    let t24: i64 = c[48 * stride];
    let t25: i64 = c[50 * stride];
    let t26: i64 = c[52 * stride];
    let t27: i64 = c[54 * stride];
    let t28: i64 = c[56 * stride];
    let t29: i64 = c[58 * stride];
    let t30: i64 = c[60 * stride];
    let t31: i64 = c[62 * stride];

    c[0] = clip(t0 + t63a, min, max);
    c[stride] = clip(t1 + t62, min, max);
    c[2 * stride] = clip(t2 + t61a, min, max);
    c[3 * stride] = clip(t3 + t60, min, max);
    c[4 * stride] = clip(t4 + t59a, min, max);
    c[5 * stride] = clip(t5 + t58, min, max);
    c[6 * stride] = clip(t6 + t57a, min, max);
    c[7 * stride] = clip(t7 + t56, min, max);
    c[8 * stride] = clip(t8 + t55a, min, max);
    c[9 * stride] = clip(t9 + t54, min, max);
    c[10 * stride] = clip(t10 + t53a, min, max);
    c[11 * stride] = clip(t11 + t52, min, max);
    c[12 * stride] = clip(t12 + t51a, min, max);
    c[13 * stride] = clip(t13 + t50, min, max);
    c[14 * stride] = clip(t14 + t49a, min, max);
    c[15 * stride] = clip(t15 + t48, min, max);
    c[16 * stride] = clip(t16 + t47, min, max);
    c[17 * stride] = clip(t17 + t46a, min, max);
    c[18 * stride] = clip(t18 + t45, min, max);
    c[19 * stride] = clip(t19 + t44a, min, max);
    c[20 * stride] = clip(t20 + t43, min, max);
    c[21 * stride] = clip(t21 + t42a, min, max);
    c[22 * stride] = clip(t22 + t41, min, max);
    c[23 * stride] = clip(t23 + t40a, min, max);
    c[24 * stride] = clip(t24 + t39, min, max);
    c[25 * stride] = clip(t25 + t38a, min, max);
    c[26 * stride] = clip(t26 + t37, min, max);
    c[27 * stride] = clip(t27 + t36a, min, max);
    c[28 * stride] = clip(t28 + t35, min, max);
    c[29 * stride] = clip(t29 + t34a, min, max);
    c[30 * stride] = clip(t30 + t33, min, max);
    c[31 * stride] = clip(t31 + t32a, min, max);
    c[32 * stride] = clip(t31 - t32a, min, max);
    c[33 * stride] = clip(t30 - t33, min, max);
    c[34 * stride] = clip(t29 - t34a, min, max);
    c[35 * stride] = clip(t28 - t35, min, max);
    c[36 * stride] = clip(t27 - t36a, min, max);
    c[37 * stride] = clip(t26 - t37, min, max);
    c[38 * stride] = clip(t25 - t38a, min, max);
    c[39 * stride] = clip(t24 - t39, min, max);
    c[40 * stride] = clip(t23 - t40a, min, max);
    c[41 * stride] = clip(t22 - t41, min, max);
    c[42 * stride] = clip(t21 - t42a, min, max);
    c[43 * stride] = clip(t20 - t43, min, max);
    c[44 * stride] = clip(t19 - t44a, min, max);
    c[45 * stride] = clip(t18 - t45, min, max);
    c[46 * stride] = clip(t17 - t46a, min, max);
    c[47 * stride] = clip(t16 - t47, min, max);
    c[48 * stride] = clip(t15 - t48, min, max);
    c[49 * stride] = clip(t14 - t49a, min, max);
    c[50 * stride] = clip(t13 - t50, min, max);
    c[51 * stride] = clip(t12 - t51a, min, max);
    c[52 * stride] = clip(t11 - t52, min, max);
    c[53 * stride] = clip(t10 - t53a, min, max);
    c[54 * stride] = clip(t9 - t54, min, max);
    c[55 * stride] = clip(t8 - t55a, min, max);
    c[56 * stride] = clip(t7 - t56, min, max);
    c[57 * stride] = clip(t6 - t57a, min, max);
    c[58 * stride] = clip(t5 - t58, min, max);
    c[59 * stride] = clip(t4 - t59a, min, max);
    c[60 * stride] = clip(t3 - t60, min, max);
    c[61 * stride] = clip(t2 - t61a, min, max);
    c[62 * stride] = clip(t1 - t62, min, max);
    c[63 * stride] = clip(t0 - t63a, min, max);
}

/// dav1d `src/itx_1d.c:784-802` (`inv_adst4_1d_internal_c`) — mekanik port; `out_s = -stride`
/// flip durumu `flip` ile ifade edilir (negatif stride yerine indeks eşlemesi).
fn inv_adst4_1d_internal(c: &mut [i64], stride: usize, _min: i64, _max: i64, flip: bool) {
    let out_idx = |k: usize| {
        if flip {
            (4 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: i64 = c[0];
    let in1: i64 = c[stride];
    let in2: i64 = c[2 * stride];
    let in3: i64 = c[3 * stride];

    c[out_idx(0)] =
        ((1321 * in0 + (3803 - 4096) * in2 + (2482 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in2
            + in3
            + in1;
    c[out_idx(1)] =
        (((2482 - 4096) * in0 - 1321 * in2 - (3803 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            - in3
            + in1;
    c[out_idx(2)] = (209 * (in0 - in2 + in3) + 128) >> 8;
    c[out_idx(3)] =
        (((3803 - 4096) * in0 + (2482 - 4096) * in2 - 1321 * in3 - (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            + in2
            - in1;
}

/// dav1d `src/itx_1d.c:805-851` (`inv_adst8_1d_internal_c`) — mekanik port; `out_s = -stride`
/// flip durumu `flip` ile ifade edilir (negatif stride yerine indeks eşlemesi).
fn inv_adst8_1d_internal(c: &mut [i64], stride: usize, min: i64, max: i64, flip: bool) {
    let out_idx = |k: usize| {
        if flip {
            (8 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: i64 = c[0];
    let in1: i64 = c[stride];
    let in2: i64 = c[2 * stride];
    let in3: i64 = c[3 * stride];
    let in4: i64 = c[4 * stride];
    let in5: i64 = c[5 * stride];
    let in6: i64 = c[6 * stride];
    let in7: i64 = c[7 * stride];

    let t0a: i64 = (((4076 - 4096) * in7 + 401 * in0 + 2048) >> 12) + in7;
    let t1a: i64 = ((401 * in7 - (4076 - 4096) * in0 + 2048) >> 12) - in0;
    let t2a: i64 = (((3612 - 4096) * in5 + 1931 * in2 + 2048) >> 12) + in5;
    let t3a: i64 = ((1931 * in5 - (3612 - 4096) * in2 + 2048) >> 12) - in2;
    let mut t4a: i64 = (1299 * in3 + 1583 * in4 + 1024) >> 11;
    let mut t5a: i64 = (1583 * in3 - 1299 * in4 + 1024) >> 11;
    let mut t6a: i64 = ((1189 * in1 + (3920 - 4096) * in6 + 2048) >> 12) + in6;
    let mut t7a: i64 = (((3920 - 4096) * in1 - 1189 * in6 + 2048) >> 12) + in1;

    let t0: i64 = clip(t0a + t4a, min, max);
    let t1: i64 = clip(t1a + t5a, min, max);
    let mut t2: i64 = clip(t2a + t6a, min, max);
    let mut t3: i64 = clip(t3a + t7a, min, max);
    let t4: i64 = clip(t0a - t4a, min, max);
    let t5: i64 = clip(t1a - t5a, min, max);
    let mut t6: i64 = clip(t2a - t6a, min, max);
    let mut t7: i64 = clip(t3a - t7a, min, max);

    t4a = (((3784 - 4096) * t4 + 1567 * t5 + 2048) >> 12) + t4;
    t5a = ((1567 * t4 - (3784 - 4096) * t5 + 2048) >> 12) - t5;
    t6a = (((3784 - 4096) * t7 - 1567 * t6 + 2048) >> 12) + t7;
    t7a = ((1567 * t7 + (3784 - 4096) * t6 + 2048) >> 12) + t6;

    c[out_idx(0)] = clip(t0 + t2, min, max);
    c[out_idx(7)] = -clip(t1 + t3, min, max);
    t2 = clip(t0 - t2, min, max);
    t3 = clip(t1 - t3, min, max);
    c[out_idx(1)] = -clip(t4a + t6a, min, max);
    c[out_idx(6)] = clip(t5a + t7a, min, max);
    t6 = clip(t4a - t6a, min, max);
    t7 = clip(t5a - t7a, min, max);

    c[out_idx(3)] = -(((t2 + t3) * 181 + 128) >> 8);
    c[out_idx(4)] = ((t2 - t3) * 181 + 128) >> 8;
    c[out_idx(2)] = ((t6 + t7) * 181 + 128) >> 8;
    c[out_idx(5)] = -(((t6 - t7) * 181 + 128) >> 8);
}

/// dav1d `src/itx_1d.c:854-962` (`inv_adst16_1d_internal_c`) — mekanik port; `out_s = -stride`
/// flip durumu `flip` ile ifade edilir (negatif stride yerine indeks eşlemesi).
fn inv_adst16_1d_internal(c: &mut [i64], stride: usize, min: i64, max: i64, flip: bool) {
    let out_idx = |k: usize| {
        if flip {
            (16 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: i64 = c[0];
    let in1: i64 = c[stride];
    let in2: i64 = c[2 * stride];
    let in3: i64 = c[3 * stride];
    let in4: i64 = c[4 * stride];
    let in5: i64 = c[5 * stride];
    let in6: i64 = c[6 * stride];
    let in7: i64 = c[7 * stride];
    let in8: i64 = c[8 * stride];
    let in9: i64 = c[9 * stride];
    let in10: i64 = c[10 * stride];
    let in11: i64 = c[11 * stride];
    let in12: i64 = c[12 * stride];
    let in13: i64 = c[13 * stride];
    let in14: i64 = c[14 * stride];
    let in15: i64 = c[15 * stride];

    let mut t0: i64 = ((in15 * (4091 - 4096) + in0 * 201 + 2048) >> 12) + in15;
    let mut t1: i64 = ((in15 * 201 - in0 * (4091 - 4096) + 2048) >> 12) - in0;
    let mut t2: i64 = ((in13 * (3973 - 4096) + in2 * 995 + 2048) >> 12) + in13;
    let mut t3: i64 = ((in13 * 995 - in2 * (3973 - 4096) + 2048) >> 12) - in2;
    let mut t4: i64 = ((in11 * (3703 - 4096) + in4 * 1751 + 2048) >> 12) + in11;
    let mut t5: i64 = ((in11 * 1751 - in4 * (3703 - 4096) + 2048) >> 12) - in4;
    let mut t6: i64 = (in9 * 1645 + in6 * 1220 + 1024) >> 11;
    let mut t7: i64 = (in9 * 1220 - in6 * 1645 + 1024) >> 11;
    let mut t8: i64 = ((in7 * 2751 + in8 * (3035 - 4096) + 2048) >> 12) + in8;
    let mut t9: i64 = ((in7 * (3035 - 4096) - in8 * 2751 + 2048) >> 12) + in7;
    let mut t10: i64 = ((in5 * 2106 + in10 * (3513 - 4096) + 2048) >> 12) + in10;
    let mut t11: i64 = ((in5 * (3513 - 4096) - in10 * 2106 + 2048) >> 12) + in5;
    let mut t12: i64 = ((in3 * 1380 + in12 * (3857 - 4096) + 2048) >> 12) + in12;
    let mut t13: i64 = ((in3 * (3857 - 4096) - in12 * 1380 + 2048) >> 12) + in3;
    let mut t14: i64 = ((in1 * 601 + in14 * (4052 - 4096) + 2048) >> 12) + in14;
    let mut t15: i64 = ((in1 * (4052 - 4096) - in14 * 601 + 2048) >> 12) + in1;

    let t0a: i64 = clip(t0 + t8, min, max);
    let t1a: i64 = clip(t1 + t9, min, max);
    let mut t2a: i64 = clip(t2 + t10, min, max);
    let mut t3a: i64 = clip(t3 + t11, min, max);
    let mut t4a: i64 = clip(t4 + t12, min, max);
    let mut t5a: i64 = clip(t5 + t13, min, max);
    let mut t6a: i64 = clip(t6 + t14, min, max);
    let mut t7a: i64 = clip(t7 + t15, min, max);
    let mut t8a: i64 = clip(t0 - t8, min, max);
    let mut t9a: i64 = clip(t1 - t9, min, max);
    let mut t10a: i64 = clip(t2 - t10, min, max);
    let mut t11a: i64 = clip(t3 - t11, min, max);
    let mut t12a: i64 = clip(t4 - t12, min, max);
    let mut t13a: i64 = clip(t5 - t13, min, max);
    let mut t14a: i64 = clip(t6 - t14, min, max);
    let mut t15a: i64 = clip(t7 - t15, min, max);

    t8 = ((t8a * (4017 - 4096) + t9a * 799 + 2048) >> 12) + t8a;
    t9 = ((t8a * 799 - t9a * (4017 - 4096) + 2048) >> 12) - t9a;
    t10 = ((t10a * 2276 + t11a * (3406 - 4096) + 2048) >> 12) + t11a;
    t11 = ((t10a * (3406 - 4096) - t11a * 2276 + 2048) >> 12) + t10a;
    t12 = ((t13a * (4017 - 4096) - t12a * 799 + 2048) >> 12) + t13a;
    t13 = ((t13a * 799 + t12a * (4017 - 4096) + 2048) >> 12) + t12a;
    t14 = ((t15a * 2276 - t14a * (3406 - 4096) + 2048) >> 12) - t14a;
    t15 = ((t15a * (3406 - 4096) + t14a * 2276 + 2048) >> 12) + t15a;

    t0 = clip(t0a + t4a, min, max);
    t1 = clip(t1a + t5a, min, max);
    t2 = clip(t2a + t6a, min, max);
    t3 = clip(t3a + t7a, min, max);
    t4 = clip(t0a - t4a, min, max);
    t5 = clip(t1a - t5a, min, max);
    t6 = clip(t2a - t6a, min, max);
    t7 = clip(t3a - t7a, min, max);
    t8a = clip(t8 + t12, min, max);
    t9a = clip(t9 + t13, min, max);
    t10a = clip(t10 + t14, min, max);
    t11a = clip(t11 + t15, min, max);
    t12a = clip(t8 - t12, min, max);
    t13a = clip(t9 - t13, min, max);
    t14a = clip(t10 - t14, min, max);
    t15a = clip(t11 - t15, min, max);

    t4a = ((t4 * (3784 - 4096) + t5 * 1567 + 2048) >> 12) + t4;
    t5a = ((t4 * 1567 - t5 * (3784 - 4096) + 2048) >> 12) - t5;
    t6a = ((t7 * (3784 - 4096) - t6 * 1567 + 2048) >> 12) + t7;
    t7a = ((t7 * 1567 + t6 * (3784 - 4096) + 2048) >> 12) + t6;
    t12 = ((t12a * (3784 - 4096) + t13a * 1567 + 2048) >> 12) + t12a;
    t13 = ((t12a * 1567 - t13a * (3784 - 4096) + 2048) >> 12) - t13a;
    t14 = ((t15a * (3784 - 4096) - t14a * 1567 + 2048) >> 12) + t15a;
    t15 = ((t15a * 1567 + t14a * (3784 - 4096) + 2048) >> 12) + t14a;

    c[out_idx(0)] = clip(t0 + t2, min, max);
    c[out_idx(15)] = -clip(t1 + t3, min, max);
    t2a = clip(t0 - t2, min, max);
    t3a = clip(t1 - t3, min, max);
    c[out_idx(3)] = -clip(t4a + t6a, min, max);
    c[out_idx(12)] = clip(t5a + t7a, min, max);
    t6 = clip(t4a - t6a, min, max);
    t7 = clip(t5a - t7a, min, max);
    c[out_idx(1)] = -clip(t8a + t10a, min, max);
    c[out_idx(14)] = clip(t9a + t11a, min, max);
    t10 = clip(t8a - t10a, min, max);
    t11 = clip(t9a - t11a, min, max);
    c[out_idx(2)] = clip(t12 + t14, min, max);
    c[out_idx(13)] = -clip(t13 + t15, min, max);
    t14a = clip(t12 - t14, min, max);
    t15a = clip(t13 - t15, min, max);

    c[out_idx(7)] = -(((t2a + t3a) * 181 + 128) >> 8);
    c[out_idx(8)] = ((t2a - t3a) * 181 + 128) >> 8;
    c[out_idx(4)] = ((t6 + t7) * 181 + 128) >> 8;
    c[out_idx(11)] = -(((t6 - t7) * 181 + 128) >> 8);
    c[out_idx(6)] = ((t10 + t11) * 181 + 128) >> 8;
    c[out_idx(9)] = -(((t10 - t11) * 181 + 128) >> 8);
    c[out_idx(5)] = -(((t14a + t15a) * 181 + 128) >> 8);
    c[out_idx(10)] = ((t14a - t15a) * 181 + 128) >> 8;
}

/// dav1d `src/itx_1d.c:983-991` (`inv_identity4_1d_c`) — mekanik port.
fn inv_identity4_1d(c: &mut [i64], stride: usize, _min: i64, _max: i64) {
    for i in 0..4 {
        let val: i64 = c[stride * i];
        c[stride * i] = val + ((val * 1697 + 2048) >> 12);
    }
}

/// dav1d `src/itx_1d.c:993-999` (`inv_identity8_1d_c`) — mekanik port.
fn inv_identity8_1d(c: &mut [i64], stride: usize, _min: i64, _max: i64) {
    for i in 0..8 {
        c[stride * i] *= 2;
    }
}

/// dav1d `src/itx_1d.c:1001-1009` (`inv_identity16_1d_c`) — mekanik port.
fn inv_identity16_1d(c: &mut [i64], stride: usize, _min: i64, _max: i64) {
    for i in 0..16 {
        let val: i64 = c[stride * i];
        c[stride * i] = 2 * val + ((val * 1697 + 1024) >> 11);
    }
}

/// dav1d `src/itx_1d.c:1011-1017` (`inv_identity32_1d_c`) — mekanik port.
fn inv_identity32_1d(c: &mut [i64], stride: usize, _min: i64, _max: i64) {
    for i in 0..32 {
        c[stride * i] *= 4;
    }
}

/// dav1d `src/itx_1d.c:1066-1081` (`dav1d_inv_wht4_1d_c`) — mekanik port.
fn dav1d_inv_wht4_1d_internal(c: &mut [i64], stride: usize) {
    let in0: i64 = c[0];
    let in1: i64 = c[stride];
    let in2: i64 = c[2 * stride];
    let in3: i64 = c[3 * stride];

    let t0: i64 = in0 + in1;
    let t2: i64 = in2 - in3;
    let t4: i64 = (t0 - t2) >> 1;
    let t3: i64 = t4 - in3;
    let t1: i64 = t4 - in1;

    c[0] = t0 - t3;
    c[stride] = t3;
    c[2 * stride] = t1;
    c[3 * stride] = t2 + t1;
}

// ---------------------------------------------------------------------------
// dav1d `src/itx_1d.c:92-96,145-149,240-244,430-434` — ince sarmalayıcılar
// (C'de `inv_dctN_1d_c`, burada aynı adla; `tx64 = false`).
// ---------------------------------------------------------------------------

/// dav1d `src/itx_1d.c:92-96` (`inv_dct4_1d_c`).
pub fn inv_dct4_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_dct4_1d_internal(c, stride, min, max, false);
}

/// dav1d `src/itx_1d.c:145-149` (`inv_dct8_1d_c`).
pub fn inv_dct8_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_dct8_1d_internal(c, stride, min, max, false);
}

/// dav1d `src/itx_1d.c:240-244` (`inv_dct16_1d_c`).
pub fn inv_dct16_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_dct16_1d_internal(c, stride, min, max, false);
}

/// dav1d `src/itx_1d.c:430-434` (`inv_dct32_1d_c`).
pub fn inv_dct32_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_dct32_1d_internal(c, stride, min, max, false);
}

/// dav1d `src/itx_1d.c:436` (`inv_dct64_1d_c`).
pub fn inv_dct64_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_dct64_1d_internal(c, stride, min, max);
}

/// dav1d `src/itx_1d.c:965-968` (`inv_adst4_1d_c`).
pub fn inv_adst4_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_adst4_1d_internal(c, stride, min, max, false);
}

/// dav1d `src/itx_1d.c:969-973` (`inv_flipadst4_1d_c`).
pub fn inv_flipadst4_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_adst4_1d_internal(c, stride, min, max, true);
}

/// dav1d `src/itx_1d.c:965-968` (`inv_adst8_1d_c`).
pub fn inv_adst8_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_adst8_1d_internal(c, stride, min, max, false);
}

/// dav1d `src/itx_1d.c:969-973` (`inv_flipadst8_1d_c`).
pub fn inv_flipadst8_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_adst8_1d_internal(c, stride, min, max, true);
}

/// dav1d `src/itx_1d.c:965-968` (`inv_adst16_1d_c`).
pub fn inv_adst16_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_adst16_1d_internal(c, stride, min, max, false);
}

/// dav1d `src/itx_1d.c:969-973` (`inv_flipadst16_1d_c`).
pub fn inv_flipadst16_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_adst16_1d_internal(c, stride, min, max, true);
}

/// dav1d `src/itx_1d.c:983-991` (`inv_identity4_1d_c`).
pub fn inv_identity4_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_identity4_1d(c, stride, min, max);
}

/// dav1d `src/itx_1d.c:993-999` (`inv_identity8_1d_c`).
pub fn inv_identity8_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_identity8_1d(c, stride, min, max);
}

/// dav1d `src/itx_1d.c:1001-1009` (`inv_identity16_1d_c`).
pub fn inv_identity16_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_identity16_1d(c, stride, min, max);
}

/// dav1d `src/itx_1d.c:1011-1017` (`inv_identity32_1d_c`).
pub fn inv_identity32_1d_c(c: &mut [i64], stride: usize, min: i64, max: i64) {
    inv_identity32_1d(c, stride, min, max);
}

/// dav1d `src/itx_1d.c:1066-1081` (`dav1d_inv_wht4_1d_c`).
pub fn dav1d_inv_wht4_1d_c(c: &mut [i64], stride: usize) {
    dav1d_inv_wht4_1d_internal(c, stride);
}

// ---------------------------------------------------------------------------
// dav1d `src/itx_1d.c:1019-1041` (`dav1d_tx1d_fns`) — boyut × tür tablosu.
// `None`, dav1d'de tablo hücresinin boş (NULL) olmasıyla aynıdır.
// ---------------------------------------------------------------------------

/// dav1d `dav1d_tx1d_fns[N_TX_SIZES][N_TX_1D_TYPES]` (`src/itx_1d.c:1019-1041`).
///
/// İlk indeks kare tx boyutu (`TX_4X4..TX_64X64`), ikincisi [`Tx1dType`].
pub const TX1D_FNS: [[Option<Itx1dFn>; 4]; 5] = [
    // TX_4X4
    [
        Some(inv_dct4_1d_c),
        Some(inv_adst4_1d_c),
        Some(inv_identity4_1d_c),
        Some(inv_flipadst4_1d_c),
    ],
    // TX_8X8
    [
        Some(inv_dct8_1d_c),
        Some(inv_adst8_1d_c),
        Some(inv_identity8_1d_c),
        Some(inv_flipadst8_1d_c),
    ],
    // TX_16X16
    [
        Some(inv_dct16_1d_c),
        Some(inv_adst16_1d_c),
        Some(inv_identity16_1d_c),
        Some(inv_flipadst16_1d_c),
    ],
    // TX_32X32 (ADST/flipADST yok — dav1d tablosunda boş)
    [Some(inv_dct32_1d_c), None, Some(inv_identity32_1d_c), None],
    // TX_64X64 (yalnız DCT)
    [Some(inv_dct64_1d_c), None, None, None],
];
