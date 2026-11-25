//! F3b (P25) — RSK-07 sentinel testleri: iki AVX2 pilotu (CDEF, MC) belirteçli
//! girişleriyle, bütün genişlik/yükseklik/stride sınır durumlarında.
//!
//! Her durumda:
//!
//! - **Çıktı ve ara tamponların** hemen önüne ve arkasına bilinen desende
//!   koruma baytı konur (`KORUMA` bayt); satırlar arası boşluklar (stride −
//!   genişlik) da korunur. Çağrıdan sonra yazılması gereken bölge dışında tek
//!   bir bayt bile değişmemelidir.
//! - **Kaynak dilimlerin dışından okuma** dolaylı ölçülür: koruma ve boşluk
//!   baytları iki farklı desenle (`0x00`, `0xFF`) doldurulur; çekirdek
//!   yalnız sözleşmedeki baytları okuyorsa iki koşunun çıktısı bit-aynıdır.
//! - Skaler ve AVX2 yolu **aynı girdilerle** koşulur ve eşit olmalıdır.
//!
//! AVX2 yolu yalnız kurulumda üretilen belirteçle açılır
//! (`DspDispatch::select(CpuFlags::detect())`); AVX2'siz makinede yalnız
//! skaler yol sınanır.

mod common;

#[cfg(target_arch = "x86_64")]
use av1oxide::cpu::CpuFlags;
#[cfg(target_arch = "x86_64")]
use av1oxide::dsp::DspDispatch;
use av1oxide::dsp::cdef::{
    CdefBitDepth, CdefBlockSize, CdefDirFn, CdefEdgeFlags, CdefFilterFn, CdefFilterParams,
};
use av1oxide::dsp::mc::{McBitDepth, McFn};
use av1oxide::dsp::scalar::{cdef as scalar_cdef, mc as scalar_mc};
use common::mc_fixture::Rng;
#[cfg(target_arch = "x86_64")]
use common::on_kosul::{self, Neden};

/// Koruma bölgesi uzunluğu (bayt): en geniş erişimden (32 bayt) büyük.
const KORUMA: usize = 48;
/// Koruma deseni (yazma taşması ölçümü).
const DESEN: u8 = 0xA5;

/// Kurulumda üretilen tablo (belirteçli; yalnız x86 MC yollarında kullanılır).
#[cfg(target_arch = "x86_64")]
fn dsp() -> DspDispatch {
    DspDispatch::select(CpuFlags::detect())
}

/// P29: sentinel testleri üç bit derinliğinde koşar (16 bpc AVX2 gövdesi).
const BIT_DERINLIKLERI: [CdefBitDepth; 3] =
    [CdefBitDepth::B8, CdefBitDepth::B10, CdefBitDepth::B12];

/// `CdefBitDepth` → bit sayısı.
fn bd_bit(bd: CdefBitDepth) -> u8 {
    match bd {
        CdefBitDepth::B8 => 8,
        CdefBitDepth::B10 => 10,
        CdefBitDepth::B12 => 12,
    }
}

/// Sınanacak çekirdek yolları: skaler her zaman; P30: bu makinede belirteci
/// üretilen **her** ISA düzeyinin gövdesi (SSSE3, SSE4.1, AVX2, AVX512ICL;
/// adaylardan, varsayılan olup olmadığına bakılmadan).
fn cdef_dirs(bd: CdefBitDepth) -> Vec<(&'static str, CdefDirFn)> {
    let skaler: CdefDirFn = match bd {
        CdefBitDepth::B8 => scalar_cdef::find_dir::<8>,
        CdefBitDepth::B10 => scalar_cdef::find_dir::<10>,
        CdefBitDepth::B12 => scalar_cdef::find_dir::<12>,
    };
    let mut v: Vec<(&'static str, CdefDirFn)> = vec![("skaler", skaler)];
    let (isa, _) = common::cdef_fixture::isa_govdeleri(bd_bit(bd));
    v.extend(isa.iter().filter_map(|g| g.dir.map(|f| (g.ad, f))));
    v
}

fn cdef_filters(bd: CdefBitDepth) -> Vec<(&'static str, CdefFilterFn)> {
    let skaler: CdefFilterFn = match bd {
        CdefBitDepth::B8 => scalar_cdef::filter::<8>,
        CdefBitDepth::B10 => scalar_cdef::filter::<10>,
        CdefBitDepth::B12 => scalar_cdef::filter::<12>,
    };
    let mut v: Vec<(&'static str, CdefFilterFn)> = vec![("skaler", skaler)];
    let (isa, _) = common::cdef_fixture::isa_govdeleri(bd_bit(bd));
    v.extend(isa.iter().filter_map(|g| g.fb.map(|f| (g.ad, f))));
    v
}

/// `n` bayt; 16 bpc'de piksel değerleri bit derinliği aralığında
/// (sözleşme: piksel ≤ `bitdepth_max`).
fn rastgele_piksel(rng: &mut Rng, n: usize, bd: CdefBitDepth) -> Vec<u8> {
    if bd == CdefBitDepth::B8 {
        return rastgele(rng, n);
    }
    let maske = bd.max_pixel();
    (0..n / 2)
        .flat_map(|_| ((rng.u64() as u16) & maske).to_ne_bytes())
        .collect()
}

fn mc_kernels() -> Vec<(&'static str, McFn)> {
    let skaler: (&'static str, McFn) = ("skaler", scalar_mc::put_8tap_regular);
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = dsp().avx2_token() {
        return vec![
            skaler,
            ("avx2", av1oxide::dsp::x86::mc::put_8tap_regular_fn(token)),
        ];
    }
    vec![skaler]
}

/// Korumalı tampon: `[KORUMA koruma][len veri][KORUMA koruma]`.
struct Korumali {
    raw: Vec<u8>,
    len: usize,
}

impl Korumali {
    /// Veri `data`, koruma ve `gap` maskesindeki baytlar `fill` ile.
    fn new(data: &[u8], fill: u8, gap: impl Fn(usize) -> bool) -> Self {
        let len = data.len();
        let mut raw = vec![fill; KORUMA + len + KORUMA];
        for (i, &b) in data.iter().enumerate() {
            raw[KORUMA + i] = if gap(i) { fill } else { b };
        }
        Self { raw, len }
    }

    fn data(&self) -> &[u8] {
        &self.raw[KORUMA..KORUMA + self.len]
    }

    fn data_mut(&mut self) -> &mut [u8] {
        &mut self.raw[KORUMA..KORUMA + self.len]
    }

    /// Ön/arka koruma bölgeleri değişmedi mi?
    fn guards_ok(&self, fill: u8) -> bool {
        self.raw[..KORUMA].iter().all(|&b| b == fill)
            && self.raw[KORUMA + self.len..].iter().all(|&b| b == fill)
    }
}

/// `rows × stride` düzeninde yalnız `(y < h, x < w)` hücrelerinin yazılabildiği
/// bir hedefte, yazılmaması gereken baytları (satır boşlukları + korumalar)
/// karşılaştırır. Değişen ilk bayt indeksini döndürür.
fn yazma_disi_degisti(
    before: &[u8],
    after: &[u8],
    stride: usize,
    w: usize,
    h: usize,
) -> Option<usize> {
    (0..before.len()).find(|&i| {
        let (y, x) = (i / stride, i % stride);
        let yazilabilir = y < h && x < w;
        !yazilabilir && before[i] != after[i]
    })
}

fn rastgele(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n).map(|_| rng.u64() as u8).collect()
}

/// CDEF `dir`: `img.len() = 7*stride + 8` (sözleşmenin en kısa hâli); stride
/// sınırları; satır boşlukları ve korumalar iki desenle — çıktı desenden
/// bağımsız ve iki yol eşit.
#[test]
fn f3b_cdef_dir_sinir_durumlari() {
    let mut rng = Rng::new(0xF3B0_0001);
    let mut n = 0usize;
    for bd in BIT_DERINLIKLERI {
        let bpp = bd.bytes();
        // 16 bpc: bayt adımı çift (piksel hizası), en kısa satır 16 bayt.
        let adimlar: &[usize] = if bpp == 1 {
            &[8, 9, 15, 16, 33, 64]
        } else {
            &[16, 18, 30, 32, 66, 128]
        };
        for &stride in adimlar {
            for _ in 0..64 {
                let len = 7 * stride + 8 * bpp;
                let data = rastgele_piksel(&mut rng, len, bd);
                let gap = |i: usize| i % stride >= 8 * bpp;
                let mut sonuc = Vec::new();
                for fill in [0x00u8, 0xFF] {
                    let img = Korumali::new(&data, fill, gap);
                    for (ad, k) in cdef_dirs(bd) {
                        let r = k(img.data(), stride, bd);
                        assert!(
                            img.guards_ok(fill),
                            "dir {ad} {bd:?} stride={stride}: koruma değişti"
                        );
                        sonuc.push((ad, r.dir, r.var));
                    }
                }
                let ilk = (sonuc[0].1, sonuc[0].2);
                for (ad, d, v) in &sonuc {
                    assert_eq!(
                        (*d, *v),
                        ilk,
                        "dir {ad} {bd:?} stride={stride}: yol/desen farkı"
                    );
                }
                n += 1;
            }
        }
    }
    println!(
        "[F3b] CDEF dir: {n} durum (8/10/12 bit) × {} yol × 2 desen",
        cdef_dirs(CdefBitDepth::B8).len()
    );
}

/// CDEF `filter`: üç blok boyutu × 16 kenar birleşimi × stride sınırları ×
/// güç türleri × yön × sönüm sınırları. `dst` (okunur + yazılır), `left`,
/// `top`, `bottom` sözleşme uzunluğunda ve korumalı.
#[test]
fn f3b_cdef_filter_sinir_durumlari() {
    let mut rng = Rng::new(0xF3B0_0002);
    let mut n = 0usize;
    let sizes = [
        CdefBlockSize::W8H8,
        CdefBlockSize::W4H8,
        CdefBlockSize::W4H4,
    ];
    for bd in BIT_DERINLIKLERI {
        let (bpp, s) = (bd.bytes(), bd.min_8());
        for size in sizes {
            let w = size.w();
            for edges in 0u8..16 {
                // Bayt adımı: sözleşmenin en kısası `(w+4)·bpp`, bir piksel fazlası, 32 piksel.
                for stride in [(w + 4) * bpp, (w + 5) * bpp, 32 * bpp] {
                    for (pri, sec) in [(9u8, 0u8), (0, 2), (15, 4)] {
                        for dir in [0u8, 3, 7] {
                            // Sönüm uçları: 16 bpc'de `cdef_damping + 3 + bitdepth_min_8`.
                            for damping in [3u8, 6] {
                                let params = CdefFilterParams {
                                    bit_depth: bd,
                                    size,
                                    pri_strength: pri << s,
                                    sec_strength: sec << s,
                                    dir,
                                    damping: damping + s as u8,
                                    edges: CdefEdgeFlags::from_bits(edges).expect("kenar"),
                                };
                                filter_durumu(&mut rng, params, stride);
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "[F3b] CDEF filter: {n} durum (8/10/12 bit) × {} yol × 2 desen",
        cdef_filters(CdefBitDepth::B8).len()
    );
}

fn filter_durumu(rng: &mut Rng, params: CdefFilterParams, stride: usize) {
    let bd = params.bit_depth;
    let bpp = bd.bytes();
    let (w, h) = (params.size.w(), params.size.h());
    let dst_len = (h - 1) * stride + (w + 2) * bpp;
    let lr_len = h * 2 * bpp;
    let tb_len = stride + (w + 4) * bpp;
    let dst0 = rastgele_piksel(rng, dst_len, bd);
    let left = rastgele_piksel(rng, lr_len, bd);
    let top = rastgele_piksel(rng, tb_len, bd);
    let bottom = rastgele_piksel(rng, tb_len, bd);
    // `dst` satır boşlukları (x >= w+2) sözleşme dışıdır.
    let dst_gap = |i: usize| i % stride >= (w + 2) * bpp;
    let tb_gap = |i: usize| i % stride >= (w + 4) * bpp && i < stride;
    let mut ref_out: Option<Vec<u8>> = None;
    for fill in [0x00u8, 0xFF] {
        for (ad, k) in cdef_filters(bd) {
            let mut dst = Korumali::new(&dst0, fill, dst_gap);
            let l = Korumali::new(&left, fill, |_| false);
            let t = Korumali::new(&top, fill, tb_gap);
            let b = Korumali::new(&bottom, fill, tb_gap);
            let before = dst.raw.clone();
            k(dst.data_mut(), stride, l.data(), t.data(), b.data(), params);
            let ctx = format!("filter {ad} {w}x{h} stride={stride} {params:?} desen={fill:#04x}");
            assert!(dst.guards_ok(fill), "{ctx}: dst koruması değişti");
            let degisen = yazma_disi_degisti(
                &before[KORUMA..KORUMA + dst_len],
                dst.data(),
                stride,
                w * bpp,
                h,
            );
            assert_eq!(degisen, None, "{ctx}: yazma bölgesi dışında bayt değişti");
            assert!(
                l.guards_ok(fill) && t.guards_ok(fill) && b.guards_ok(fill),
                "{ctx}"
            );
            // Yazılan bölge (h × w) yol ve desenden bağımsız olmalı.
            let out: Vec<u8> = (0..h)
                .flat_map(|y| dst.data()[y * stride..y * stride + w * bpp].to_vec())
                .collect();
            match &ref_out {
                None => ref_out = Some(out),
                Some(r) => assert_eq!(&out, r, "{ctx}: yol/desen farkı (okuma taşması?)"),
            }
        }
    }
    // F4-CDEF a: çözme hattı eksik kenarın dilimini **boş** geçirir
    // (`decode/cdef.rs`); AVX2 gövdesi bu dilimi okumamalı ve aynı sonucu
    // vermelidir (yeni okuma yolu).
    let e = params.edges;
    let bos: &[u8] = &[];
    for (ad, k) in cdef_filters(bd) {
        let mut dst = Korumali::new(&dst0, 0xA5, dst_gap);
        let l = if e.contains(CdefEdgeFlags::HAVE_LEFT) {
            &left[..]
        } else {
            bos
        };
        let t = if e.contains(CdefEdgeFlags::HAVE_TOP) {
            &top[..]
        } else {
            bos
        };
        let b = if e.contains(CdefEdgeFlags::HAVE_BOTTOM) {
            &bottom[..]
        } else {
            bos
        };
        k(dst.data_mut(), stride, l, t, b, params);
        let out: Vec<u8> = (0..h)
            .flat_map(|y| dst.data()[y * stride..y * stride + w * bpp].to_vec())
            .collect();
        assert!(dst.guards_ok(0xA5), "filter {ad} boş kenar dilimi: koruma");
        assert_eq!(
            Some(&out),
            ref_out.as_ref(),
            "filter {ad} {w}x{h} stride={stride} {params:?}: boş kenar dilimi farkı"
        );
    }
}

/// MC `put_8tap_regular` (dispatch yuvası): genişlik/yükseklik sınırları ×
/// dört yol (kopya, yatay, dikey, 2B) × kaynak/hedef stride sınırları.
/// Kaynak sözleşme uzunluğunda (`(h+6)*src_stride + w+7 + MC_KUYRUK`, P32
/// kopyasız sözleşmesi), satır boşlukları ve kuyruk iki desenle; hedef
/// korumalı, satır boşlukları denetlenir. Standart olmayan genişlik ve
/// yükseklikler (1, 3, 7, 17, …) AVX2 girişinin skaler devrini sınar.
#[test]
fn f3b_mc_sinir_durumlari() {
    let mut rng = Rng::new(0xF3B0_0003);
    let mut n = 0usize;
    let widths = [1usize, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128];
    for &w in &widths {
        let heights: &[usize] = if w >= 64 {
            &[1, 2, 7, 64, 128]
        } else {
            &[1, 2, 4, 7, 8, 16, 33]
        };
        for &h in heights {
            for (mx, my) in [(0u8, 0u8), (5, 0), (0, 11), (7, 9)] {
                for (ss_pad, ds_pad) in [(0usize, 0usize), (13, 9)] {
                    mc_durumu(&mut rng, w, h, mx, my, w + 7 + ss_pad, w + ds_pad);
                    n += 1;
                }
            }
        }
    }
    println!("[F3b] MC: {n} durum × {} yol × 2 desen", mc_kernels().len());
}

fn mc_durumu(rng: &mut Rng, w: usize, h: usize, mx: u8, my: u8, ss: usize, ds: usize) {
    let bolge = (h + 6) * ss + w + 7;
    let src_len = bolge + av1oxide::dsp::mc::MC_KUYRUK;
    let dst_len = (h - 1) * ds + w;
    let src0 = rastgele(rng, src_len);
    let dst0 = rastgele(rng, dst_len);
    let src_gap = |i: usize| i % ss >= w + 7 || i >= bolge;
    let mut ref_out: Option<Vec<u8>> = None;
    for fill in [0x00u8, 0xFF] {
        for (ad, k) in mc_kernels() {
            let src = Korumali::new(&src0, fill, src_gap);
            let mut dst = Korumali::new(&dst0, DESEN, |_| false);
            let before = dst.data().to_vec();
            k(
                dst.data_mut(),
                ds,
                src.data(),
                ss,
                w,
                h,
                mx,
                my,
                McBitDepth::B8,
            );
            let ctx = format!("mc {ad} {w}x{h} mx={mx} my={my} ss={ss} ds={ds} desen={fill:#04x}");
            assert!(dst.guards_ok(DESEN), "{ctx}: hedef koruması değişti");
            assert!(src.guards_ok(fill), "{ctx}: kaynak koruması değişti");
            assert_eq!(
                yazma_disi_degisti(&before, dst.data(), ds, w, h),
                None,
                "{ctx}: yazma bölgesi dışında bayt değişti"
            );
            let out: Vec<u8> = (0..h)
                .flat_map(|y| dst.data()[y * ds..y * ds + w].to_vec())
                .collect();
            match &ref_out {
                None => ref_out = Some(out),
                Some(r) => assert_eq!(&out, r, "{ctx}: yol/desen farkı (okuma taşması?)"),
            }
        }
    }
}

/// MC çekirdeği geniş satır adımlı bir tamponda (satır adımı `align32(w+39)`,
/// veri `w+7`, satır kuyruğu): belirteçli `put_8tap_regular_prepared`
/// girişiyle (P32'de kopyasız üretim gövdesinin kendisi), tampon dilimi
/// korumalı bir tahsisin içinde. Satır kuyrukları ve korumalar iki desenle —
/// çıktı desenden bağımsız (çöp lane'ler atılır) ve skaler yolla eşit.
/// `dsp::x86` girişi olduğu için yalnız x86_64'te derlenir (P29).
#[cfg(target_arch = "x86_64")]
#[test]
fn f3b_mc_hazir_ara_tampon_korumasi() {
    let Some(token) = dsp().avx2_token() else {
        on_kosul::eksik(
            "f3b_sentinel::f3b_mc_hazir_ara_tampon_korumasi",
            Neden::avx2(),
            "[F3b] MC ara tampon testi AVX2 belirteci ister",
        );
        return;
    };
    let mut rng = Rng::new(0xF3B0_0004);
    let mut n = 0usize;
    for w in [2usize, 4, 8, 16, 32, 64, 128] {
        for h in [2usize, 4, 16, 64] {
            for (mx, my) in [(0u8, 0u8), (5, 0), (0, 11), (7, 9)] {
                let lstride = (w + 39).next_multiple_of(32);
                let rows = h + 7;
                let len = rows * lstride;
                let data = rastgele(&mut rng, len);
                let tail = |i: usize| i % lstride >= w + 7;
                let ds = w + 3;
                let dst_len = (h - 1) * ds + w;
                let mut ref_out: Option<Vec<u8>> = None;
                for fill in [0x00u8, 0xFF] {
                    let padded = Korumali::new(&data, fill, tail);
                    let mut dst = Korumali::new(&vec![0x5au8; dst_len], DESEN, |_| false);
                    let before = dst.data().to_vec();
                    av1oxide::dsp::x86::mc::put_8tap_regular_prepared(
                        token,
                        dst.data_mut(),
                        ds,
                        padded.data(),
                        lstride,
                        w,
                        h,
                        mx,
                        my,
                    );
                    let ctx = format!("hazır {w}x{h} mx={mx} my={my} desen={fill:#04x}");
                    assert!(
                        dst.guards_ok(DESEN) && padded.guards_ok(fill),
                        "{ctx}: koruma"
                    );
                    assert_eq!(
                        yazma_disi_degisti(&before, dst.data(), ds, w, h),
                        None,
                        "{ctx}: yazma bölgesi dışında bayt değişti"
                    );
                    let out: Vec<u8> = (0..h)
                        .flat_map(|y| dst.data()[y * ds..y * ds + w].to_vec())
                        .collect();
                    match &ref_out {
                        None => ref_out = Some(out),
                        Some(r) => assert_eq!(&out, r, "{ctx}: kuyruk deseni çıktıya sızdı"),
                    }
                }
                // Skaler yol aynı dolgu düzenini (satır −3, sütun −3) okur.
                let padded = Korumali::new(&data, 0, tail);
                let mut skaler = vec![0x5au8; dst_len];
                scalar_mc::put_8tap_regular(
                    &mut skaler,
                    ds,
                    padded.data(),
                    lstride,
                    w,
                    h,
                    mx,
                    my,
                    McBitDepth::B8,
                );
                let skaler_out: Vec<u8> = (0..h)
                    .flat_map(|y| skaler[y * ds..y * ds + w].to_vec())
                    .collect();
                assert_eq!(
                    ref_out.as_ref(),
                    Some(&skaler_out),
                    "hazır {w}x{h}: skaler farkı"
                );
                n += 1;
            }
        }
    }
    println!("[F3b] MC hazır ara tampon: {n} durum × 2 desen");
}

/// Koruma ölçümünün kendisi çalışıyor: bilinçli taşma yakalanır (negatif
/// kontrol).
#[test]
fn f3b_sentinel_negatif_kontrol() {
    let mut buf = Korumali::new(&[1, 2, 3, 4], DESEN, |_| false);
    buf.raw[KORUMA + 4] = 0; // arka korumanın ilk baytı
    assert!(!buf.guards_ok(DESEN));
    let before = [0u8; 12];
    let mut after = before;
    after[5] = 1; // satır 1, x = 1 → stride 4, w = 1: yazma dışı
    assert_eq!(yazma_disi_degisti(&before, &after, 4, 1, 3), Some(5));
}
