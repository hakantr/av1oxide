//! P30 (bulgu d) — atlama kaydına eşzamanlı yazma: satırlar birleşmez.
//!
//! Kapılar testleri paralel koşturur; aynı test ikilisindeki iş parçacıkları
//! `AV1OXIDE_ATLAMA_KAYDI` dosyasına aynı anda `ATLANDI <test> <neden>`
//! satırı ekleyebilir. [`common::on_kosul::kayda_ekle`] satırı ve satır sonunu
//! tek `write` çağrısıyla `O_APPEND` dosyaya yazar; bu test çok iş
//! parçacığından yüzlerce satır yazdırır ve her satırın biçime uyduğunu, sayının
//! tam olduğunu denetler. Satırı ikiye bölen bir yazma (ör. satır ve `\n` ayrı)
//! bu testi düşürür (mutasyon: `scripts/corpus/run_f4c_cdef_mutation.py`).
//!
//! P30 düzeltme 1 (bulgu c): zorunlu kipte listede olmayan eksik ön koşul
//! paniklemeden önce kayda `BEKLENMEYEN <test> <neden>` yazar; kapı özeti onu
//! beklenmeyen sayar. Sınama ortam değişkenine dokunmaz, ayarı açıkça kurar
//! (mutasyon M2: `scripts/corpus/run_p30d1_mutation.py`).

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use common::on_kosul::{Ayar, Neden, eksik_ayarla, kayda_ekle};
use std::collections::BTreeMap;

const IS_PARCACIGI: usize = 32;
const SATIR: usize = 400;

#[test]
fn p30_eszamanli_atlama_satirlari_birlesmez() {
    let dizin = std::env::temp_dir().join(format!("av1oxide-on-kosul-{}", std::process::id()));
    std::fs::create_dir_all(&dizin).expect("geçici dizin");
    let yol = dizin.join("atlama.kayit");
    let _ = std::fs::remove_file(&yol);
    let bariyer = std::sync::Barrier::new(IS_PARCACIGI);
    std::thread::scope(|s| {
        for t in 0..IS_PARCACIGI {
            let (yol, bariyer) = (&yol, &bariyer);
            s.spawn(move || {
                bariyer.wait();
                for i in 0..SATIR {
                    let neden = if i % 2 == 0 {
                        "x86-degil"
                    } else {
                        "oracle-verisi-yok"
                    };
                    kayda_ekle(yol, &format!("ATLANDI ikili_{t}::test_{i} {neden}"));
                }
            });
        }
    });
    let metin = std::fs::read_to_string(&yol).expect("kayıt okunur");
    std::fs::remove_dir_all(&dizin).expect("geçici dizin silinir");
    let mut bozuk = Vec::new();
    let mut sayi: BTreeMap<usize, usize> = BTreeMap::new();
    for satir in metin.lines() {
        let alan: Vec<&str> = satir.split(' ').collect();
        let ok = alan.len() == 3
            && alan[0] == "ATLANDI"
            && matches!(alan[2], "x86-degil" | "oracle-verisi-yok")
            && alan[1]
                .strip_prefix("ikili_")
                .and_then(|r| r.split_once("::test_"))
                .is_some_and(|(t, i)| t.parse::<usize>().is_ok() && i.parse::<usize>().is_ok());
        if ok {
            let t: usize = alan[1][6..]
                .split("::")
                .next()
                .unwrap_or("")
                .parse()
                .unwrap_or(0);
            *sayi.entry(t).or_default() += 1;
        } else {
            bozuk.push(satir.to_string());
        }
    }
    assert!(
        bozuk.is_empty(),
        "{} bozuk satır (birleşmiş yazma), ilk: {:?}",
        bozuk.len(),
        bozuk.first()
    );
    assert_eq!(metin.lines().count(), IS_PARCACIGI * SATIR, "satır sayısı");
    assert!(metin.ends_with('\n'), "son satır sonu eksik");
    assert_eq!(sayi.len(), IS_PARCACIGI);
    assert!(
        sayi.values().all(|&n| n == SATIR),
        "iş parçacığı başına sayı: {sayi:?}"
    );
}

#[test]
fn p30d1_zorunlu_kipte_beklenmeyen_once_kayda_yazilir() {
    let dizin = std::env::temp_dir().join(format!("av1oxide-on-kosul-z-{}", std::process::id()));
    std::fs::create_dir_all(&dizin).expect("geçici dizin");
    let liste = dizin.join("liste.txt");
    std::fs::write(&liste, "t::beklenen isa-yok  # yorum\n").expect("liste");
    let kayit = dizin.join("atlama.kayit");
    let _ = std::fs::remove_file(&kayit);
    let ayar = Ayar {
        zorunlu: true,
        beklenen: Some(liste),
        kayit: Some(kayit.clone()),
    };
    // Listede olan çift: panik yok, ATLANDI.
    eksik_ayarla(&ayar, "t::beklenen", Neden::IsaYok, "sınama");
    // Listede olmayan çift: önce kayıt, sonra panik.
    let sonuc = std::panic::catch_unwind(|| {
        eksik_ayarla(&ayar, "t::yok", Neden::IsaYok, "sınama");
    });
    assert!(
        sonuc.is_err(),
        "zorunlu kipte beklenmeyen eksiklik paniklemeli"
    );
    // Zorunlu değilken aynı çift yalnız ATLANDI yazar.
    let serbest = Ayar {
        zorunlu: false,
        ..ayar.clone()
    };
    eksik_ayarla(&serbest, "t::yok", Neden::IsaYok, "sınama");
    let metin = std::fs::read_to_string(&kayit).expect("kayıt okunamadı");
    let _ = std::fs::remove_dir_all(&dizin);
    assert_eq!(
        metin, "ATLANDI t::beklenen isa-yok\nBEKLENMEYEN t::yok isa-yok\nATLANDI t::yok isa-yok\n",
        "zorunlu kipte beklenmeyen satırı panikten önce kayda düşmeli"
    );
}
