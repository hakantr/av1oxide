//! Girdi paketi ve sahiplik sözleşmesi (BLK-02).
//!
//! ```
//! # use av1oxide::api::Packet;
//! let data = [0x0Au8, 0x0B, 0x0C];
//! let packet = Packet::with_timestamp(&data, 42);
//! assert_eq!(packet.len(), 3);
//! assert_eq!(packet.timestamp(), Some(42));
//! ```
//!
//! ## Sahiplik (BLK-02)
//!
//! [`Packet`] veriyi **ödünç alır** (`&'a [u8]`): çağrı süresince geçerli
//! olmalıdır. Borç verilen veri ile **saklanan/asenkron** işlerin sahipliği
//! karıştırılmaz; çözücü girdiyi çağrıdan sonra saklayacaksa **sahiplenilmiş
//! kopya** (`Vec<u8>` / `Arc<[u8]>`) almak zorundadır. F2'de bu ayrım
//! `send_packet` sözleşmesinde açıkça yazılacaktır.

#![forbid(unsafe_code)]

/// Tek bir girdi paketi: AV1 bit akışı parçası + zaman damgası.
///
/// Paket, **bir veya birden çok OBU** içerebilir; kısmi paket, düzensiz akış ve
/// paket içinde birden çok kare durumları F2'de bu tip üzerinden ele alınır
/// (BLK-01).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packet<'a> {
    data: &'a [u8],
    timestamp: Option<i64>,
}

impl<'a> Packet<'a> {
    /// Zaman damgasız paket.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            timestamp: None,
        }
    }

    /// Zaman damgalı paket (mikrosaniye).
    #[must_use]
    pub const fn with_timestamp(data: &'a [u8], timestamp: i64) -> Self {
        Self {
            data,
            timestamp: Some(timestamp),
        }
    }

    /// Paket verisi (ödünç).
    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Paket uzunluğu (bayt).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Paket boş mu? (Boş paket geçerli bir çağrıdır; çözücü onu yok sayar.)
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Zaman damgası (varsa).
    #[must_use]
    pub const fn timestamp(&self) -> Option<i64> {
        self.timestamp
    }
}

#[cfg(test)]
mod tests {
    use super::Packet;

    #[test]
    fn paket_veriyi_odunc_alir() {
        let data = [1u8, 2, 3, 4];
        let packet = Packet::new(&data);
        assert_eq!(packet.data(), &data);
        assert_eq!(packet.len(), 4);
        assert!(!packet.is_empty());
        assert_eq!(packet.timestamp(), None);
    }

    #[test]
    fn bos_paket_gecerlidir() {
        let packet = Packet::new(&[]);
        assert!(packet.is_empty());
        assert_eq!(packet.len(), 0);
    }

    #[test]
    fn zaman_damgasi_tasinir() {
        let packet = Packet::with_timestamp(&[0u8; 8], -7);
        assert_eq!(packet.timestamp(), Some(-7));

        let copy = packet;
        assert_eq!(
            copy, packet,
            "Packet Copy olmalı (ödünç veri + kopya metaveri)"
        );
    }
}
