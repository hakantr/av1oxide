//! Intra kenar kullanılabilirliği ağacı (S2b.3).
//!
//! dav1d `src/intra_edge.c` + `src/intra_edge.h` karşılığı. Ağaç, bir
//! süper blok içindeki partition sırasına göre **sağ-üst** ve **sol-alt**
//! kenar piksellerinin kullanılabilirliğini taşır (`EDGE_*` bayrakları).
//! `ipred_prepare` bu bayrakları `Z1`/`Z3`/`FILTER_PRED` kenarlarında
//! kullanır; yanlış bayrak piksel farkı üretir.
//!
//! ## Kaynak (dav1d @ `99f3354`, salt-okunur)
//!
//! | Rust | dav1d |
//! |---|---|
//! | [`EdgeTree::build`] | `src/intra_edge.c:70-139` (`init_edges`, `init_mode_node`, `init_intra_edge_tree`) |
//! | `EDGE_*` sabitleri | `src/intra_edge.h:33-50` |
//! | Çocuk seçimi | `src/decode.c:2123-2395` (`decode_sb`) |
//!
//! AV1 §7.11.2 "Intra prediction process" kenar kullanılabilirliği
//! kuralları. dav1d işaretçi aritmetiği yerine indeksli düğüm kullanılır;
//! ağaç yapısı ve bayrak değerleri birebir aynıdır.

#![forbid(unsafe_code)]

/// Blok seviyesi: 8×8 (`BL_8X8`).
pub(crate) const BL_8X8: u8 = 4;

/// `EDGE_I444_TOP_HAS_RIGHT`.
pub(crate) const EDGE_I444_TOP_HAS_RIGHT: u8 = 1 << 0;
/// `EDGE_I422_TOP_HAS_RIGHT`.
pub(crate) const EDGE_I422_TOP_HAS_RIGHT: u8 = 1 << 1;
/// `EDGE_I420_TOP_HAS_RIGHT`.
pub(crate) const EDGE_I420_TOP_HAS_RIGHT: u8 = 1 << 2;
/// `EDGE_I444_LEFT_HAS_BOTTOM`.
pub(crate) const EDGE_I444_LEFT_HAS_BOTTOM: u8 = 1 << 3;
/// `EDGE_I422_LEFT_HAS_BOTTOM`.
pub(crate) const EDGE_I422_LEFT_HAS_BOTTOM: u8 = 1 << 4;
/// `EDGE_I420_LEFT_HAS_BOTTOM`.
pub(crate) const EDGE_I420_LEFT_HAS_BOTTOM: u8 = 1 << 5;
/// `EDGE_ALL_TOP_HAS_RIGHT`.
pub(crate) const EDGE_ALL_TOP_HAS_RIGHT: u8 =
    EDGE_I444_TOP_HAS_RIGHT | EDGE_I422_TOP_HAS_RIGHT | EDGE_I420_TOP_HAS_RIGHT;
/// `EDGE_ALL_LEFT_HAS_BOTTOM`.
pub(crate) const EDGE_ALL_LEFT_HAS_BOTTOM: u8 =
    EDGE_I444_LEFT_HAS_BOTTOM | EDGE_I422_LEFT_HAS_BOTTOM | EDGE_I420_LEFT_HAS_BOTTOM;
/// `EDGE_ALL_TR_AND_BL`.
pub(crate) const EDGE_ALL_TR_AND_BL: u8 = EDGE_ALL_TOP_HAS_RIGHT | EDGE_ALL_LEFT_HAS_BOTTOM;

/// Ağaç düğümü indeksi (`EdgeNode *`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeId {
    /// `true` ise [`EdgeTip`], değilse [`EdgeBranch`].
    pub tip: bool,
    /// İlgili vektördeki indeks.
    pub idx: u16,
}

#[derive(Clone, Copy, Debug, Default)]
struct EdgeBranch {
    o: u8,
    h: [u8; 2],
    v: [u8; 2],
    h4: u8,
    v4: u8,
    split: [NodeId; 4],
}

#[derive(Clone, Copy, Debug, Default)]
struct EdgeTip {
    o: u8,
    h: [u8; 2],
    v: [u8; 2],
    split: [u8; 3],
}

const NO_NODE: NodeId = NodeId { tip: false, idx: 0 };

/// Bir süper blok boyutu için kenar ağacı (`branch_sb128` / `branch_sb64`).
#[derive(Clone, Debug)]
pub(crate) struct EdgeTree {
    branches: Vec<EdgeBranch>,
    tips: Vec<EdgeTip>,
    root: NodeId,
}

impl EdgeTree {
    /// Ağacı kurar: `root_bl` = 0 (`BL_128X128`) veya 1 (`BL_64X64`).
    pub(crate) fn build(root_bl: u8) -> Self {
        let mut tree = Self {
            branches: Vec::with_capacity(if root_bl == 0 { 85 } else { 21 }),
            tips: Vec::with_capacity(if root_bl == 0 { 256 } else { 64 }),
            root: NO_NODE,
        };
        tree.root = tree.build_node(root_bl, true, false);
        tree
    }

    /// `init_mode_node`: `top_has_right`/`left_has_bottom` bayraklarıyla
    /// düğümü ve çocuklarını kurar.
    fn build_node(&mut self, bl: u8, top_has_right: bool, left_has_bottom: bool) -> NodeId {
        let flags = (if top_has_right {
            EDGE_ALL_TOP_HAS_RIGHT
        } else {
            0
        }) | (if left_has_bottom {
            EDGE_ALL_LEFT_HAS_BOTTOM
        } else {
            0
        });
        if bl == BL_8X8 {
            let idx = self.tips.len() as u16;
            let mut tip = EdgeTip {
                o: flags,
                h: [0; 2],
                v: [0; 2],
                split: [0; 3],
            };
            tip.h[0] = flags | EDGE_ALL_LEFT_HAS_BOTTOM;
            tip.v[0] = flags | EDGE_ALL_TOP_HAS_RIGHT;
            tip.h[1] = flags & (EDGE_ALL_LEFT_HAS_BOTTOM | EDGE_I420_TOP_HAS_RIGHT);
            tip.v[1] = flags
                & (EDGE_ALL_TOP_HAS_RIGHT | EDGE_I420_LEFT_HAS_BOTTOM | EDGE_I422_LEFT_HAS_BOTTOM);
            tip.split[0] = (flags & EDGE_ALL_TOP_HAS_RIGHT) | EDGE_I422_LEFT_HAS_BOTTOM;
            tip.split[1] = flags | EDGE_I444_TOP_HAS_RIGHT;
            tip.split[2] = flags
                & (EDGE_I420_TOP_HAS_RIGHT | EDGE_I420_LEFT_HAS_BOTTOM | EDGE_I422_LEFT_HAS_BOTTOM);
            self.tips.push(tip);
            return NodeId { tip: true, idx };
        }
        let idx = self.branches.len() as u16;
        self.branches.push(EdgeBranch {
            o: flags,
            h: [
                flags | EDGE_ALL_LEFT_HAS_BOTTOM,
                flags & EDGE_ALL_LEFT_HAS_BOTTOM,
            ],
            v: [
                flags | EDGE_ALL_TOP_HAS_RIGHT,
                flags & EDGE_ALL_TOP_HAS_RIGHT,
            ],
            h4: EDGE_ALL_LEFT_HAS_BOTTOM,
            v4: EDGE_ALL_TOP_HAS_RIGHT,
            split: [NO_NODE; 4],
        });
        // `BL_16X16` = 3 (dav1d `intra_edge.c:100-105`: `if (bl == BL_16X16)`).
        // T30'a kadar burada yanlışlıkla `bl == 2` (BL_32X32) yazıyordu:
        // 32x32 H4/V4 düğümleri kroma TR/BL bayraklarını **fazladan**, 16x16
        // düğümleri ise **eksik** taşıyordu. Kroma 4:2:0/4:2:2'de bu, alt-sol
        // (ve üst-sağ) komşunun yanlış "kullanılabilir" sayılmasına ve
        // henüz çözülmemiş (sıfır) piksellerin tahmine girmesine yol açıyordu.
        if bl == 3 {
            // BL_16X16: `h4`/`v4` ek bayrakları.
            self.branches[idx as usize].h4 |= flags & EDGE_I420_TOP_HAS_RIGHT;
            self.branches[idx as usize].v4 |=
                flags & (EDGE_I420_LEFT_HAS_BOTTOM | EDGE_I422_LEFT_HAS_BOTTOM);
        }
        let mut splits = [NO_NODE; 4];
        for (n, slot) in splits.iter_mut().enumerate() {
            let child_top = !(n == 3 || (n == 1 && !top_has_right));
            let child_left = n == 0 || (n == 2 && left_has_bottom);
            *slot = self.build_node(bl + 1, child_top, child_left);
        }
        self.branches[idx as usize].split = splits;
        NodeId { tip: false, idx }
    }

    pub(crate) fn root(&self) -> NodeId {
        self.root
    }

    fn branch(&self, id: NodeId) -> &EdgeBranch {
        const EMPTY: EdgeBranch = EdgeBranch {
            o: 0,
            h: [0; 2],
            v: [0; 2],
            h4: 0,
            v4: 0,
            split: [NO_NODE; 4],
        };
        self.branches.get(id.idx as usize).unwrap_or(&EMPTY)
    }

    fn tip(&self, id: NodeId) -> &EdgeTip {
        const EMPTY: EdgeTip = EdgeTip {
            o: 0,
            h: [0; 2],
            v: [0; 2],
            split: [0; 3],
        };
        self.tips.get(id.idx as usize).unwrap_or(&EMPTY)
    }

    /// `node->o`.
    pub(crate) fn o(&self, id: NodeId) -> u8 {
        if id.tip {
            self.tip(id).o
        } else {
            self.branch(id).o
        }
    }

    /// `node->h[i]`.
    pub(crate) fn h(&self, id: NodeId, i: usize) -> u8 {
        if id.tip {
            self.tip(id).h[i]
        } else {
            self.branch(id).h[i]
        }
    }

    /// `node->v[i]`.
    pub(crate) fn v(&self, id: NodeId, i: usize) -> u8 {
        if id.tip {
            self.tip(id).v[i]
        } else {
            self.branch(id).v[i]
        }
    }

    /// `EdgeBranch::h4` (yalnız dal düğümü).
    pub(crate) fn h4(&self, id: NodeId) -> u8 {
        self.branch(id).h4
    }

    /// `EdgeBranch::v4` (yalnız dal düğümü).
    pub(crate) fn v4(&self, id: NodeId) -> u8 {
        self.branch(id).v4
    }

    /// `INTRA_EDGE_SPLIT(node, i)` — dal düğümünün çocuğu.
    pub(crate) fn split_node(&self, id: NodeId, i: usize) -> NodeId {
        self.branch(id).split[i]
    }

    /// `EdgeTip::split[i]` — 8×8 yaprak düğümünün bayrağı.
    pub(crate) fn split_tip(&self, id: NodeId, i: usize) -> u8 {
        self.tip(id).split[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kok_bayraklari_dav1d_ile_ayni() {
        for (root_bl, expected_o) in [(0u8, EDGE_ALL_TOP_HAS_RIGHT), (1, EDGE_ALL_TOP_HAS_RIGHT)] {
            let tree = EdgeTree::build(root_bl);
            let root = tree.root();
            assert_eq!(tree.o(root), expected_o);
            assert_eq!(tree.h(root, 0), expected_o | EDGE_ALL_LEFT_HAS_BOTTOM);
            assert_eq!(tree.v(root, 0), expected_o | EDGE_ALL_TOP_HAS_RIGHT);
            assert_eq!(tree.h(root, 1), expected_o & EDGE_ALL_LEFT_HAS_BOTTOM);
            assert_eq!(tree.v(root, 1), expected_o & EDGE_ALL_TOP_HAS_RIGHT);
        }
    }

    #[test]
    fn dugum_sayilari_dav1d_dizileriyle_ayni() {
        let t = EdgeTree::build(0);
        assert_eq!(t.branches.len(), 1 + 4 + 16 + 64);
        assert_eq!(t.tips.len(), 256);
        let t = EdgeTree::build(1);
        assert_eq!(t.branches.len(), 1 + 4 + 16);
        assert_eq!(t.tips.len(), 64);
    }

    #[test]
    fn yaprak_bayraklari_kaynak_ile_ayni() {
        // init_edges BL_8X8: split[0], split[1], split[2].
        let tree = EdgeTree::build(1);
        let root = tree.root();
        let mut node = root;
        while !node.tip {
            node = tree.split_node(node, 3);
        }
        assert_eq!(tree.split_tip(node, 0), EDGE_I422_LEFT_HAS_BOTTOM);
        assert_eq!(tree.split_tip(node, 1), EDGE_I444_TOP_HAS_RIGHT);
        assert_eq!(tree.split_tip(node, 2), 0);
    }
}
