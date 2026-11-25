//! Skaler (referans) DSP backend'i — S15.4 pilotu.
//!
//! Skaler yol **her zaman** mevcuttur ve davranışın tek kaynağıdır (DSP-01/6,
//! CEK-02): ISA yolları (F4+) yalnız bu modülle **bit-exact** olduklarını
//! kanıtladıktan sonra devreye alınır. Bu yüzden skaler kod "yavaş sürüm"
//! değil, **sözleşmenin kendisidir**.
//!
//! Çalışma alanındaki `dsp/scalar` ağacı burada açılır: CDEF (S15.4), MC
//! (S15.5), itx (S2b.1), intra tahmin/CfL/palette (S2b.2), loop restoration
//! (S2c.2b), superres büyütme (S2c.3), film grain (S2d.1); F3c'de loop filter
//! (`lf`) ve refmvs çekirdekleri çözme hattından buraya taşındı. ISA yolları
//! F4'te eklenir.

#![forbid(unsafe_code)]

pub mod cdef;
pub mod fg;
pub mod ipred;
pub mod itx;
pub mod itx_1d;
pub mod lf;
pub mod lr;
pub mod mc;
pub mod pal;
pub mod refmvs;
pub mod resize;
