//! Rust implementations of ground motion models, exposed to Python as
//! `gm_models._core`. Each model is one Rust module (e.g. [`p21`]): its input
//! pyclass from [`row_inputs!`], a [`gmm::GMM`] implementation and a
//! pyfunction calling [`gmm::evaluate`]. The Python side (`gm_models.Model`)
//! builds the inputs from the rupture dataframe, the constructor signatures
//! naming the columns, and calls the pyfunction per IM.
//!
//! # Tectonic type
//!
//! The public tectonic type is the Python `gm_models.constants.TectType`; there
//! is no shared Rust equivalent. Each model encodes it in whatever actually
//! varies with it, so unsupported or contradictory combinations cannot be
//! constructed.

pub mod a18;
pub mod a22;
pub mod ag20;
pub mod as16;
pub mod ask14;
pub mod ba18;
pub mod bch16;
pub mod br13;
pub mod branch;
pub mod bssa14;
pub mod cb10;
pub mod cb14;
pub mod cy14;
pub mod error;
pub mod gmm;
pub mod im;
pub mod k20;
pub mod nz22_const;
#[cfg(test)]
mod oq_test;
pub mod p20;
pub mod p21;
pub mod result;
pub mod s22;
pub mod utils;
pub mod za06;

use pyo3::prelude::*;

#[pymodule]
mod _core {
    #[pymodule_export]
    use crate::a18::{a18, A18Inputs};
    #[pymodule_export]
    use crate::a22::{a22, A22Inputs};
    #[pymodule_export]
    use crate::ag20::{ag20, ag20_nz, AG20Inputs};
    #[pymodule_export]
    use crate::as16::{as16, AS16Inputs};
    #[pymodule_export]
    use crate::ask14::{ask14, Ask14Inputs};
    #[pymodule_export]
    use crate::ba18::{ba18, BA18Inputs};
    #[pymodule_export]
    use crate::bch16::{bch16, Bch16Inputs};
    #[pymodule_export]
    use crate::br13::{br13, Br13Inputs};
    #[pymodule_export]
    use crate::branch::Branch;
    #[pymodule_export]
    use crate::bssa14::{bssa14, Bssa14Inputs};
    #[pymodule_export]
    use crate::cb10::{cb10, CB10Inputs};
    #[pymodule_export]
    use crate::cb14::{cb14, CB14Inputs};
    #[pymodule_export]
    use crate::cy14::{cy14, CY14Inputs};
    #[pymodule_export]
    use crate::im::IM;
    #[pymodule_export]
    use crate::k20::{k20, k20_nz, K20Inputs};
    #[pymodule_export]
    use crate::p20::{p20, P20Inputs};
    #[pymodule_export]
    use crate::p21::{p21, P21Inputs};
    #[pymodule_export]
    use crate::result::GMMResult;
    #[pymodule_export]
    use crate::s22::{s22, S22Inputs};
    #[pymodule_export]
    use crate::za06::{za06, Za06Inputs};
}
