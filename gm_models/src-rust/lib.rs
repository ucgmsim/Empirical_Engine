//! Rust implementations of ground motion models, exposed to Python as
//! `gm_models._core`. Each model is one Rust module (e.g. [`p21`]) plus a thin
//! Python wrapper (e.g. `gm_models/p21.py`) that validates the rupture
//! dataframe, builds the model's input pyclass, and calls
//! `gm_models.utils.evaluate_ims`.
//!
//! # Tectonic type
//!
//! The public tectonic type is the Python `gm_models.constants.TectType`; there
//! is no shared Rust equivalent. Each model encodes it in whatever actually
//! varies with it, so unsupported or contradictory combinations cannot be
//! constructed.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/p21_coefficients.rs"));
}
pub mod error;
pub mod im;
pub mod nz22_const;
pub mod p21;
pub mod result;
pub mod utils;

use pyo3::prelude::*;

#[pymodule]
mod _core {
    #[pymodule_export]
    use crate::im::IM;
    #[pymodule_export]
    use crate::p21::{p21, P21Inputs};
    #[pymodule_export]
    use crate::result::GMMResult;
}
