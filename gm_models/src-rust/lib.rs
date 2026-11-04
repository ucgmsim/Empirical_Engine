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
