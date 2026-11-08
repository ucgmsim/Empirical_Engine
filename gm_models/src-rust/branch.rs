//! Epistemic branch for models whose lower/upper adjustments are not a
//! symmetric multiple of one standard deviation (A_22, S_22).

use pyo3::prelude::*;

/// Epistemic mean branch. Exposed to Python as `gm_models._core.Branch`.
#[pyclass(frozen, eq, eq_int, from_py_object, module = "gm_models._core")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Branch {
    Lower,
    Central,
    Upper,
}
