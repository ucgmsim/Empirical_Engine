//! Intensity measures shared by all ground motion models.

use pyo3::prelude::*;

/// Intensity measure. Exposed to Python as `gm_models._core.IM`, so string
/// parsing stays on the Python side.
#[pyclass(frozen, eq, from_py_object, module = "gm_models._core")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IM {
    PGA {},
    PGV {},
    /// Pseudo-spectral acceleration at `period` seconds.
    SA {
        period: f64,
    },
}
