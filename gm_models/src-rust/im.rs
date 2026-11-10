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
    /// Cumulative absolute velocity.
    CAV {},
    /// Significant durations (OpenQuake's `RSD575`, `RSD595`, `RSD2080`).
    Ds575 {},
    Ds595 {},
    Ds2080 {},
    /// Effective amplitude spectrum at `frequency` Hz.
    EAS {
        frequency: f64,
    },
}
