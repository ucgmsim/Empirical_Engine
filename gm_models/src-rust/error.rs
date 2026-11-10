//! Errors shared by all ground motion models.

use crate::im::IM;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

#[derive(Debug, thiserror::Error)]
pub enum GMMError {
    #[error("period {period} s is not supported (range [{min}, {max}] s)")]
    UnsupportedPeriod { period: f64, min: f64, max: f64 },
    #[error("{0:?} is not supported by this model")]
    UnsupportedIM(IM),
    #[error("{x} is outside the interpolation range [{min}, {max}]")]
    OutOfInterpolationRange { x: f64, min: f64, max: f64 },
    #[error("'{name}' has length {len}, expected {expected}")]
    LengthMismatch {
        name: &'static str,
        len: usize,
        expected: usize,
    },
}

impl From<GMMError> for PyErr {
    fn from(err: GMMError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}
