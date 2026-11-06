//! Output types shared by all ground motion models.

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

/// Mean and standard deviations for a single rupture-site pair.
#[derive(Debug, Clone, Copy)]
pub struct MeanStd {
    /// Mean, in natural-log space (ln(g) for PGA/SA, ln(cm/s) for PGV).
    pub mean: f64,
    pub total: f64,
    pub inter: f64,
    pub intra: f64,
}

/// Mean and standard deviations for every rupture-site pair, one array per
/// quantity. Exposed to Python as `gm_models._core.GMMResult`.
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug, Default)]
pub struct GMMResult {
    /// Mean, in natural-log space.
    pub ln_mean: Vec<f64>,
    pub std_total: Vec<f64>,
    pub std_inter: Vec<f64>,
    pub std_intra: Vec<f64>,
}

impl FromIterator<MeanStd> for GMMResult {
    fn from_iter<I: IntoIterator<Item = MeanStd>>(iter: I) -> Self {
        let mut result = Self::default();
        for r in iter {
            result.ln_mean.push(r.mean);
            result.std_total.push(r.total);
            result.std_inter.push(r.inter);
            result.std_intra.push(r.intra);
        }
        result
    }
}

#[pymethods]
impl GMMResult {
    #[getter]
    fn ln_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.ln_mean.to_pyarray(py)
    }

    #[getter]
    fn std_total<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.std_total.to_pyarray(py)
    }

    #[getter]
    fn std_inter<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.std_inter.to_pyarray(py)
    }

    #[getter]
    fn std_intra<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.std_intra.to_pyarray(py)
    }
}
