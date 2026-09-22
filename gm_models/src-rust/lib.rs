pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/p21_coefficients.rs"));
}
pub mod nz22_const;
pub mod p21;

use pyo3::prelude::*;

#[pymodule]
mod _utils {
    use crate::p21;

    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    fn parse_im(im: &str, period: f64) -> PyResult<p21::Im> {
        match im {
            "PGA" => Ok(p21::Im::Pga),
            "PGV" => Ok(p21::Im::Pgv),
            "SA" => Ok(p21::Im::Sa(period)),
            other => Err(PyValueError::new_err(format!(
                "Unknown IM '{other}': expected one of 'PGA', 'PGV', 'SA'"
            ))),
        }
    }

    fn check_len(name: &str, len: usize, expected: usize) -> PyResult<()> {
        if len != expected {
            return Err(PyValueError::new_err(format!(
                "'{name}' has length {len}, expected {expected}"
            )));
        }
        Ok(())
    }

    /// Mean + standard deviations for `NZNSHM2022_ParkerEtAl2020SInter`
    /// (global/GLO region), vectorised over `mag`/`rrup`/`vs30`.
    ///
    /// Returns `(mean, std_total, std_inter, std_intra)`, mean in ln space.
    #[pyfunction]
    #[pyo3(signature = (mag, rrup, vs30, im, period, sigma_mu_epsilon, modified_sigma))]
    #[allow(clippy::too_many_arguments)]
    fn _p21_interface<'py>(
        py: Python<'py>,
        mag: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        im: &str,
        period: f64,
        sigma_mu_epsilon: f64,
        modified_sigma: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mag = mag.as_array();
        let rrup = rrup.as_array();
        let vs30 = vs30.as_array();
        let n = mag.len();
        check_len("rrup", rrup.len(), n)?;
        check_len("vs30", vs30.len(), n)?;
        let im = parse_im(im, period)?;

        let mut mean = Vec::with_capacity(n);
        let mut total = Vec::with_capacity(n);
        let mut inter = Vec::with_capacity(n);
        let mut intra = Vec::with_capacity(n);
        for i in 0..n {
            let inputs = p21::Inputs {
                mag: mag[i],
                rrup: rrup[i],
                vs30: vs30[i],
                hypo_depth: 0.0,
                backarc: false,
            };
            let result = p21::compute(
                p21::TectType::Interface,
                im,
                &inputs,
                sigma_mu_epsilon,
                modified_sigma,
            );
            mean.push(result.mean);
            total.push(result.total);
            inter.push(result.inter);
            intra.push(result.intra);
        }

        Ok((
            mean.into_pyarray(py),
            total.into_pyarray(py),
            inter.into_pyarray(py),
            intra.into_pyarray(py),
        ))
    }

    /// Mean + standard deviations for `NZNSHM2022_ParkerEtAl2020SSlab`
    /// (global/GLO region), vectorised over `mag`/`rrup`/`vs30`/`hypo_depth`/`backarc`.
    ///
    /// Returns `(mean, std_total, std_inter, std_intra)`, mean in ln space.
    #[pyfunction]
    #[pyo3(signature = (mag, rrup, vs30, hypo_depth, backarc, im, period, sigma_mu_epsilon, modified_sigma))]
    #[allow(clippy::too_many_arguments)]
    fn _p21_slab<'py>(
        py: Python<'py>,
        mag: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        hypo_depth: PyReadonlyArray1<f64>,
        backarc: PyReadonlyArray1<bool>,
        im: &str,
        period: f64,
        sigma_mu_epsilon: f64,
        modified_sigma: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mag = mag.as_array();
        let rrup = rrup.as_array();
        let vs30 = vs30.as_array();
        let hypo_depth = hypo_depth.as_array();
        let backarc = backarc.as_array();
        let n = mag.len();
        check_len("rrup", rrup.len(), n)?;
        check_len("vs30", vs30.len(), n)?;
        check_len("hypo_depth", hypo_depth.len(), n)?;
        check_len("backarc", backarc.len(), n)?;
        let im = parse_im(im, period)?;

        let mut mean = Vec::with_capacity(n);
        let mut total = Vec::with_capacity(n);
        let mut inter = Vec::with_capacity(n);
        let mut intra = Vec::with_capacity(n);
        for i in 0..n {
            let inputs = p21::Inputs {
                mag: mag[i],
                rrup: rrup[i],
                vs30: vs30[i],
                hypo_depth: hypo_depth[i],
                backarc: backarc[i],
            };
            let result = p21::compute(
                p21::TectType::Slab,
                im,
                &inputs,
                sigma_mu_epsilon,
                modified_sigma,
            );
            mean.push(result.mean);
            total.push(result.total);
            inter.push(result.inter);
            intra.push(result.intra);
        }

        Ok((
            mean.into_pyarray(py),
            total.into_pyarray(py),
            inter.into_pyarray(py),
            intra.into_pyarray(py),
        ))
    }
}
