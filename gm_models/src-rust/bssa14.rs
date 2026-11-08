//! Boore, Stewart, Seyhan and Atkinson (2014) NGA-West2 GMPE ("BSSA_14" in `oq_wrapper`).
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.boore_2014.BooreEtAl2014`,
//! specialised to the configuration `oq_wrapper` instantiates it with for both
//! `ACTIVE_SHALLOW` and `VOLCANIC`: defaults `region="nobasin"`, `sof=True`,
//! `usgs_basin_scaling=False`, `cybershake_basin_adj=False`, with the epistemic
//! branches applied through `sigma_mu_epsilon`. The basin term (Eq. 9, switched
//! off by `region="nobasin"`, so `z1pt0` is not an input), the unspecified
//! style-of-faulting option and the regional/CyberShake options are therefore omitted.
//!
//! The model itself only needs Rjb, but OpenQuake's epistemic adjustment
//! (`get_epistemic_sigma`) reads `rrup`, so both distances are inputs.
//!
//! Equation numbers ("Eq. N") refer to Boore, Stewart, Seyhan and Atkinson
//! (2014), Earthquake Spectra 30(3), 1057-1085.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    // Some coefficients (e.g. 0.318) are near-constants by coincidence.
    #![allow(clippy::approx_constant)]
    include!(concat!(env!("OUT_DIR"), "/bssa14_coefficients.rs"));
}

use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::utils::{check_lengths, nga_west2_epistemic_sigma};
use coeffs::{Coeffs, PGA_COEFFS};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Mref and Rref in Eqs. 2 and 3.
const MREF: f64 = 4.5;
const RREF: f64 = 1.0;
/// Vref in Eqs. 6 and 7.
const VREF: f64 = 760.0;
/// Fnl's f3 (Eq. 7), and the `f1 = 0` of the same equation.
const F3: f64 = 0.1;
/// V1 and V2 of Eq. 15.
const V1: f64 = 225.0;
const V2: f64 = 300.0;

/// A single rupture-site pair (distances in km, rake in degrees).
#[derive(Debug, Clone, Copy)]
pub struct Bssa14Row {
    pub mag: f64,
    pub rake: f64,
    pub rjb: f64,
    pub rrup: f64,
    pub vs30: f64,
}

/// BSSA14 inputs, one value per rupture-site pair. Exposed to Python as
/// `gm_models._core.Bssa14Inputs`. Columns are copied and their lengths checked
/// on construction.
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub struct Bssa14Inputs {
    rows: Vec<Bssa14Row>,
}

#[pymethods]
impl Bssa14Inputs {
    #[new]
    fn new(
        mag: PyReadonlyArray1<f64>,
        rake: PyReadonlyArray1<f64>,
        rjb: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let (mag, rake, rjb, rrup, vs30) = (
            mag.as_array(),
            rake.as_array(),
            rjb.as_array(),
            rrup.as_array(),
            vs30.as_array(),
        );
        let n = check_lengths(&[
            ("mag", mag.len()),
            ("rake", rake.len()),
            ("rjb", rjb.len()),
            ("rrup", rrup.len()),
            ("vs30", vs30.len()),
        ])?;
        let rows = (0..n)
            .map(|i| Bssa14Row {
                mag: mag[i],
                rake: rake[i],
                rjb: rjb[i],
                rrup: rrup[i],
                vs30: vs30[i],
            })
            .collect();
        Ok(Self { rows })
    }
}

/// Magnitude scaling with style of faulting, Eq. 2 (rake classes as in OpenQuake:
/// strike-slip within 30 degrees of horizontal, reverse 30 to 150, otherwise normal).
fn magnitude_scaling(c: &Coeffs, mag: f64, rake: f64) -> f64 {
    let sof = if rake > 30.0 && rake < 150.0 {
        c.e3
    } else if rake.abs() <= 30.0 || 180.0 - rake.abs() <= 30.0 {
        c.e1
    } else {
        c.e2
    };
    let dmag = mag - c.mh;
    sof + if mag <= c.mh {
        c.e4 * dmag + c.e5 * dmag.powi(2)
    } else {
        c.e6 * dmag
    }
}

/// Path scaling, Eq. 3 (global anelastic attenuation, `Dc3 = 0` in the table).
fn path_scaling(c: &Coeffs, mag: f64, rjb: f64) -> f64 {
    let r = rjb.hypot(c.h);
    (c.c1 + c.c2 * (mag - MREF)) * (r / RREF).ln() + (c.c3 + c.dc3) * (r - RREF)
}

/// Mean + standard deviations for `BooreEtAl2014` (`region="nobasin"`), one
/// value per rupture-site pair. SA periods outside the table are an error. Exposed to Python as
/// `gm_models._core.bssa14`.
#[pyfunction]
pub fn bssa14(
    py: Python<'_>,
    inputs: PyRef<'_, Bssa14Inputs>,
    im: IM,
    sigma_mu_epsilon: f64,
) -> PyResult<GMMResult> {
    let coeffs = Coeffs::for_im(im)?;
    let inputs = &*inputs;
    // Releases the GIL: `inputs` owns copies of the arrays.
    Ok(py.detach(|| {
        inputs
            .rows
            .iter()
            .map(|row| compute(&coeffs, row, sigma_mu_epsilon))
            .collect::<GMMResult>()
    }))
}

/// Computes mean + standard deviations for a single rupture-site pair.
///
/// `c` must be `Coeffs::for_im(im)` for the IM of interest.
pub fn compute(c: &Coeffs, row: &Bssa14Row, sigma_mu_epsilon: f64) -> MeanStd {
    let Bssa14Row {
        mag,
        rake,
        rjb,
        rrup,
        vs30,
    } = *row;

    // Median PGA on rock (the Fnl driver), from the PGA coefficients.
    let pga_rock =
        (magnitude_scaling(&PGA_COEFFS, mag, rake) + path_scaling(&PGA_COEFFS, mag, rjb)).exp();

    // Site scaling, Eq. 5: linear (Eq. 6) + nonlinear (Eqs. 7 and 8). Basin term is 0.
    let flin = c.c * (vs30.min(c.vc) / VREF).ln();
    let f2 = c.f4 * ((c.f5 * (vs30.min(VREF) - 360.0)).exp() - (c.f5 * 400.0).exp());
    let fnl = f2 * ((pga_rock + F3) / F3).ln();

    let mean = magnitude_scaling(c, mag, rake)
        + path_scaling(c, mag, rjb)
        + flin
        + fnl
        + sigma_mu_epsilon * nga_west2_epistemic_sigma(mag, rrup);

    // Between-event standard deviation (Eqs. 13 to 17).
    let tau = if mag <= 4.5 {
        c.tau1
    } else if mag >= 5.5 {
        c.tau2
    } else {
        c.tau1 + (c.tau2 - c.tau1) * (mag - 4.5)
    };

    // Within-event standard deviation: magnitude dependence (Eq. 17), distance
    // dependence (Eq. 16) and Vs30 dependence (Eq. 15). OpenQuake applies the
    // two Vs30 branches independently, so at Vs30 == V1 == 225 m/s DfV is
    // subtracted twice; reproduced here.
    let mut phi = if mag <= 4.5 {
        c.f1
    } else if mag >= 5.5 {
        c.f2
    } else {
        c.f1 + (c.f2 - c.f1) * (mag - 4.5)
    };
    if rjb > c.r2 {
        phi += c.dfr;
    } else if rjb > c.r1 {
        phi += c.dfr * (rjb / c.r1).ln() / (c.r2 / c.r1).ln();
    }
    if vs30 <= V1 {
        phi -= c.dfv;
    }
    if (V1..=V2).contains(&vs30) {
        phi -= c.dfv * (V2 / vs30).ln() / (V2 / V1).ln();
    }

    MeanStd {
        mean,
        total: tau.hypot(phi),
        inter: tau,
        intra: phi,
    }
}

/// Verification against OpenQuake's reference CSVs (generated from Boore's
/// Fortran implementation), read from the `external/oq-engine` submodule.
/// `#[ignore]`d so a plain `cargo test` works without it.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv_tol;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> Bssa14Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        // The reference CSVs have no rrup; it only enters the epistemic term,
        // which is off (sigma_mu_epsilon = 0) here.
        Bssa14Row {
            mag: num("rup_mag"),
            rake: num("rup_rake"),
            rjb: num("dist_rjb"),
            rrup: f64::NAN,
            vs30: num("site_vs30"),
        }
    }

    /// OpenQuake's own test accepts 2% on the mean and 1% on the standard
    /// deviations against these Fortran tables (`boore_2014_test.py`:
    /// `MEAN_DISCREP`, `STDDEV_DISCREP`). The mean really needs it: at the
    /// interpolated periods 0.21 and 0.23 s OpenQuake itself is off by 0.1% and 1.8%.
    fn check(file: &str, max_relative: f64, pick: fn(&MeanStd) -> f64) {
        check_csv_tol(
            &format!("BSSA2014/{file}"),
            row,
            |im, row| pick(&compute(&Coeffs::for_im(im).unwrap(), row, 0.0)),
            max_relative,
        );
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn matches_openquake() {
        check("BSSA_2014_MEAN.csv", 0.02, |r| r.mean.exp());
        check("BSSA_2014_TOTAL_STD.csv", 0.01, |r| r.total);
        check("BSSA_2014_INTER_STD.csv", 0.01, |r| r.inter);
        check("BSSA_2014_INTRA_STD.csv", 0.01, |r| r.intra);
    }
}

/// Property tests over the supported input domain. These need no external
/// data, so they run in a plain `cargo test`.
#[cfg(test)]
mod proptests {
    use super::*;
    use approx::relative_eq;
    use proptest::prelude::*;

    fn im() -> impl Strategy<Value = IM> {
        prop_oneof![
            Just(IM::PGA {}),
            Just(IM::PGV {}),
            (0.01..=10.0f64).prop_map(|period| IM::SA { period }),
        ]
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(
            im in im(),
            (mag, rake, rjb, vs30) in (3.0..8.5f64, -180.0..180.0f64, 0.0..400.0f64, 100.0..1500.0f64),
            sigma_mu_epsilon in -1.2815..=1.2815f64,
        ) {
            let c = Coeffs::for_im(im).unwrap();
            let row = Bssa14Row { mag, rake, rjb, rrup: rjb + 1.0, vs30 };
            let r = compute(&c, &row, sigma_mu_epsilon);
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }
    }
}
