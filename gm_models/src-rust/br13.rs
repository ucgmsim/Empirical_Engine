//! Bradley (2013) New Zealand active shallow crust GMPE ("Br_13" in
//! `oq_wrapper`), with OpenQuake's `sigma_mu_epsilon` epistemic adjustment.
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.bradley_2013`
//! (`Bradley2013` for active shallow, `Bradley2013Volc` for volcanic),
//! specialised to default constructor arguments, the only configuration
//! `oq_wrapper` uses. The Christchurch (`Chch*`, `*AdditionalSigma`, `*Maps`)
//! and larger-of-two-components variants are omitted, so the additional sigma
//! is always 0.
//!
//! Equation numbers ("Eq. N") refer to Bradley (2013), "A New Zealand-Specific
//! Pseudospectral Acceleration Ground-Motion Prediction Equation for Active
//! Shallow Crustal Earthquakes Based on Foreign Models", BSSA 103(3).
//!
//! The volcanic class differs only in the Taupo Volcanic Zone path distance
//! `rtvz`, which OpenQuake sets equal to `rrup` (the whole path is in the TVZ)
//! rather than taking it as an input, so the volcanic variant needs no extra
//! column. The model supports PGA and SA only (not PGV). PGA uses period 0 in
//! the V1 term of Eq. 6.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

// Tabulated coefficients that happen to look like math constants.
#[allow(clippy::approx_constant)]
pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/br13_coefficients.rs"));
}

use crate::error::GMMError;
use crate::gmm::{evaluate, GMM};
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::row_inputs;
use crate::utils::nga_west2_epistemic_sigma;
use coeffs::Coeffs;
use pyo3::prelude::*;

/// A single rupture-site pair of [`Br13Inputs`]. `z1pt0` is in metres, angles
/// in degrees. `volcanic` rows use `Bradley2013Volc`.
#[derive(Debug, Clone, Copy)]
pub struct Br13Row {
    pub mag: f64,
    pub dip: f64,
    pub rake: f64,
    pub ztor: f64,
    pub rrup: f64,
    pub rjb: f64,
    pub rx: f64,
    pub vs30: f64,
    pub vs30measured: bool,
    pub z1pt0: f64,
    pub is_aftershock: bool,
    pub volcanic: bool,
}

row_inputs! {
    /// Br13 inputs, one row per rupture-site pair. Exposed to Python as
    /// `gm_models._core.Br13Inputs`, built with `Br13Inputs.crust(...)` or
    /// `Br13Inputs.volcanic(...)`.
    Br13Inputs, Br13Row {
        /// Active shallow crust inputs (`Bradley2013`).
        crust(
            mag: f64,
            dip: f64,
            rake: f64,
            ztor: f64,
            rrup: f64,
            rjb: f64,
            rx: f64,
            vs30: f64,
            vs30measured: bool,
            z1pt0: f64,
            is_aftershock: bool,
        ) {
            volcanic: false,
        },
        /// Volcanic inputs (`Bradley2013Volc`).
        volcanic(
            mag: f64,
            dip: f64,
            rake: f64,
            ztor: f64,
            rrup: f64,
            rjb: f64,
            rx: f64,
            vs30: f64,
            vs30measured: bool,
            z1pt0: f64,
            is_aftershock: bool,
        ) {
            volcanic: true,
        },
    }
}

/// ln of the reference-rock median, Eq. 4. Volcanic rows set `rtvz = rrup`.
fn ln_y_ref(c: &Coeffs, r: &Br13Row) -> f64 {
    let (mag, rrup) = (r.mag, r.rrup);
    let rtvz = if r.volcanic { rrup } else { 0.0 };
    let frv = f64::from((30.0..=150.0).contains(&r.rake));
    let fnm = f64::from((-120.0..=-60.0).contains(&r.rake));
    let fhw = f64::from(r.rx >= 0.0);
    let aftershock = f64::from(r.is_aftershock);

    c.c1 + (c.c1a * frv + c.c1b * fnm + c.c7 * (r.ztor.min(c.c8) - 4.0)) * (1.0 - aftershock)
        + (c.c10 + c.c7a * (r.ztor - 4.0)) * aftershock
        + c.c2 * (mag - 6.0)
        + ((c.c2 - c.c3) / c.cn) * (1.0 + (c.cn * (c.cm - mag)).exp()).ln()
        + c.c4 * (rrup + c.c5 * (c.c6 * (mag - c.chm).max(0.0)).cosh()).ln()
        + (c.c4a - c.c4) * rrup.hypot(c.crb).ln()
        + (c.cg1 + c.cg2 / (mag - c.cg3).max(0.0).cosh()) * ((1.0 + c.ctvz * (rtvz / rrup)) * rrup)
        + c.c9
            * fhw
            * (r.rx * r.dip.to_radians().cos().powi(2) / c.c9a).tanh()
            * (1.0 - r.rjb.hypot(r.ztor) / (rrup + 0.001))
}

/// Period-dependent Vs30 above which the site amplification is constant,
/// Eqs. 2 and 6 (1800 m/s for PGA).
fn v1(period: f64) -> f64 {
    if period == 0.0 {
        1800.0
    } else {
        (1130.0 * (period / 0.75).powf(-0.11)).clamp(1130.0, 1800.0)
    }
}

/// The Br_13 model: `Bradley2013`, or `Bradley2013Volc` for volcanic rows.
/// SA periods outside the table are an error. PGV is an error.
pub struct Br13;

impl GMM for Br13 {
    type Row = Br13Row;
    type Coeffs = Coeffs;
    /// `sigma_mu_epsilon`.
    type Params = f64;

    fn coeffs(im: IM, _: f64) -> Result<Coeffs, GMMError> {
        Coeffs::for_im(im)
    }

    fn compute(
        im: IM,
        c: &Coeffs,
        row: &Br13Row,
        sigma_mu_epsilon: f64,
    ) -> Result<MeanStd, GMMError> {
        Ok(compute(im, c, row, sigma_mu_epsilon))
    }
}

/// [`Br13`] for every row. The tectonic type is set by how `inputs` was
/// built. Exposed to Python as `gm_models._core.br13`.
#[pyfunction]
pub fn br13(
    py: Python<'_>,
    inputs: PyRef<'_, Br13Inputs>,
    im: IM,
    sigma_mu_epsilon: f64,
) -> PyResult<GMMResult> {
    evaluate::<Br13>(py, &inputs.rows, im, sigma_mu_epsilon)
}

/// Computes mean + standard deviations for a single rupture-site pair. `c`
/// must be `Coeffs::for_im(im)`.
pub fn compute(im: IM, c: &Coeffs, r: &Br13Row, sigma_mu_epsilon: f64) -> MeanStd {
    let period = match im {
        IM::SA { period } => period,
        _ => 0.0,
    };
    let ln_y_ref = ln_y_ref(c, r);
    let y_ref = ln_y_ref.exp();
    // exp1 and exp2 of Eq. 7.
    let exp1 = (c.phi3 * (r.vs30.min(1130.0) - 360.0)).exp();
    let exp2 = (c.phi3 * (1130.0 - 360.0)).exp();

    // Basin term; z1pt0 == -999 is replaced by the Chiou and Youngs (2008)
    // Vs30-based reference.
    let z1pt0 = if r.z1pt0 == -999.0 {
        (28.5 - (3.82 / 8.0) * (r.vs30.powi(8) + 378.7f64.powi(8)).ln()).exp()
    } else {
        r.z1pt0
    };
    let f_basin = c.phi5 * (1.0 - 1.0 / (c.phi6 * (z1pt0 - c.phi7).max(0.0)).cosh())
        + c.phi8 / (0.15 * (z1pt0 - 15.0).max(0.0)).cosh();

    // Eq. 5, with the sigma_mu_epsilon epistemic term.
    let mean = ln_y_ref
        + c.phi1 * (r.vs30.min(v1(period)) / 1130.0).ln()
        + c.phi2 * (exp1 - exp2) * ((y_ref + c.phi4) / c.phi4).ln()
        + f_basin
        + sigma_mu_epsilon * nga_west2_epistemic_sigma(r.mag, r.rrup);

    // Eqs. 19 to 21 (no additional sigma).
    let mag_test = (r.mag - 5.0).clamp(0.0, 2.0);
    let tau = c.tau1 + (c.tau2 - c.tau1) / 2.0 * mag_test;
    let nl = c.phi2 * (exp1 - exp2) * y_ref / (y_ref + c.phi4);
    let phi_vs = if r.vs30measured { 0.7 } else { c.sig3 };
    let intra = (c.sig1 + 0.5 * (c.sig2 - c.sig1) * mag_test + c.sig4 * f64::from(r.is_aftershock))
        * (phi_vs + (1.0 + nl).powi(2)).sqrt();
    let inter = ((1.0 + nl) * tau).abs();

    MeanStd {
        mean,
        total: (inter.powi(2) + intra.powi(2)).sqrt(),
        inter,
        intra,
    }
}

/// Verification against OpenQuake's reference tables (generated from Bradley's
/// MATLAB code), read from the `external/oq-engine` submodule. `#[ignore]`d so
/// a plain `cargo test` works without it. The files are those of OpenQuake's
/// `Bradley2013TestCase` (SS, RV, NM) and `Bradley2013VolcTestCase`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> Br13Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        Br13Row {
            mag: num("rup_mag"),
            dip: num("rup_dip"),
            rake: num("rup_rake"),
            ztor: num("rup_ztor"),
            rrup: num("dist_rrup"),
            rjb: num("dist_rjb"),
            rx: num("dist_rx"),
            vs30: num("site_vs30"),
            vs30measured: num("site_vs30measured") != 0.0,
            z1pt0: num("site_z1pt0"),
            is_aftershock: false,
            volcanic: false,
        }
    }

    fn check(file: &str, volcanic: bool, pick: fn(&MeanStd) -> f64) {
        check_csv(
            &format!("BRADLEY13/{file}"),
            |rec| Br13Row {
                volcanic,
                ..row(rec)
            },
            |im, row| pick(&compute(im, &Coeffs::for_im(im).unwrap(), row, 0.0)),
        );
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn matches_openquake() {
        for style in ["SS", "RV", "NM"] {
            check(&format!("Bradley2013_MEAN_{style}.csv"), false, |r| {
                r.mean.exp()
            });
            check(
                &format!("Bradley2013_TOTAL_STDDEV_{style}.csv"),
                false,
                |r| r.total,
            );
            check(
                &format!("Bradley2013_INTER_EVENT_STDDEV_{style}.csv"),
                false,
                |r| r.inter,
            );
            check(
                &format!("Bradley2013_INTRA_EVENT_STDDEV_{style}.csv"),
                false,
                |r| r.intra,
            );
        }
        check("Bradley2013Volc_MEAN.csv", true, |r| r.mean.exp());
        check("Bradley2013Volc_TOTAL_STDDEV.csv", true, |r| r.total);
        check("Bradley2013Volc_INTER_EVENT_STDDEV.csv", true, |r| r.inter);
        check("Bradley2013Volc_INTRA_EVENT_STDDEV.csv", true, |r| r.intra);
    }
}

/// Property tests over the supported input domain (no external data).
#[cfg(test)]
mod proptests {
    use super::*;
    use approx::relative_eq;
    use proptest::prelude::*;

    fn im() -> impl Strategy<Value = IM> {
        prop_oneof![
            Just(IM::PGA {}),
            (0.01..=10.0f64).prop_map(|period| IM::SA { period }),
        ]
    }

    fn row() -> impl Strategy<Value = Br13Row> {
        (
            (3.5..8.5f64, 10.0..90.0f64, -180.0..180.0f64, 0.0..20.0f64),
            (0.1..500.0f64, 0.0..500.0f64, -200.0..200.0f64),
            (
                150.0..1500.0f64,
                any::<bool>(),
                0.0..2000.0f64,
                any::<bool>(),
                any::<bool>(),
            ),
        )
            .prop_map(
                |(
                    (mag, dip, rake, ztor),
                    (rrup, rjb, rx),
                    (vs30, vs30measured, z1pt0, is_aftershock, volcanic),
                )| Br13Row {
                    mag,
                    dip,
                    rake,
                    ztor,
                    rrup,
                    rjb: rjb.min(rrup),
                    rx,
                    vs30,
                    vs30measured,
                    z1pt0,
                    is_aftershock,
                    volcanic,
                },
            )
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(
            im in im(),
            row in row(),
            sigma_mu_epsilon in -1.2815..=1.2815f64,
        ) {
            let r = compute(im, &Coeffs::for_im(im).unwrap(), &row, sigma_mu_epsilon);
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }
    }

    #[test]
    fn pgv_is_unsupported() {
        assert!(Coeffs::for_im(IM::PGV {}).is_err());
    }
}
