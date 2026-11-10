//! Afshari and Stewart (2016) significant duration GMPE ("AS_16" in
//! `oq_wrapper`), for the 5-75%, 5-95% and 20-80% Arias intensity durations
//! (`Ds575`, `Ds595`, `Ds2080`).
//!
//! Transcribed from OpenQuake's
//! `openquake.hazardlib.gsim.afshari_stewart_2016.AfshariStewart2016`,
//! specialised to the only configuration `oq_wrapper` instantiates (active
//! shallow crust, California basin model). The Japan variant
//! (`AfshariStewart2016Japan`, which only swaps the `z1pt0` reference
//! relation) is not ported.
//!
//! Equation numbers ("Eq. N") refer to Afshari and Stewart (2016), "Physically
//! parameterized prediction equations for significant duration in active
//! crustal regions", Earthquake Spectra 32(4), 2057-2081.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

// Table values such as 0.3183 are coefficients, not 1/pi.
#[allow(clippy::approx_constant)]
pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/as16_coefficients.rs"));
}

use crate::error::GMMError;
use crate::gmm::{evaluate, GMM};
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::row_inputs;
use coeffs::Coeffs;
use pyo3::prelude::*;

/// Constants of Eqs. 3 to 9 (`CONSTANTS` in OpenQuake).
const MSTAR: f64 = 6.0;
const R1: f64 = 10.0;
const R2: f64 = 50.0;
const V1: f64 = 600.0;
const DZ1_REF: f64 = 200.0;

/// OpenQuake's "no measured z1pt0" sentinel, replaced by the reference value.
const Z1PT0_MISSING: f64 = -999.0;

row_inputs! {
    /// AS16 inputs, one row per rupture-site pair. Exposed to Python as
    /// `gm_models._core.AS16Inputs`.
    AS16Inputs, AS16Row {
        mag: f64,
        rake: f64,
        rrup: f64,
        vs30: f64,
        /// Metres.
        z1pt0: f64,
    }
}
/// Magnitude scaling term (Eq. 3): the source duration 1/f0, with the corner
/// frequency f0 from Eq. 4, the seismic moment from Eq. 5 and the
/// magnitude-dependent stress drop from Eq. 6. Below `m1` the duration is the
/// constant b0 (style-of-faulting dependent).
fn magnitude_term(c: &Coeffs, mag: f64, rake: f64) -> f64 {
    let (b0, b1) = if (45.0..=135.0).contains(&rake) {
        (c.b0r, c.b1r)
    } else if (-135.0..=-45.0).contains(&rake) {
        (c.b0n, c.b1n)
    } else {
        (c.b0ss, c.b1ss)
    };
    if mag <= c.m1 {
        return b0;
    }
    let m0 = 10f64.powf(1.5 * mag + 16.05);
    let b1 = b1
        + if mag > c.m2 {
            c.b2 * (c.m2 - MSTAR) + c.b3 * (mag - c.m2)
        } else {
            c.b2 * (mag - MSTAR)
        };
    let f0 = 4.9e6 * 3.2 * (b1.exp() / m0).cbrt();
    1.0 / f0
}

/// Distance scaling term, the piecewise-linear path duration of Eq. 7.
fn distance_term(c: &Coeffs, rrup: f64) -> f64 {
    if rrup > R2 {
        c.c1 * R1 + c.c2 * (R2 - R1) + c.c3 * (rrup - R2)
    } else if rrup > R1 {
        c.c1 * R1 + c.c2 * (rrup - R1)
    } else {
        c.c1 * rrup
    }
}

/// Site term: the basin term (Eq. 9) plus the linear Vs30 term. The
/// California reference depth is Eq. 11.
fn site_term(c: &Coeffs, vs30: f64, z1pt0: f64) -> f64 {
    let z1_ref = (-(7.15 / 4.0)
        * ((vs30.powi(4) + 570.94f64.powi(4)) / (1360f64.powi(4) + 570.94f64.powi(4))).ln()
        - 1000f64.ln())
    .exp();
    let z1 = if z1pt0 == Z1PT0_MISSING {
        z1_ref
    } else {
        z1pt0
    };
    let dz1 = z1 - z1_ref;
    let basin = c.c5 * if dz1 > DZ1_REF { DZ1_REF } else { dz1 };
    basin + c.c4 * (vs30.min(V1) / c.vref).ln()
}

/// Mean (ln s) and standard deviations for one rupture-site pair. `tau` is
/// Eq. 14, `phi` is Eq. 15, both piecewise linear in magnitude.
pub fn compute(c: &Coeffs, row: &AS16Row) -> MeanStd {
    let tau = if row.mag >= 7.0 {
        c.tau2
    } else if row.mag < 6.5 {
        c.tau1
    } else {
        c.tau1 + (c.tau2 - c.tau1) * ((row.mag - 6.5) / 0.5)
    };
    let phi = if row.mag >= 5.75 {
        c.phi2
    } else if row.mag < 5.5 {
        c.phi1
    } else {
        c.phi1 + (c.phi2 - c.phi1) * ((row.mag - 5.5) / 0.25)
    };
    MeanStd {
        mean: (magnitude_term(c, row.mag, row.rake) + distance_term(c, row.rrup)).ln()
            + site_term(c, row.vs30, row.z1pt0),
        total: tau.hypot(phi),
        inter: tau,
        intra: phi,
    }
}

/// The AS_16 model: `AfshariStewart2016`. Supports `Ds575`, `Ds595` and
/// `Ds2080`.
pub struct AS16;

impl GMM for AS16 {
    type Row = AS16Row;
    type Coeffs = Coeffs;
    type Params = ();

    fn coeffs(im: IM, (): ()) -> Result<Coeffs, GMMError> {
        Coeffs::for_im(im)
    }

    fn compute(_im: IM, c: &Coeffs, row: &AS16Row, (): ()) -> Result<MeanStd, GMMError> {
        Ok(compute(c, row))
    }
}

/// [`AS16`] for every row. Exposed to Python as `gm_models._core.as16`.
#[pyfunction]
pub fn as16(py: Python<'_>, inputs: PyRef<'_, AS16Inputs>, im: IM) -> PyResult<GMMResult> {
    evaluate::<AS16>(py, &inputs.rows, im, ())
}
/// Verification against OpenQuake's reference tables (generated from the SCEC
/// Broadband Platform implementation), read from the `external/oq-engine`
/// submodule (`#[ignore]`d so a plain `cargo test` works without it; run them
/// with `cargo test -- --ignored`), and against OpenQuake's own output.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv_tol;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> AS16Row {
        let num = |k: &str| -> f64 { rec[k].trim().parse().unwrap() };
        AS16Row {
            mag: num("rup_mag"),
            rake: num("rup_rake"),
            rrup: num("dist_rrup"),
            vs30: num("site_vs30"),
            z1pt0: num("site_z1pt0"),
        }
    }

    fn predict(im: IM, row: &AS16Row) -> MeanStd {
        compute(&Coeffs::for_im(im).unwrap(), row)
    }

    // OpenQuake's own test accepts 3% on the mean: its reference tables are
    // rounded SCEC Broadband Platform output, and differ from OpenQuake's
    // implementation by up to that much (e.g. 2.8% in the first row).
    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake_reference() {
        check_csv_tol(
            "as16/AS16_RSD_MEAN.csv",
            row,
            |im, row| predict(im, row).mean.exp(),
            3e-2,
        );
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn stddevs_match_openquake_reference() {
        for (file, pick) in [
            (
                "AS16_RSD_TOTAL_STDDEV.csv",
                (|r: &MeanStd| r.total) as fn(&MeanStd) -> f64,
            ),
            ("AS16_RSD_INTER_EVENT_STDDEV.csv", |r| r.inter),
            ("AS16_RSD_INTRA_EVENT_STDDEV.csv", |r| r.intra),
        ] {
            check_csv_tol(
                &format!("as16/{file}"),
                row,
                |im, row| pick(&predict(im, row)),
                1e-3,
            );
        }
    }
}

/// Property tests over the supported input domain. These need no external
/// data, so they run in a plain `cargo test`.
#[cfg(test)]
mod proptests {
    use super::*;
    use approx::relative_eq;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(
            im in prop_oneof![Just(IM::Ds575 {}), Just(IM::Ds595 {}), Just(IM::Ds2080 {})],
            mag in 3.0..9.0f64,
            rake in -180.0..180.0f64,
            rrup in 0.0..500.0f64,
            vs30 in 100.0..2000.0f64,
            z1pt0 in prop_oneof![Just(Z1PT0_MISSING), 0.0..2000.0f64],
        ) {
            let c = Coeffs::for_im(im).unwrap();
            let r = compute(&c, &AS16Row { mag, rake, rrup, vs30, z1pt0 });
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }
    }

    #[test]
    fn unsupported_ims_are_rejected() {
        for im in [
            IM::PGA {},
            IM::SA { period: 1.0 },
            IM::EAS { frequency: 1.0 },
        ] {
            assert!(Coeffs::for_im(im).is_err());
        }
    }
}
