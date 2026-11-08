//! Bayless and Abrahamson (2018, 2019) Fourier amplitude spectrum GMPE
//! ("BA_18" in `oq_wrapper`), reported as the effective amplitude spectrum
//! (`EAS`) in g-s.
//!
//! Transcribed from OpenQuake's
//! `openquake.hazardlib.gsim.bayless_abrahamson_2018.BaylessAbrahamson2018`.
//! The class has no constructor options and `oq_wrapper` only instantiates it
//! for active shallow crust, so nothing is dropped. Like OpenQuake, the
//! smoothing of the non-linear site term described in the paper is not
//! implemented.
//!
//! References: Bayless and Abrahamson (2019), "Summary of the BA18
//! ground-motion model for Fourier amplitude spectra for crustal earthquakes
//! in California", BSSA 109(5), 2088-2105 (page and equation numbers below are
//! those OpenQuake cites).
//!
//! OpenQuake keys its EAS coefficient table by `IMT('EAS(%.6f)', period=1/f)`
//! and interpolates in `ln(period)`, so the table is stored by period (see
//! `scripts/export_coeffs.py`) and read with [`Coeffs::for_sa_period`] at
//! `1/frequency`. Frequencies outside [0.1, 100] Hz are errors.
//!
//! # Above the regression range (deviates from OpenQuake)
//!
//! The coefficients are regressed up to 24 Hz (the 23.988321 Hz row); the
//! table rows above it hold placeholders (-9999.9999) for every coefficient
//! except the standard deviation components s1 to s6 and f3 to f5. Above the
//! last regressed row the median is the kappa extrapolation of Eq. 17
//! (p. 2096), EAS(f) = EAS(24 Hz) D(kappa, f - 24). Two OpenQuake behaviours
//! are deliberately not reproduced, as both read placeholders:
//!
//! - OpenQuake's total standard deviation (Eq. 18) takes c1a from the row at f,
//!   so it is about 1e4 above 23.988321 Hz. c1a is the difference between the
//!   regressed and the slope-constrained c1 at low frequencies (p. 2095), and
//!   is 0 wherever c1 was not modified (4.6 to 24 Hz in the table), so here it
//!   is 0 above the regression range. This reproduces the authors' reference
//!   values (`BA18_STD_TOTAL_reduced.csv`, e.g. 0.9435 at 26.9 Hz), which
//!   OpenQuake's own test never reads.
//! - OpenQuake switches to the extrapolation only at 24 Hz, so between
//!   23.988321 and 24 Hz its mean interpolates towards the placeholder c1.
//!   Here the extrapolation starts at the last regressed row, where it is
//!   continuous (D = 1).
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/ba18_coefficients.rs"));
}

use crate::error::GMMError;
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::utils::check_lengths;
use coeffs::{Coeffs, SA_COEFFS};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Frequency (Hz) whose coefficients give the reference-rock Ir (consistent
/// with the authors' MATLAB script).
const IR_FREQUENCY: f64 = 5.0118725;
/// Highest regressed frequency (Hz), the paper's "24 Hz". Above it the model
/// extrapolates from this row with a kappa diminution factor (Eq. 17).
const MAX_FREQUENCY: f64 = 23.988321;
/// Reference Vs30 (m/s) of the non-linear site term.
const VS_REF: f64 = 760.0;

/// OpenQuake's "no measured z1pt0" sentinel, replaced by the reference value.
const Z1PT0_MISSING: f64 = -999.0;

/// BA18 input columns, one value per rupture-site pair. `z1pt0` is in metres.
/// Exposed to Python as `gm_models._core.BA18Inputs`. Column lengths are
/// checked in [`BA18Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub struct BA18Inputs {
    mag: Vec<f64>,
    rake: Vec<f64>,
    ztor: Vec<f64>,
    rrup: Vec<f64>,
    vs30: Vec<f64>,
    z1pt0: Vec<f64>,
}

/// A single row of [`BA18Inputs`].
#[derive(Debug, Clone, Copy)]
pub struct BA18Row {
    pub mag: f64,
    pub rake: f64,
    pub ztor: f64,
    pub rrup: f64,
    pub vs30: f64,
    pub z1pt0: f64,
}

impl BA18Inputs {
    /// The rupture-site pairs, in order. Fails if the columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = BA18Row> + '_, GMMError> {
        let n = check_lengths(&[
            ("mag", self.mag.len()),
            ("rake", self.rake.len()),
            ("ztor", self.ztor.len()),
            ("rrup", self.rrup.len()),
            ("vs30", self.vs30.len()),
            ("z1pt0", self.z1pt0.len()),
        ])?;
        Ok((0..n).map(|i| BA18Row {
            mag: self.mag[i],
            rake: self.rake[i],
            ztor: self.ztor[i],
            rrup: self.rrup[i],
            vs30: self.vs30[i],
            z1pt0: self.z1pt0[i],
        }))
    }
}

#[pymethods]
impl BA18Inputs {
    #[new]
    fn new(
        mag: PyReadonlyArray1<f64>,
        rake: PyReadonlyArray1<f64>,
        ztor: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        z1pt0: PyReadonlyArray1<f64>,
    ) -> Self {
        Self {
            mag: mag.as_array().to_vec(),
            rake: rake.as_array().to_vec(),
            ztor: ztor.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
            z1pt0: z1pt0.as_array().to_vec(),
        }
    }
}

/// Source and path terms: magnitude scaling, path scaling, top-of-rupture
/// scaling (capped at 20 km) and the normal-faulting term (normal for
/// -150 < rake < -30).
fn source_path_terms(c: &Coeffs, row: &BA18Row) -> f64 {
    let mag_scaling = c.c1
        + c.c2 * (row.mag - 6.0)
        + (c.c2 - c.c3) / c.cn * (1.0 + (c.cn * (c.cm - row.mag)).exp()).ln();
    let path_scaling = c.c4 * (row.rrup + c.c5 * (c.c6 * (row.mag - c.chm).max(0.0)).cosh()).ln()
        + (-0.5 - c.c4) * (row.rrup.powi(2) + 50f64.powi(2)).sqrt().ln()
        + c.c7 * row.rrup;
    let normal = if row.rake > -150.0 && row.rake < -30.0 {
        c.c10
    } else {
        0.0
    };
    mag_scaling + path_scaling + c.c9 * row.ztor.min(20.0) + normal
}

/// Linear site term plus the soil-depth (basin) term, with c11 by Vs30 class
/// (Eq. 13b, p. 2093).
fn site_terms(c: &Coeffs, row: &BA18Row) -> f64 {
    let c11 = if row.vs30 > 500.0 {
        c.c11d
    } else if row.vs30 > 300.0 {
        c.c11c
    } else if row.vs30 > 200.0 {
        c.c11b
    } else {
        c.c11a
    };
    // Reference depth (m).
    let z1_ref = (-7.67 / 4.0
        * ((row.vs30.powi(4) + 610f64.powi(4)) / (1360f64.powi(4) + 610f64.powi(4))).ln())
    .exp();
    let z1 = if row.z1pt0 == Z1PT0_MISSING {
        z1_ref
    } else {
        row.z1pt0
    };
    // Depths enter in km, capped at 2 km.
    let basin = c11 * (((z1 / 1000.0).min(2.0) + 0.01) / (z1_ref / 1000.0 + 0.01)).ln();
    c.c8 * (row.vs30.min(1000.0) / 1000.0).ln() + basin
}

/// Non-linear site term (from the f3, f4, f5 of every tabulated frequency) at
/// `frequency`, for reference-rock peak acceleration `ln_ir`. Applies the
/// correction (p. 2093) that holds the term at its minimum above the
/// frequency where that minimum occurs, then interpolates linearly in
/// frequency (clamped at the ends, like `numpy.interp`).
fn nonlinear_site_term(frequency: f64, vs30: f64, ln_ir: f64) -> f64 {
    let ir = ln_ir.exp();
    // Ascending frequency, i.e. descending period.
    let (freqs, mut fnl): (Vec<f64>, Vec<f64>) = SA_COEFFS
        .iter()
        .rev()
        .zip(coeffs::SA_PERIODS.iter().rev())
        .map(|(c, period)| {
            let f2 = c.f4
                * ((c.f5 * (vs30.min(VS_REF) - 360.0)).exp() - (c.f5 * (VS_REF - 360.0)).exp());
            (1.0 / period, f2 * ((ir + c.f3) / c.f3).ln())
        })
        .unzip();
    let (mut idx, mut min) = (0, fnl[0]);
    for (j, &v) in fnl.iter().enumerate() {
        if v < min {
            (idx, min) = (j, v);
        }
    }
    fnl[idx..].fill(min);
    let i = freqs.partition_point(|&f| f < frequency);
    if i == 0 {
        fnl[0]
    } else if i == freqs.len() {
        fnl[i - 1]
    } else {
        fnl[i - 1] + (frequency - freqs[i - 1]) / (freqs[i] - freqs[i - 1]) * (fnl[i] - fnl[i - 1])
    }
}

/// Standard deviations from the magnitude-dependent `s*` coefficients
/// (Eq. 19): `(total, tau, phi)`. Total also contains `c1a` (Eq. 18), so
/// total^2 != tau^2 + phi^2 where `c1a` is non-zero.
fn stddevs(c: &Coeffs, c1a: f64, mag: f64) -> (f64, f64, f64) {
    let blend = |lo: f64, hi: f64| {
        if mag > 6.0 {
            hi
        } else if mag > 4.0 {
            lo + (hi - lo) / 2.0 * (mag - 4.0)
        } else {
            lo
        }
    };
    let (tau, phi_s2s, phi_ss) = (blend(c.s1, c.s2), blend(c.s3, c.s4), blend(c.s5, c.s6));
    (
        (tau.powi(2) + phi_s2s.powi(2) + phi_ss.powi(2) + c1a.powi(2)).sqrt(),
        tau,
        phi_s2s.hypot(phi_ss),
    )
}

/// The frequency as OpenQuake sees it: `oq_wrapper` runs the model through a
/// `ContextMaker`, which re-parses IMTs from their string `EAS(%.6f)`, so
/// requested frequencies are rounded to 6 decimals. (The Ir lookup builds its
/// IMT directly, so it keeps the unrounded [`IR_FREQUENCY`].)
pub fn round_frequency(frequency: f64) -> f64 {
    format!("{frequency:.6}").parse().unwrap()
}

/// Mean (ln g-s) and standard deviations for one rupture-site pair at
/// `frequency` Hz.
///
/// `frequency` must be rounded by [`round_frequency`], and `c` must be
/// `Coeffs::for_sa_period(1.0 / frequency)`. Errors if the
/// reference coefficients (fixed frequencies, always in the table) cannot be
/// looked up.
pub fn compute(frequency: f64, c: &Coeffs, row: &BA18Row) -> Result<MeanStd, GMMError> {
    // OpenQuake recovers the frequency as 1 / period, with period = 1 / frequency.
    let frequency = 1.0 / (1.0 / frequency);
    // Peak acceleration at rock outcrop, Eq. 10e (p. 2092).
    let ln_ir = 1.238 + 0.846 * source_path_terms(&Coeffs::for_sa_period(1.0 / IR_FREQUENCY)?, row);
    let above_regression = frequency > MAX_FREQUENCY;
    let linear = if above_regression {
        let c_max = Coeffs::for_sa_period(1.0 / MAX_FREQUENCY)?;
        let kappa = (-0.4 * (row.vs30 / 760.0).ln() - 3.5).exp();
        let diminution = (-std::f64::consts::PI * kappa * (frequency - MAX_FREQUENCY)).exp();
        (diminution * (source_path_terms(&c_max, row) + site_terms(&c_max, row)).exp()).ln()
    } else {
        source_path_terms(c, row) + site_terms(c, row)
    };
    // c1a only exists where c1 was constrained (low frequencies); the table's
    // placeholder above the regression range is not a value (see module doc).
    let (total, inter, intra) = stddevs(c, if above_regression { 0.0 } else { c.c1a }, row.mag);
    Ok(MeanStd {
        mean: linear + nonlinear_site_term(frequency, row.vs30, ln_ir),
        total,
        inter,
        intra,
    })
}

/// Mean + standard deviations for `BaylessAbrahamson2018`, one value per
/// rupture-site pair in `inputs`. Only `IM::EAS` is supported, at frequencies
/// rounded as by [`round_frequency`]. Exposed to Python as
/// `gm_models._core.ba18`.
#[pyfunction]
pub fn ba18(py: Python<'_>, inputs: PyRef<'_, BA18Inputs>, im: IM) -> PyResult<GMMResult> {
    let IM::EAS { frequency } = im else {
        return Err(GMMError::UnsupportedIM(im).into());
    };
    let frequency = round_frequency(frequency);
    let coeffs = Coeffs::for_sa_period(1.0 / frequency)?;
    let inputs = &*inputs;
    // Releases the GIL: `inputs` owns copies of the arrays.
    Ok(py.detach(|| {
        inputs
            .iter()?
            .map(|row| compute(frequency, &coeffs, &row))
            .collect::<Result<GMMResult, GMMError>>()
    })?)
}

/// Verification against OpenQuake's reference tables, read from the
/// `external/oq-engine` submodule (`#[ignore]`d so a plain `cargo test` works
/// without it; run with `cargo test -- --ignored`). The exact comparison
/// with OpenQuake below 24 Hz, including the stddevs, is the `oq_wrapper`
/// benchmark fixture in `tests/test_oq_wrapper_parity.py`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv_tol;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> BA18Row {
        let num = |k: &str| -> f64 { rec[k].trim().parse().unwrap() };
        BA18Row {
            mag: num("rup_mag"),
            rake: num("rup_rake"),
            ztor: num("rup_ztor"),
            rrup: num("dist_rrup"),
            vs30: num("site_vs30"),
            z1pt0: num("site_z1pt0"),
        }
    }

    fn predict(im: IM, row: &BA18Row) -> MeanStd {
        let IM::EAS { frequency } = im else {
            panic!("not an EAS column")
        };
        let frequency = round_frequency(frequency);
        compute(
            frequency,
            &Coeffs::for_sa_period(1.0 / frequency).unwrap(),
            row,
        )
        .unwrap()
    }

    // OpenQuake's own test accepts 0.8% on the mean (the reference tables come
    // from the authors' MATLAB script, without the smoothing OpenQuake omits).
    // Its `BA18_STD_TOTAL_reduced.csv` is not checked here: OpenQuake's test
    // never compares it (its result_type `TOTAL` is not one the harness looks
    // for), and its 26.9 and 39.8 Hz columns disagree with OpenQuake itself
    // (see the module doc). The stddevs are covered by the `oq_wrapper`
    // benchmark fixture.
    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake_reference() {
        check_csv_tol(
            "BA18/BA18_mean_reduced.csv",
            row,
            |im, row| predict(im, row).mean.exp(),
            8e-3,
        );
    }

    /// The authors' total standard deviations, including 26.9 and 39.8 Hz,
    /// where OpenQuake differs (see the module doc).
    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn total_stddev_matches_authors_reference() {
        check_csv_tol(
            "BA18/BA18_STD_TOTAL_reduced.csv",
            row,
            |im, row| predict(im, row).total,
            1e-3,
        );
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
            frequency in 0.1..100.0f64,
            mag in 3.0..8.5f64,
            rake in -180.0..180.0f64,
            ztor in 0.0..20.0f64,
            rrup in 0.1..400.0f64,
            vs30 in 150.0..1500.0f64,
            z1pt0 in prop_oneof![Just(Z1PT0_MISSING), 0.0..3000.0f64],
        ) {
            let frequency = round_frequency(frequency);
            let c = Coeffs::for_sa_period(1.0 / frequency).unwrap();
            let r = compute(frequency, &c, &BA18Row { mag, rake, ztor, rrup, vs30, z1pt0 }).unwrap();
            let c1a = if frequency > MAX_FREQUENCY { 0.0 } else { c.c1a };
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            // Unlike most models, total also includes c1a: total^2 = inter^2 + intra^2 + c1a^2.
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2) + c1a.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 + c1a^2 {rhs}");
        }

        #[test]
        fn frequencies_outside_table_are_rejected(frequency in prop_oneof![0.001..0.0999f64, 100.1..1e3f64]) {
            prop_assert!(Coeffs::for_sa_period(1.0 / frequency).is_err(), "frequency {frequency} accepted");
        }
    }

    #[test]
    fn only_eas_is_supported() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let inputs = Py::new(
                py,
                BA18Inputs {
                    mag: vec![],
                    rake: vec![],
                    ztor: vec![],
                    rrup: vec![],
                    vs30: vec![],
                    z1pt0: vec![],
                },
            )
            .unwrap();
            assert!(ba18(py, inputs.borrow(py), IM::PGA {}).is_err());
        });
    }

    #[test]
    fn inputs_reject_mismatched_lengths() {
        let col = |n| vec![1.0; n];
        let inputs = |z1pt0_len| BA18Inputs {
            mag: col(2),
            rake: col(2),
            ztor: col(2),
            rrup: col(2),
            vs30: col(2),
            z1pt0: col(z1pt0_len),
        };
        assert!(inputs(1).iter().is_err());
        assert_eq!(inputs(2).iter().unwrap().count(), 2);
    }
}
