//! Stafford (2022) NZ NSHM backbone GMM ("S_22" in `oq_wrapper`), for active
//! shallow crust (also used for volcanic).
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.nz22.stafford_2022`
//! (`Stafford2022`), which modifies Chiou and Youngs (2014) (functions from
//! `openquake.hazardlib.gsim.chiou_youngs_2014`), specialised to the
//! configuration `oq_wrapper` uses: default constructor arguments except for
//! the epistemic mean branch (`mu_branch`, exposed as [`Branch`]). That means
//! `sigma_branch="Central"` (the other two sigma branches are dropped) and all
//! four `adjust_*` flags on. Also omitted, since `oq_wrapper` never provides
//! them: the directivity term (OpenQuake skips it without a `rcdpp` context
//! attribute) and the `z1pt0 == -999` "use the Vs30-based default" sentinel
//! (`oq_wrapper` always passes z1pt0 in metres from km). The result is in g;
//! PGV is not defined by the model, PGA uses the SA(0.01 s) coefficients and
//! T = 0.01 s throughout, as OpenQuake does.
//!
//! Equation numbers are not given: OpenQuake cites Stafford's GNS report
//! ("Eq. 8.16" for the standard deviation model) and CY14 (Chiou and Youngs,
//! 2014, Earthquake Spectra 30(3)), Eqs. 11 to 13 for the mean.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/s22_coefficients.rs"));
}

use crate::branch::Branch;
use crate::error::GMMError;
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::utils::check_lengths;
use coeffs::Coeffs;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// CY14 constants c2, c4, c4a, crb (the table's own columns equal them).
const C2: f64 = 1.06;
const C4: f64 = -2.1;
const C4A: f64 = -0.5;
const CRB: f64 = 50.0;
/// phi6 in the basin term, and the reference Vs30 of the site terms.
const PHI6: f64 = 300.0;
const REF_VS: f64 = 1130.0;
/// 10th/90th percentile z-score and the 0.893 factor accounting for the
/// reduction in epistemic uncertainty when rupture scenarios are not perfectly
/// correlated (see the OpenQuake source comments).
const EPISTEMIC_Z: f64 = 1.28155 * 0.893;
/// Correlation between the empirical and saturation epistemic terms.
const RHO_EH_EX: f64 = 0.4;

/// S22 input columns, one value per rupture-site pair. `z1pt0` is in metres.
/// Exposed to Python as `gm_models._core.S22Inputs`. Column lengths are checked
/// in [`S22Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub struct S22Inputs {
    mag: Vec<f64>,
    rake: Vec<f64>,
    dip: Vec<f64>,
    ztor: Vec<f64>,
    rrup: Vec<f64>,
    rjb: Vec<f64>,
    rx: Vec<f64>,
    vs30: Vec<f64>,
    z1pt0: Vec<f64>,
}

/// A single row of [`S22Inputs`].
#[derive(Debug, Clone, Copy)]
pub struct S22Row {
    pub mag: f64,
    pub rake: f64,
    pub dip: f64,
    pub ztor: f64,
    pub rrup: f64,
    pub rjb: f64,
    pub rx: f64,
    pub vs30: f64,
    pub z1pt0: f64,
}

impl S22Inputs {
    /// The rupture-site pairs, in order. Fails if the columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = S22Row> + '_, GMMError> {
        let n = check_lengths(&[
            ("mag", self.mag.len()),
            ("rake", self.rake.len()),
            ("dip", self.dip.len()),
            ("ztor", self.ztor.len()),
            ("rrup", self.rrup.len()),
            ("rjb", self.rjb.len()),
            ("rx", self.rx.len()),
            ("vs30", self.vs30.len()),
            ("z1pt0", self.z1pt0.len()),
        ])?;
        Ok((0..n).map(move |i| S22Row {
            mag: self.mag[i],
            rake: self.rake[i],
            dip: self.dip[i],
            ztor: self.ztor[i],
            rrup: self.rrup[i],
            rjb: self.rjb[i],
            rx: self.rx[i],
            vs30: self.vs30[i],
            z1pt0: self.z1pt0[i],
        }))
    }
}

#[pymethods]
impl S22Inputs {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mag: PyReadonlyArray1<f64>,
        rake: PyReadonlyArray1<f64>,
        dip: PyReadonlyArray1<f64>,
        ztor: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        rjb: PyReadonlyArray1<f64>,
        rx: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        z1pt0: PyReadonlyArray1<f64>,
    ) -> Self {
        Self {
            mag: mag.as_array().to_vec(),
            rake: rake.as_array().to_vec(),
            dip: dip.as_array().to_vec(),
            ztor: ztor.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            rjb: rjb.as_array().to_vec(),
            rx: rx.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
            z1pt0: z1pt0.as_array().to_vec(),
        }
    }
}

/// Sigmoid from `start` to `finish`, centred at `centre` (OpenQuake's `_sigmoid1d`).
fn sigmoid(x: f64, start: f64, finish: f64, centre: f64, slope: f64) -> f64 {
    start + (finish - start) / (1.0 + (-(x - centre) / slope).exp())
}

/// Epistemic standard deviation of the empirical constraint (`_empirical_sigma`
/// with its default start = 0.4 and finish = 0.15).
fn empirical_sigma(t: f64, r: f64, m: f64) -> f64 {
    let s_lnr = sigmoid(
        r.max(1e-10).ln(),
        0.0,
        1.0,
        sigmoid(t, 3.75, 5.75, 3.5, 0.5),
        sigmoid(t, 0.225, 1.3, 3.5, 0.5),
    );
    let s_m = sigmoid(
        m,
        1.0,
        0.0,
        sigmoid(t, 5.82, 6.4, 3.5, 0.5),
        sigmoid(t, 0.42, 0.38, 2.5, 0.2),
    );
    0.4 + (0.15 - 0.4) * s_lnr * s_m
}

/// Standard deviation of the near-source saturation model.
fn saturation_sigma(m: f64, r: f64) -> f64 {
    sigmoid(
        r.max(1e-10).ln(),
        0.6638 - 0.2570 * (m.min(7.0) - 6.0),
        0.0,
        0.7990,
        0.9835,
    )
}

/// Mean anelastic attenuation correction (`_anelastic_correction`).
fn anelastic_correction(t: f64) -> f64 {
    (0.0024 * (0.8 * (t / 1.8).ln()).tanh()).min(0.0)
}

/// Mean-branch adjustments (delta_c1 + delta_c1hm, delta_c7, delta_c7b,
/// delta_cg1) of OpenQuake's `get_adjustments`, with all `adjust_*` flags on.
fn adjustments(branch: Branch, t: f64, rrup: f64, mag: f64) -> (f64, f64, f64, f64) {
    let ln_t = t.ln();
    let anelastic_sigma = sigmoid(ln_t, 0.00054, 0.0004, 0.9f64.ln(), 0.2);
    let dc1 = EPISTEMIC_Z * empirical_sigma(t, rrup, mag);
    let dc1hm = EPISTEMIC_Z * RHO_EH_EX * saturation_sigma(mag, rrup);
    match branch {
        Branch::Lower => (
            -(dc1 + dc1hm),
            -0.02578,
            sigmoid(ln_t, -0.05737, 0.05733, -1.0324, 0.04875),
            anelastic_correction(t) - EPISTEMIC_Z * anelastic_sigma,
        ),
        Branch::Central => (
            0.0,
            0.0,
            sigmoid(ln_t, -0.0865, 0.0, -1.5364, 0.3266),
            anelastic_correction(t),
        ),
        Branch::Upper => (
            dc1 + dc1hm,
            0.0,
            0.0,
            anelastic_correction(t) + EPISTEMIC_Z * anelastic_sigma,
        ),
    }
}

/// Between-event standard deviation, a function of period and magnitude.
fn between_event(t: f64, m: f64) -> f64 {
    let ln_t = t.ln();
    let tau0 = sigmoid(ln_t, 0.45, 0.4, 0.0, 0.2);
    let tau1 = sigmoid(ln_t, 0.35, 0.4, 0.0, 0.2);
    tau0 + (tau1 - tau0) * (m.clamp(5.0, 6.5) - 5.0) / 1.5
}

/// Between-station standard deviation, a function of period.
fn between_station(t: f64) -> f64 {
    0.3171 + 0.3248 * (t / 0.7170).powf(0.7911) * (-0.8970 * (t / 0.7170)).exp()
}

/// Effective number of observations in the NZ data (`_neff_model`).
fn neff(t: f64) -> f64 {
    if t <= 0.3 {
        sigmoid(t.ln(), 190.34, 221.16, -2.4341, 0.10366)
    } else {
        221.16 / (t / 0.3).powf(0.10299)
    }
}

/// ln Gamma(z) for z >= 0.5 (Lanczos, g = 7, n = 9; about 1e-15 relative).
fn ln_gamma(z: f64) -> f64 {
    const LANCZOS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = z - 1.0;
    let series = LANCZOS[0]
        + LANCZOS[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| c / (z + i as f64 + 1.0))
            .sum::<f64>();
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// Regularised lower incomplete gamma function P(a, x), by its series for
/// x < a + 1 and the continued fraction (modified Lentz) otherwise.
fn gamma_p(a: f64, x: f64) -> f64 {
    let prefactor = (-x + a * x.ln() - ln_gamma(a)).exp();
    if x < a + 1.0 {
        let (mut term, mut sum, mut ap) = (1.0 / a, 1.0 / a, a);
        while term.abs() > sum.abs() * 1e-16 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
        }
        sum * prefactor
    } else {
        let tiny = 1e-300;
        let (mut b, mut c, mut d) = (x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a));
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - prefactor * h
    }
}

/// Median of the chi-squared distribution with `k` degrees of freedom
/// (`scipy.stats.chi2.ppf(0.5, k)`): Newton's method on [`gamma_p`], started
/// from the Wilson-Hilferty approximation.
fn chi2_median(k: f64) -> f64 {
    let a = k / 2.0;
    let mut y = a * (1.0 - 1.0 / (9.0 * a)).powi(3);
    for _ in 0..100 {
        let step = (gamma_p(a, y) - 0.5) / ((-y + (a - 1.0) * y.ln() - ln_gamma(a)).exp());
        y -= step;
        if step.abs() < 1e-15 * y {
            break;
        }
    }
    2.0 * y
}

/// Period used for `im` and its coefficients: PGA uses the SA(0.01 s) row and
/// T = 0.01 s, as OpenQuake does. PGV and other IMs are not defined.
fn period_and_coeffs(im: IM) -> Result<(f64, Coeffs), GMMError> {
    match im {
        IM::PGA {} => Ok((0.01, Coeffs::for_sa_period(0.01)?)),
        IM::SA { period } => Ok((period, Coeffs::for_sa_period(period)?)),
        _ => Err(GMMError::UnsupportedIM(im)),
    }
}

/// Mean + standard deviations of `Stafford2022` (`sigma_branch="Central"`),
/// one value per rupture-site pair in `inputs`. SA periods outside the table
/// are an error. Exposed to Python as `gm_models._core.s22`.
#[pyfunction]
pub fn s22(
    py: Python<'_>,
    inputs: PyRef<'_, S22Inputs>,
    im: IM,
    branch: Branch,
) -> PyResult<GMMResult> {
    let (t, coeffs) = period_and_coeffs(im)?;
    let inputs = &*inputs;
    // Releases the GIL: `inputs` owns copies of the arrays.
    Ok(py.detach(|| {
        inputs
            .iter()?
            .map(|row| Ok(compute(t, &coeffs, &row, branch)))
            .collect::<Result<GMMResult, GMMError>>()
    })?)
}

/// Computes mean + standard deviations for a single site/rupture pair at
/// period `t` (from [`period_and_coeffs`], with its coefficients `c`).
pub fn compute(t: f64, c: &Coeffs, row: &S22Row, branch: Branch) -> MeanStd {
    let S22Row {
        mag,
        rake,
        dip,
        ztor,
        rrup,
        rjb,
        rx,
        vs30,
        z1pt0,
    } = *row;
    let (dc1, dc7, dc7b, dcg1) = adjustments(branch, t, rrup, mag);

    // Reference-rock motion, CY14 Eq. 11 with the NZ adjustments. ztor is
    // centred on the magnitude-dependent mean (reverse or other faulting).
    let mean_ztor = if (30.0..=150.0).contains(&rake) {
        (2.704 - 1.226 * (mag - 5.849).max(0.0)).max(0.0).powi(2)
    } else {
        (2.673 - 1.136 * (mag - 4.970).max(0.0)).max(0.0).powi(2)
    };
    let coshm = (2.0 * (mag - 4.5).max(0.0)).cosh();
    let style = if (30.0..=150.0).contains(&rake) {
        c.c1a + c.c1c / coshm
    } else if (-120.0..=-60.0).contains(&rake) {
        c.c1b + c.c1d / coshm
    } else {
        0.0
    };
    let f_src = style
        + ((c.c7 + dc7) + (c.c7b + dc7b) / coshm) * (ztor - mean_ztor)
        + (c.c11b / coshm) * dip.to_radians().cos().powi(2);
    let f_hw = if rx >= 0.0 {
        let fdist =
            (1.0 - rjb.hypot(ztor) / (rrup + 1.0)) * (c.c9a + (1.0 - c.c9a) * (rx / c.c9b).tanh());
        c.c9 * dip.to_radians().cos() * fdist
    } else {
        0.0
    };
    let ln_y_ref = c.c1
        + dc1
        + C2 * (mag - 6.0)
        + ((C2 - c.c3) / c.cn) * (1.0 + (c.cn * (c.cm - mag)).exp()).ln()
        + f_src
        + f_hw
        + C4 * (rrup + c.c5 * (c.c6 * (mag - c.chm).max(0.0)).cosh()).ln()
        + (C4A - C4) * rrup.hypot(CRB).ln()
        + (c.cg1 + dcg1 + c.cg2 / (mag - c.cg3).max(0.0).cosh()) * rrup;

    // Site terms: basin depth (z1pt0 in m, centred on the global Vs30
    // relation; no deep-soil correction for z1pt0 <= 0), linear and nonlinear.
    let mean_z1pt0 = (-7.15 / 4.0)
        * ((vs30.powi(4) + 570.94f64.powi(4)) / (1360f64.powi(4) + 570.94f64.powi(4))).ln();
    let dz1pt0 = if z1pt0 <= 0.0 {
        0.0
    } else {
        z1pt0 - mean_z1pt0.exp()
    };
    let f_z1pt0 = c.phi5 * (1.0 - (-dz1pt0 / PHI6).exp());
    let f_lin = c.phi1 * (vs30 / REF_VS).ln().min(0.0);
    let vs = vs30.min(REF_VS);
    let f_nl_scaling = c.phi2 * ((c.phi3 * (vs - 360.0)).exp() - (c.phi3 * (REF_VS - 360.0)).exp());
    let y_ref = ln_y_ref.exp();
    let f_nl = ((y_ref + c.phi4) / c.phi4).ln() * f_nl_scaling;

    // Standard deviations (the report's Eq. 8.16): nonlinear site variance
    // inflation, then deflation to the median of the chi-squared sampling
    // distribution of the variance (sigma_branch = "Central"). The total is
    // therefore slightly below sqrt(inter^2 + intra^2).
    let tau = between_event(t, mag);
    let phi_ss: f64 = 0.39;
    let nl = f_nl_scaling * (y_ref / (y_ref + c.phi4));
    let phi2 = between_station(t).powi(2)
        + (1.0 + nl).powi(2) * 0.736 * phi_ss.powi(2)
        + 0.264 * phi_ss.powi(2);
    let sigma2 = (1.0 + nl).powi(2) * tau.powi(2) + phi2;
    let nfac = (empirical_sigma(t, rrup, mag) / empirical_sigma(t, 80.0, 5.0)).powi(2);
    let dof = neff(t) / nfac - 1.0;

    MeanStd {
        mean: ln_y_ref + f_lin + f_nl + f_z1pt0,
        total: (sigma2 / dof * chi2_median(dof)).sqrt(),
        inter: ((1.0 + nl) * tau).abs(),
        intra: phi2.sqrt(),
    }
}

/// Layer 1 verification against OpenQuake's reference CSVs (GNS-produced),
/// read from the `external/oq-engine` submodule. `#[ignore]`d so a plain
/// `cargo test` works without the submodule; run with `cargo test -- --ignored`.
/// Mean checks use OpenQuake's `EPI_SCALED` lower/upper files, as its own tests
/// do; only `sigma_branch = "Central"` is covered (the other sigma branches
/// are not implemented). The CSVs have no inter/intra values.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> S22Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        S22Row {
            mag: num("rup_mag"),
            rake: num("rup_rake"),
            dip: num("rup_dip"),
            ztor: num("rup_ztor"),
            rrup: num("dist_rrup"),
            rjb: num("dist_rjb"),
            rx: num("dist_rx"),
            vs30: num("site_vs30"),
            z1pt0: num("site_z1pt0"),
        }
    }

    fn check(file: &str, branch: Branch, pick: fn(&MeanStd) -> f64) {
        check_csv(&format!("nz22/STAFFORD2022/{file}"), row, |im, row| {
            let (t, c) = period_and_coeffs(im).unwrap();
            pick(&compute(t, &c, row, branch))
        });
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake() {
        for style in ["SS", "REV", "NM"] {
            for (branch, file) in [
                (
                    Branch::Central,
                    if style == "REV" {
                        "MEAN_CENTERAL_v2"
                    } else {
                        "MEAN_CENTRAL_v2"
                    },
                ),
                (Branch::Lower, "MEAN_LOWER_v2_EPI_SCALED"),
                (Branch::Upper, "MEAN_UPPER_v2_EPI_SCALED"),
            ] {
                check(&format!("STAFFORD_NZ_{style}_{file}.csv"), branch, |r| {
                    r.mean.exp()
                });
            }
        }
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn total_stddev_matches_openquake() {
        for style in ["SS", "REV", "NM"] {
            check(
                &format!("STAFFORD_NZ_{style}_SIGMA_CENTRAL_v2.csv"),
                Branch::Central,
                |r| r.total,
            );
        }
    }

    /// Values from `scipy.stats.chi2.ppf(0.5, k)`.
    #[test]
    fn chi2_median_matches_scipy() {
        for (k, expected) in [
            (2.5, 1.8738477677808796),
            (40.0, 39.33534484661134),
            (100.5, 99.83412524761707),
            (150.0, 149.3338626493889),
            (300.0, 299.33359734921805),
        ] {
            approx::assert_relative_eq!(chi2_median(k), expected, max_relative = 1e-12);
        }
    }
}

/// Property tests over the supported input domain. These need no external
/// data, so they run in a plain `cargo test`.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn row() -> impl Strategy<Value = S22Row> {
        (
            (5.0..9.0f64, -180.0..180.0f64, 20.0..90.0f64, 0.0..20.0f64),
            (
                0.1..400.0f64,
                -300.0..300.0f64,
                100.0..1500.0f64,
                0.0..1.0f64,
            ),
        )
            .prop_map(
                |((mag, rake, dip, ztor), (rrup, rx, vs30, z1pt0_km))| S22Row {
                    mag,
                    rake,
                    dip,
                    ztor,
                    rrup,
                    rjb: rrup * 0.9,
                    rx,
                    vs30,
                    z1pt0: z1pt0_km * 1000.0,
                },
            )
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_total_is_deflated(
            t in 0.01..=10.0f64,
            row in row(),
            branch in prop_oneof![Just(Branch::Lower), Just(Branch::Central), Just(Branch::Upper)],
        ) {
            let c = Coeffs::for_sa_period(t).unwrap();
            let r = compute(t, &c, &row, branch);
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0 && r.total > 0.0);
            prop_assert!(r.total.powi(2) <= (r.inter.powi(2) + r.intra.powi(2)) * (1.0 + 1e-12));
        }
    }

    #[test]
    fn pgv_is_rejected() {
        assert!(period_and_coeffs(IM::PGV {}).is_err());
    }
}
