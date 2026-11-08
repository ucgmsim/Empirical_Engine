//! Campbell and Bozorgnia (2014) NGA-West2 GMPE ("CB_14" in `oq_wrapper`).
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.campbell_bozorgnia_2014`
//! (`CampbellBozorgnia2014`), specialised to the configuration `oq_wrapper`
//! instantiates it with for ACTIVE_SHALLOW and VOLCANIC: `estimate_width=True`
//! (rupture width from Eq. 39, using `zbot`) and otherwise default arguments.
//! That means `SJ=False` (California basin model, no Japan site terms), the
//! default (central) `Dc20` anelastic coefficients, no USGS or CyberShake basin
//! adjustments, and `ztor` and `hypo_depth` taken as given. Only the PGA, PGV
//! and SA coefficients are used (OpenQuake's table also holds CAV and IA, which
//! the class does not declare as supported).
//!
//! Equation numbers ("Eq. N") refer to Campbell and Bozorgnia (2014), Earthquake
//! Spectra 30(3), 1087-1115. Epistemic branches use OpenQuake's
//! `sigma_mu_epsilon`, which adds `sigma_mu_epsilon * get_epistemic_sigma` to
//! the ln mean (see [`nga_west2_epistemic_sigma`]).
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/cb14_coefficients.rs"));
}

use crate::error::GMMError;
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::utils::{check_lengths, nga_west2_epistemic_sigma};
use coeffs::{Coeffs, PGA_COEFFS};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Constants `c` and `n` in Eq. 17 to 19 and 31, and `h4` in Eq. 10.
const C: f64 = 1.88;
const N: f64 = 1.18;
const H4: f64 = 1.0;
/// Vs30 of the reference rock used for PGA1100.
const VS30_REF: f64 = 1100.0;

/// CB14 input columns, one value per rupture-site pair. Angles are in degrees,
/// distances and depths in km. `zbot` is only used to estimate the rupture
/// width (Eq. 39). Exposed to Python as `gm_models._core.CB14Inputs`. Column
/// lengths are checked in [`CB14Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub struct CB14Inputs {
    mag: Vec<f64>,
    rake: Vec<f64>,
    dip: Vec<f64>,
    ztor: Vec<f64>,
    zbot: Vec<f64>,
    hypo_depth: Vec<f64>,
    rrup: Vec<f64>,
    rjb: Vec<f64>,
    rx: Vec<f64>,
    vs30: Vec<f64>,
    z2pt5: Vec<f64>,
}

/// A single row of [`CB14Inputs`], with the rupture width (km) already known.
#[derive(Debug, Clone, Copy)]
pub struct CB14Row {
    pub mag: f64,
    pub rake: f64,
    pub dip: f64,
    pub ztor: f64,
    pub width: f64,
    pub hypo_depth: f64,
    pub rrup: f64,
    pub rjb: f64,
    pub rx: f64,
    pub vs30: f64,
    /// Depth to Vs = 2.5 km/s (km). -999 means unknown (OpenQuake convention),
    /// NaN switches the basin term off (as OpenQuake does).
    pub z2pt5: f64,
}

impl CB14Inputs {
    /// The rupture-site pairs, in order, with the width estimated from Eq. 39 as
    /// OpenQuake's `_update_ctx` does for `estimate_width=True`. Fails if the
    /// columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = CB14Row> + '_, GMMError> {
        let n = check_lengths(&[
            ("mag", self.mag.len()),
            ("rake", self.rake.len()),
            ("dip", self.dip.len()),
            ("ztor", self.ztor.len()),
            ("zbot", self.zbot.len()),
            ("hypo_depth", self.hypo_depth.len()),
            ("rrup", self.rrup.len()),
            ("rjb", self.rjb.len()),
            ("rx", self.rx.len()),
            ("vs30", self.vs30.len()),
            ("z2pt5", self.z2pt5.len()),
        ])?;
        Ok((0..n).map(move |i| {
            let (mag, dip) = (self.mag[i], self.dip[i]);
            let width = 10f64.powf((mag - 4.07) / 0.98).sqrt();
            // OpenQuake only applies the zbot cap where sin(dip) != 0.
            let sin_dip = dip.to_radians().sin();
            let width = if sin_dip.abs() > 0.0 {
                width.min((self.zbot[i] - self.ztor[i]) / sin_dip)
            } else {
                width
            };
            CB14Row {
                mag,
                rake: self.rake[i],
                dip,
                ztor: self.ztor[i],
                width,
                hypo_depth: self.hypo_depth[i],
                rrup: self.rrup[i],
                rjb: self.rjb[i],
                rx: self.rx[i],
                vs30: self.vs30[i],
                z2pt5: self.z2pt5[i],
            }
        }))
    }
}

#[pymethods]
impl CB14Inputs {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mag: PyReadonlyArray1<f64>,
        rake: PyReadonlyArray1<f64>,
        dip: PyReadonlyArray1<f64>,
        ztor: PyReadonlyArray1<f64>,
        zbot: PyReadonlyArray1<f64>,
        hypo_depth: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        rjb: PyReadonlyArray1<f64>,
        rx: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        z2pt5: PyReadonlyArray1<f64>,
    ) -> Self {
        let v = |a: PyReadonlyArray1<f64>| a.as_array().to_vec();
        Self {
            mag: v(mag),
            rake: v(rake),
            dip: v(dip),
            ztor: v(ztor),
            zbot: v(zbot),
            hypo_depth: v(hypo_depth),
            rrup: v(rrup),
            rjb: v(rjb),
            rx: v(rx),
            vs30: v(vs30),
            z2pt5: v(z2pt5),
        }
    }
}

/// Reference z2pt5 (km) for a given Vs30, California basin model, Eq. 33.
fn z2pt5_ref(vs30: f64) -> f64 {
    (7.089 - 1.144 * vs30.ln()).exp()
}

/// Basin response term, Eq. 20.
fn basin_term(c: &Coeffs, z2pt5: f64) -> f64 {
    if z2pt5 < 1.0 {
        c.c14 * (z2pt5 - 1.0)
    } else if z2pt5 > 3.0 {
        c.c16 * c.k3 * (-0.75f64).exp() * (1.0 - (-0.25 * (z2pt5 - 3.0)).exp())
    } else {
        0.0
    }
}

/// Magnitude term, Eq. 2.
fn magnitude_term(c: &Coeffs, mag: f64) -> f64 {
    let mut f = c.c0 + c.c1 * mag;
    if mag > 4.5 {
        f += c.c2 * (mag - 4.5);
    }
    if mag > 5.5 {
        f += c.c3 * (mag - 5.5);
    }
    if mag > 6.5 {
        f += c.c4 * (mag - 6.5);
    }
    f
}

/// Style-of-faulting term, Eqs. 4 to 6.
fn style_of_faulting_term(c: &Coeffs, row: &CB14Row) -> f64 {
    let frv = f64::from(row.rake > 30.0 && row.rake < 150.0);
    let fnm = f64::from(row.rake > -150.0 && row.rake < -30.0);
    let fflt_m = (row.mag - 4.5).clamp(0.0, 1.0);
    (c.c8 * frv + c.c9 * fnm) * fflt_m
}

/// Hanging wall term, Eqs. 7 to 16.
fn hanging_wall_term(c: &Coeffs, row: &CB14Row) -> f64 {
    let CB14Row {
        mag,
        dip,
        rrup,
        rjb,
        rx,
        ..
    } = *row;
    // Eqs. 7 to 12 (R1, R2, f1, f2).
    let r1 = row.width * dip.to_radians().cos();
    let r2 = 62.0 * mag - 350.0;
    let f_rx = if rx >= r1 {
        let d = (rx - r1) / (r2 - r1);
        (H4 + c.h5 * d + c.h6 * d * d).max(0.0)
    } else if rx >= 0.0 {
        let x = rx / r1;
        c.h1 + c.h2 * x + c.h3 * x * x
    } else {
        0.0
    };
    // Eq. 13
    let f_rrup = if rrup > 0.0 { (rrup - rjb) / rrup } else { 1.0 };
    // Eq. 14
    let f_mag = if mag < 5.5 {
        0.0
    } else if mag > 6.5 {
        1.0 + c.a2 * (mag - 6.5)
    } else {
        (mag - 5.5) * (1.0 + c.a2 * (mag - 6.5))
    };
    // Eq. 15
    let f_ztor = if row.ztor > 16.66 {
        0.0
    } else {
        1.0 - 0.06 * row.ztor
    };
    // Eq. 16
    let f_dip = (90.0 - dip) / 45.0;
    c.c10 * f_rx * f_rrup * f_mag * f_ztor * f_dip
}

/// Hypocentral depth term, Eqs. 21 to 23.
fn hypocentral_depth_term(c: &Coeffs, row: &CB14Row) -> f64 {
    let f_h = (row.hypo_depth - 7.0).clamp(0.0, 13.0);
    let f_m = if row.mag <= 5.5 {
        c.c17
    } else if row.mag > 6.5 {
        c.c18
    } else {
        c.c17 + (c.c18 - c.c17) * (row.mag - 5.5)
    };
    f_h * f_m
}

/// Fault dip term, Eq. 24.
fn fault_dip_term(c: &Coeffs, row: &CB14Row) -> f64 {
    if row.mag < 4.5 {
        c.c19 * row.dip
    } else if row.mag > 5.5 {
        0.0
    } else {
        c.c19 * (5.5 - row.mag) * row.dip
    }
}

/// Shallow site response term, Eqs. 17 to 19 (global model, `SJ = False`).
/// `pga_rock` is PGA1100 (only used for `vs30 <= k1`).
fn site_term(c: &Coeffs, vs30: f64, pga_rock: f64) -> f64 {
    let vs_mod = vs30 / c.k1;
    let linear = c.c11 * vs_mod.ln();
    if vs30 > c.k1 {
        linear + c.k2 * N * vs_mod.ln()
    } else {
        linear + c.k2 * ((pga_rock + C * vs_mod.powf(N)).ln() - (pga_rock + C).ln())
    }
}

/// ln mean of one IM with coefficients `c`, summing the terms of OpenQuake's
/// `get_mean_values`. `vs30`/`z2pt5` are the site values, or the 1100 m/s
/// reference rock values when computing PGA1100. `pga_rock` is PGA1100.
fn mean_values(c: &Coeffs, row: &CB14Row, vs30: f64, z2pt5: f64, pga_rock: f64) -> f64 {
    let (mag, rrup) = (row.mag, row.rrup);
    magnitude_term(c, mag)
        // Eq. 3
        + (c.c5 + c.c6 * mag) * rrup.hypot(c.c7).ln()
        + style_of_faulting_term(c, row)
        + hanging_wall_term(c, row)
        + site_term(c, vs30, pga_rock)
        + basin_term(c, z2pt5)
        + hypocentral_depth_term(c, row)
        + fault_dip_term(c, row)
        // Anelastic attenuation, Eq. 25
        + if rrup >= 80.0 { (c.c20 + c.dc20) * (rrup - 80.0) } else { 0.0 }
}

/// Inter-event coefficient tau, Eq. 27.
fn tau_lny(c: &Coeffs, mag: f64) -> f64 {
    c.tau2 + (c.tau1 - c.tau2) * (5.5 - mag.clamp(4.5, 5.5))
}

/// Intra-event coefficient phi, Eq. 28.
fn phi_lny(c: &Coeffs, mag: f64) -> f64 {
    c.phi2 + (c.phi1 - c.phi2) * (5.5 - mag.clamp(4.5, 5.5))
}

/// Mean + standard deviations for OpenQuake's `CampbellBozorgnia2014(estimate_width=True)`,
/// one value per rupture-site pair in `inputs`. SA periods outside the table
/// are an error. Exposed to Python as `gm_models._core.cb14`.
#[pyfunction]
pub fn cb14(
    py: Python<'_>,
    inputs: PyRef<'_, CB14Inputs>,
    im: IM,
    sigma_mu_epsilon: f64,
) -> PyResult<GMMResult> {
    let coeffs = Coeffs::for_im(im)?;
    let inputs = &*inputs;
    // Releases the GIL: `inputs` owns copies of the arrays.
    Ok(py.detach(|| -> Result<GMMResult, GMMError> {
        Ok(inputs
            .iter()?
            .map(|row| compute(im, &coeffs, &row, sigma_mu_epsilon))
            .collect())
    })?)
}

/// Computes mean + standard deviations for a single site/rupture pair.
///
/// `c` must be `Coeffs::for_im(im)`; it is passed in so callers looping over
/// sites look it up once rather than per site.
pub fn compute(im: IM, c: &Coeffs, row: &CB14Row, sigma_mu_epsilon: f64) -> MeanStd {
    let c_pga = &PGA_COEFFS;
    let z_ref = z2pt5_ref(VS30_REF);
    // OpenQuake's -999 sentinel: estimate z2pt5 from Vs30 (Eq. 33).
    let z2pt5 = if row.z2pt5 == -999.0 {
        z2pt5_ref(row.vs30)
    } else {
        row.z2pt5
    };

    // PGA on reference rock (Vs30 = 1100 m/s, reference basin depth).
    let pga1100 = mean_values(c_pga, row, VS30_REF, z_ref, f64::NAN).exp();

    let mut mean = mean_values(c, row, row.vs30, z2pt5, pga1100);
    if matches!(im, IM::SA { period } if period < 0.25) {
        // Campbell and Bozorgnia (2013) NGA-West2 report: SA(T < 0.25 s) is not
        // allowed to fall below PGA at the site.
        let pga = mean_values(c_pga, row, row.vs30, z2pt5, pga1100);
        if mean <= pga {
            mean = pga;
        }
    }
    mean += sigma_mu_epsilon * nga_west2_epistemic_sigma(row.mag, row.rrup);

    // Linearised site amplification derivative, Eq. 31.
    let alpha = if row.vs30 < c.k1 {
        let vsk1 = row.vs30 / c.k1;
        c.k2 * pga1100 * (1.0 / (pga1100 + C * vsk1.powf(N)) - 1.0 / (pga1100 + C))
    } else {
        0.0
    };
    let tau_b = tau_lny(c, row.mag);
    let tau_pga_b = tau_lny(c_pga, row.mag);
    let phi_b = (phi_lny(c, row.mag).powi(2) - c.philnaf.powi(2)).sqrt();
    let phi_pga_b = (phi_lny(c_pga, row.mag).powi(2) - c_pga.philnaf.powi(2)).sqrt();
    // Eqs. 29 and 30
    let tau = (tau_b.powi(2)
        + alpha.powi(2) * tau_pga_b.powi(2)
        + 2.0 * alpha * c.rholny * tau_b * tau_pga_b)
        .sqrt();
    let phi = (phi_b.powi(2)
        + c.philnaf.powi(2)
        + alpha.powi(2) * phi_pga_b.powi(2)
        + 2.0 * alpha * c.rholny * phi_b * phi_pga_b)
        .sqrt();

    MeanStd {
        mean,
        total: tau.hypot(phi),
        inter: tau,
        intra: phi,
    }
}

/// Verification against OpenQuake's reference tables (generated from the
/// supplementary Excel file of the paper), read from the `external/oq-engine`
/// submodule. These CSVs give the rupture width directly (OpenQuake's test uses
/// `estimate_width=False`), so they exercise everything but Eq. 39.
/// `#[ignore]`d so a plain `cargo test` works without the submodule.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> CB14Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        CB14Row {
            mag: num("rup_mag"),
            rake: num("rup_rake"),
            dip: num("rup_dip"),
            ztor: num("rup_ztor"),
            width: num("rup_width"),
            hypo_depth: num("rup_hypo_depth"),
            rrup: num("dist_rrup"),
            rjb: num("dist_rjb"),
            rx: num("dist_rx"),
            vs30: num("site_vs30"),
            z2pt5: num("site_z2pt5"),
        }
    }

    fn check(file: &str, pick: fn(&MeanStd) -> f64) {
        check_csv(&format!("CB14/{file}"), row, |im, row| {
            pick(&compute(im, &Coeffs::for_im(im).unwrap(), row, 0.0))
        });
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake() {
        check("CB2014_MEAN.csv", |r| r.mean.exp());
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn stddevs_match_openquake() {
        check("CB2014_STD_TOTAL.csv", |r| r.total);
        check("CB2014_STD_INTER.csv", |r| r.inter);
        check("CB2014_STD_INTRA.csv", |r| r.intra);
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

    fn row() -> impl Strategy<Value = CB14Row> {
        (
            (
                4.0..8.5f64,
                -180.0..180.0f64,
                10.0..90.0f64,
                0.0..20.0f64,
                1.0..40.0f64,
            ),
            (
                0.0..40.0f64,
                0.1..300.0f64,
                -100.0..100.0f64,
                150.0..1500.0f64,
                0.0..8.0f64,
            ),
        )
            .prop_map(
                |((mag, rake, dip, ztor, width), (hypo_depth, rrup, rx, vs30, z2pt5))| CB14Row {
                    mag,
                    rake,
                    dip,
                    ztor,
                    width,
                    hypo_depth,
                    rrup,
                    rjb: rrup * 0.9,
                    rx,
                    vs30,
                    z2pt5,
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
    fn inputs_estimate_width_and_reject_mismatched_lengths() {
        let make = |zbot: Vec<f64>| CB14Inputs {
            mag: vec![7.0],
            rake: vec![90.0],
            dip: vec![45.0],
            ztor: vec![2.0],
            zbot,
            hypo_depth: vec![10.0],
            rrup: vec![10.0],
            rjb: vec![10.0],
            rx: vec![10.0],
            vs30: vec![400.0],
            z2pt5: vec![1.0],
        };
        assert!(make(vec![15.0, 15.0]).iter().is_err());
        // Eq. 39: min(sqrt(10^((M - 4.07) / 0.98)), (zbot - ztor) / sin(dip)).
        let uncapped = 10f64.powf((7.0 - 4.07) / 0.98).sqrt();
        assert_eq!(
            make(vec![1000.0]).iter().unwrap().next().unwrap().width,
            uncapped
        );
        let capped = make(vec![5.0]).iter().unwrap().next().unwrap().width;
        assert!((capped - 3.0 / 45f64.to_radians().sin()).abs() < 1e-12);
    }
}
