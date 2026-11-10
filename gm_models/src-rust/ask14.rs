//! Abrahamson, Silva and Kamai (2014) NGA-West2 GMPE ("ASK_14" in `oq_wrapper`).
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.abrahamson_2014.AbrahamsonEtAl2014`,
//! specialised to the configuration `oq_wrapper` instantiates it with for both
//! `ACTIVE_SHALLOW` and `VOLCANIC`: default California model (`region=None`,
//! `usgs_basin_scaling=False`, `cybershake_basin_adj=False`), with the epistemic
//! branches applied through `sigma_mu_epsilon`. The regional terms (CHN, JPN,
//! TWN), the USGS/CyberShake basin options and the JPN standard deviation model
//! are therefore omitted.
//!
//! The aftershock term (Eq. 20, `f11`) is also omitted: `oq_wrapper` fills
//! `is_aftershock=False` (and `crjb=NaN`) whenever the columns are missing, so
//! every rupture is a mainshock and the term is 0. `gm_models.ask14.run_ask14`
//! rejects dataframes that flag aftershocks.
//!
//! Equation numbers ("Eq. N") refer to Abrahamson, Silva and Kamai (2014),
//! Earthquake Spectra 30(3), 1025-1055.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/ask14_coefficients.rs"));
}

use crate::error::GMMError;
use crate::gmm::{evaluate, GMM};
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::row_inputs;
use crate::utils::nga_west2_epistemic_sigma;
use coeffs::Coeffs;
use pyo3::prelude::*;

/// Constants `n`, `m2` (p. 1032), `h1`, `h2`, `h3` (p. 1040).
const N: f64 = 1.5;
const M2: f64 = 5.0;
const H1: f64 = 0.25;
const H2: f64 = 1.5;
const H3: f64 = -0.75;
/// `a2hw` in Eq. 12 (p. 1041).
const A2HW: f64 = 0.2;
/// Reference rock Vs30 (m/s).
const VS30_REF: f64 = 1180.0;
/// Knots (Vs30, m/s) of the linear interpolation of the basin coefficient `f2`
/// (application guidelines, p. 1044), extended to 6000 m/s as OpenQuake does.
const BASIN_VS30: [f64; 7] = [0.0, 150.0, 250.0, 400.0, 700.0, 1000.0, 6000.0];

row_inputs! {
    /// ASK14 inputs, one row per rupture-site pair. Exposed to Python as
    /// `gm_models._core.Ask14Inputs`. Distances are in km.
    Ask14Inputs, Ask14Row {
        mag: f64,
        rrup: f64,
        rx: f64,
        ry0: f64,
        dip: f64,
        ztor: f64,
        rake: f64,
        width: f64,
        vs30: f64,
        /// Metres; negative means unknown, replaced by the reference depth.
        z1pt0: f64,
        vs30measured: bool,
    }
}

/// Site response term (p. 1033; `Vs30*` from Eqs. 8 and 9 passed as `vs30_star`)
/// at ground motion `sa1180` on reference rock.
fn site_response(c: &Coeffs, vs30: f64, vs30_star: f64, sa1180: f64) -> f64 {
    let vs30_rat = vs30_star / c.vlin;
    if vs30 >= c.vlin {
        (c.a10 + c.b * N) * vs30_rat.ln()
    } else {
        c.a10 * vs30_rat.ln() - c.b * (sa1180 + c.c).ln()
            + c.b * (sa1180 + c.c * vs30_rat.powf(N)).ln()
    }
}

/// The ASK_14 model: `AbrahamsonEtAl2014` (default California model,
/// mainshocks). SA periods outside the table are an error.
pub struct Ask14;

impl GMM for Ask14 {
    type Row = Ask14Row;
    type Coeffs = Coeffs;
    /// `sigma_mu_epsilon`.
    type Params = f64;

    fn coeffs(im: IM, _: f64) -> Result<Coeffs, GMMError> {
        Coeffs::for_im(im)
    }

    fn compute(
        im: IM,
        c: &Coeffs,
        row: &Ask14Row,
        sigma_mu_epsilon: f64,
    ) -> Result<MeanStd, GMMError> {
        compute(im, c, row, sigma_mu_epsilon)
    }
}

/// [`Ask14`] for every row. Exposed to Python as `gm_models._core.ask14`.
#[pyfunction]
pub fn ask14(
    py: Python<'_>,
    inputs: PyRef<'_, Ask14Inputs>,
    im: IM,
    sigma_mu_epsilon: f64,
) -> PyResult<GMMResult> {
    evaluate::<Ask14>(py, &inputs.rows, im, sigma_mu_epsilon)
}

/// Computes mean + standard deviations for a single rupture-site pair.
///
/// `c` must be `Coeffs::for_im(im)`. Errors only if `vs30` is outside the
/// basin-term interpolation range [0, 6000] m/s (as in OpenQuake).
pub fn compute(
    im: IM,
    c: &Coeffs,
    row: &Ask14Row,
    sigma_mu_epsilon: f64,
) -> Result<MeanStd, GMMError> {
    let Ask14Row {
        mag,
        rrup,
        rx,
        ry0,
        dip,
        ztor,
        rake,
        width,
        vs30,
        z1pt0,
        vs30measured,
    } = *row;

    // Basic form, Eq. 2, with the fictitious depth c4m.
    let c4m = if mag > 5.0 {
        c.c4
    } else if mag < 4.0 {
        1.0
    } else {
        c.c4 - (c.c4 - 1.0) * (5.0 - mag)
    };
    let r = rrup.hypot(c4m);
    let basic = c.a1
        + c.a17 * rrup
        + if mag >= c.m1 {
            c.a5 * (mag - c.m1) + c.a8 * (8.5 - mag).powi(2) + (c.a2 + c.a3 * (mag - c.m1)) * r.ln()
        } else if mag >= M2 {
            c.a4 * (mag - c.m1) + c.a8 * (8.5 - mag).powi(2) + (c.a2 + c.a3 * (mag - c.m1)) * r.ln()
        } else {
            c.a4 * (M2 - c.m1)
                + c.a8 * (8.5 - M2).powi(2)
                + c.a6 * (mag - M2)
                + c.a7 * (mag - M2).powi(2)
                + (c.a2 + c.a3 * (M2 - c.m1)) * r.ln()
        };

    // Style-of-faulting terms, Eqs. 5 and 6 (rake ranges from Table 2).
    let f_sof = (mag - 4.0).clamp(0.0, 1.0);
    let faulting = if rake > 30.0 && rake < 150.0 {
        c.a11 * f_sof
    } else if rake > -150.0 && rake < -30.0 {
        c.a12 * f_sof
    } else {
        0.0
    };

    // Hanging wall term with tapers T1 to T5 (Eqs. 12 to 15a). All
    // tapers are evaluated before the product, as OpenQuake does.
    let fhw = if rx > 0.0 && dip != 90.0 { 1.0 } else { 0.0 };
    let t1 = if dip <= 30.0 {
        60.0 / 45.0
    } else {
        (90.0 - dip) / 45.0
    };
    let t2 = if mag <= 5.5 {
        0.0
    } else if mag > 6.5 {
        1.0 + A2HW * (mag - 6.5)
    } else {
        1.0 + A2HW * (mag - 6.5) - (1.0 - A2HW) * (mag - 6.5).powi(2)
    };
    let r1 = width * dip.to_radians().cos();
    let r2 = 3.0 * r1;
    let t3 = if rx < r1 {
        H1 + H2 * (rx / r1) + H3 * (rx / r1).powi(2)
    } else if rx <= r2 {
        1.0 - (rx - r1) / (r2 - r1)
    } else {
        0.0
    };
    let t4 = if ztor <= 10.0 {
        1.0 - ztor.powi(2) / 100.0
    } else {
        0.0
    };
    let ry_excess = ry0 - rx * 20f64.to_radians().tan();
    let t5 = if ry_excess <= 0.0 {
        1.0
    } else if ry_excess < 5.0 {
        1.0 - ry_excess / 5.0
    } else {
        0.0
    };
    let hanging_wall = fhw * c.a13 * t1 * t2 * t3 * t4 * t5;

    // Depth-to-top-of-rupture term (p. 1042).
    let ztor_term = c.a15 * (ztor / 20.0).min(1.0);

    // Vs30* limit v1, Eq. 9. Applies to PGV too, which takes the PGA value.
    let v1 = match im {
        IM::SA { period } if period > 0.5 && period < 3.0 => {
            (-0.35 * (period / 0.5).ln() + 1500f64.ln()).exp()
        }
        IM::SA { period } if period >= 3.0 => 800.0,
        _ => 1500.0,
    };

    // Median on reference rock (Vs30 = 1180 m/s), with no basin term since
    // OpenQuake feeds the reference z1pt0 there. Aftershock term is 0.
    let sa1180 = (basic
        + faulting
        + site_response(c, VS30_REF, VS30_REF.min(v1), 0.0)
        + hanging_wall
        + ztor_term)
        .exp();

    // Basin depth term (p. 1042), with the reference Z1.0 of Eq. 18 (km) and the
    // linear interpolation of f2 over Vs30 from the application guidelines.
    let z1_ref = (-7.67 / 4.0
        * ((vs30.powi(4) + 610f64.powi(4)) / (1360f64.powi(4) + 610f64.powi(4))).ln())
    .exp()
        / 1000.0;
    let z1 = if z1pt0 / 1000.0 < 0.0 {
        z1_ref
    } else {
        z1pt0 / 1000.0
    };
    if !(BASIN_VS30[0]..=BASIN_VS30[6]).contains(&vs30) {
        return Err(GMMError::OutOfInterpolationRange {
            x: vs30,
            min: BASIN_VS30[0],
            max: BASIN_VS30[6],
        });
    }
    let basin_f2 = [c.a43, c.a43, c.a44, c.a45, c.a46, c.a46, c.a46];
    let i = BASIN_VS30.partition_point(|&v| v < vs30).max(1);
    let f2 = basin_f2[i - 1]
        + (vs30 - BASIN_VS30[i - 1]) / (BASIN_VS30[i] - BASIN_VS30[i - 1])
            * (basin_f2[i] - basin_f2[i - 1]);
    let basin = f2 * ((z1 + 0.01) / (z1_ref + 0.01)).ln();

    let vs30_star = vs30.min(v1);
    let mean = basic
        + hanging_wall
        + site_response(c, vs30, vs30_star, sa1180)
        + ztor_term
        + faulting
        + basin
        + sigma_mu_epsilon * nga_west2_epistemic_sigma(mag, rrup);

    // Derivative of the site term with respect to ln(Sa1180), Eq. 30.
    let der_amp = if vs30 < c.vlin {
        c.b * sa1180 * (-1.0 / (sa1180 + c.c) + 1.0 / (sa1180 + c.c * (vs30 / c.vlin).powf(N)))
    } else {
        0.0
    };

    // Within-event standard deviation, Eq. 24 (p. 1046).
    let (s1, s2) = if vs30measured {
        (c.s1m, c.s2m)
    } else {
        (c.s1e, c.s2e)
    };
    let phi_al = if mag < 4.0 {
        s1
    } else if mag >= 6.0 {
        s2
    } else {
        s1 + (s2 - s1) / 2.0 * (mag - 4.0)
    };
    // For small magnitudes and long periods phi_al can fall below phi_amp;
    // the authors' fix is to shrink phi_amp to 0.99 phi_al.
    let phi_amp = if phi_al < 0.4 { 0.99 * phi_al } else { 0.4 };
    let phi_b = (phi_al.powi(2) - phi_amp.powi(2)).sqrt();
    let phi = (phi_b.powi(2) * (1.0 + der_amp).powi(2) + phi_amp.powi(2)).sqrt();

    // Between-event standard deviation, Eq. 25 (tau_b = tau_al).
    let tau_al = if mag < 5.0 {
        c.s3
    } else if mag >= 7.0 {
        c.s4
    } else {
        c.s3 + (c.s4 - c.s3) / 2.0 * (mag - 5.0)
    };
    let tau = tau_al * (1.0 + der_amp);

    Ok(MeanStd {
        mean,
        total: tau.hypot(phi),
        inter: tau,
        intra: phi,
    })
}

/// Verification against OpenQuake's reference CSVs (generated from the Matlab
/// implementation in the paper's Annex 1), read from the `external/oq-engine`
/// submodule. `#[ignore]`d so a plain `cargo test` works without it.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> Ask14Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        Ask14Row {
            mag: num("rup_mag"),
            rrup: num("dist_rrup"),
            rx: num("dist_rx"),
            ry0: num("dist_ry0"),
            dip: num("rup_dip"),
            ztor: num("rup_ztor"),
            rake: num("rup_rake"),
            width: num("rup_width"),
            vs30: num("site_vs30"),
            z1pt0: num("site_z1pt0"),
            vs30measured: num("site_vs30measured") != 0.0,
        }
    }

    fn check(file: &str, pick: fn(&MeanStd) -> f64) {
        check_csv(&format!("ASK14/{file}"), row, |im, row| {
            pick(&compute(im, &Coeffs::for_im(im).unwrap(), row, 0.0).unwrap())
        });
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake() {
        check("ASK14_ResMEAN_RegCAL.csv", |r| r.mean.exp());
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn stddevs_match_openquake() {
        check("ASK14_ResStdTot_RegCAL.csv", |r| r.total);
        check("ASK14_ResStdPhi_RegCAL.csv", |r| r.intra);
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

    fn row() -> impl Strategy<Value = Ask14Row> {
        (
            (3.0..8.5f64, 0.1..300.0f64, -100.0..100.0f64, 0.0..100.0f64),
            (10.0..90.0f64, 0.0..20.0f64, -180.0..180.0f64, 1.0..30.0f64),
            (100.0..1500.0f64, -1.0..2000.0f64, any::<bool>()),
        )
            .prop_map(
                |((mag, rrup, rx, ry0), (dip, ztor, rake, width), (vs30, z1pt0, vs30measured))| {
                    Ask14Row {
                        mag,
                        rrup,
                        rx,
                        ry0,
                        dip,
                        ztor,
                        rake,
                        width,
                        vs30,
                        z1pt0,
                        vs30measured,
                    }
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
            let c = Coeffs::for_im(im).unwrap();
            let r = compute(im, &c, &row, sigma_mu_epsilon).unwrap();
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }
    }
}
