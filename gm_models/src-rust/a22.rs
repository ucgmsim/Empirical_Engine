//! Atkinson (2022) NZ NSHM backbone GMM ("A_22" in `oq_wrapper`), for active
//! shallow crust (also used for volcanic), subduction interface and subduction
//! slab, with the NZ NSHM2022 `modified_sigma` option.
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.nz22.atkinson_2022`
//! (`Atkinson2022Crust`, `Atkinson2022SInter`, `Atkinson2022SSlab`) and its
//! coefficient file `Atkinson22_coeffs_mod_v8b_sanjay_v2.csv`, specialised to
//! the configuration `oq_wrapper` uses: `modified_sigma` is exposed (it is
//! `True` in `oq_wrapper`), and the epistemic branch is OpenQuake's
//! `epistemic="lower"/"central"/"upper"`. The result is in g for PGA and SA;
//! PGV and other IMs are not defined by the model.
//!
//! The model is documented in Atkinson (2022), "Ground-motion model for New
//! Zealand: backbone model for the 2022 NSHM revision" (GNS report), with the
//! revisions OpenQuake cites in its source; OpenQuake gives no equation
//! numbers. The backarc term (slab only) is the one shared with P_21
//! ([`crate::p21`]), and the modified sigma follows Abrahamson and Gulerce
//! (2022), as in P_21.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/a22_coefficients.rs"));
}

use crate::branch::Branch;
use crate::error::GMMError;
use crate::gmm::{evaluate, GMM};
use crate::im::IM;
use crate::nz22_const::{backarc_term, PERIODS_AG20, RHO_BS, RHO_WS};
use crate::result::{GMMResult, MeanStd};
use crate::row_inputs;
use crate::utils::log_interp;
use coeffs::{Coeffs, PGA_COEFFS};
use pyo3::prelude::*;

/// The model's coefficients are in cm/s^2; the output is converted to g.
const G: f64 = 981.0;
/// Transition distance Rt and geometric spreading rates b1, b2 (Rt = 50 km).
const RT: f64 = 50.0;
const B1: f64 = -1.3;
const B2: f64 = -0.5;
/// Reference velocity of the Seyhan and Stewart (2014) site term.
const VS_REF: f64 = 760.0;
/// 10th/90th percentile z-score and the 0.893 factor that accounts for the
/// reduction in epistemic uncertainty when rupture scenarios are not perfectly
/// correlated (see the OpenQuake source comments).
const EPISTEMIC_Z: f64 = 1.28155 * 0.893;
/// Site variability assumed constant with period.
const PHI_AMP: f64 = 0.3;

/// Tectonic type. Selects the coefficient suffix (`crust`, `inter`, `slab`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tect {
    Crust,
    Interface,
    Slab,
}

/// A single rupture-site pair of [`A22Inputs`]. `backarc` is false unless
/// `tect` is slab.
#[derive(Debug, Clone, Copy)]
pub struct A22Row {
    pub tect: Tect,
    pub mag: f64,
    pub rrup: f64,
    pub vs30: f64,
    pub backarc: bool,
}

row_inputs! {
    /// A22 inputs, one row per rupture-site pair. Exposed to Python as
    /// `gm_models._core.A22Inputs`, built with `A22Inputs.crust(...)`,
    /// `A22Inputs.interface(...)` or `A22Inputs.slab(...)`.
    A22Inputs, A22Row {
        /// Active shallow crust (and volcanic) inputs.
        crust(mag: f64, rrup: f64, vs30: f64) { tect: Tect::Crust, backarc: false },
        /// Subduction interface inputs.
        interface(mag: f64, rrup: f64, vs30: f64) { tect: Tect::Interface, backarc: false },
        /// Subduction slab inputs.
        slab(mag: f64, rrup: f64, vs30: f64, backarc: bool) { tect: Tect::Slab },
    }
}

/// ln of the rock motion (cm/s^2): magnitude scaling, pseudo-depth geometric
/// spreading (Hassani and Atkinson 2018 form) and anelastic attenuation.
/// Slab has its own magnitude coefficients; crust and interface share the
/// `*_crust` ones, as in OpenQuake's `_fmag`.
fn ln_rock(c: &Coeffs, row: &A22Row) -> f64 {
    let (c0, c1, c2) = match row.tect {
        Tect::Slab => (c.c0_slab, c.c1_slab, c.c2_slab),
        _ => (c.c0_crust, c.c1_crust, c.c2_crust),
    };
    let fmag = c0 + c1 * row.mag + c2 * row.mag.powi(2);

    // Pseudo-depth h (with the revisions of 21.06.2022), R and Rref.
    let h = 10f64.powf(-0.405 + 0.235 * row.mag);
    let r = row.rrup.hypot(h);
    let r_ref = 1f64.hypot(h);
    let ln_z = if r > RT {
        B1 * RT.ln() + B2 * (r / RT).ln()
    } else {
        B1 * r.ln()
    };
    let fz = ln_z + (c.b3 + c.b4 * row.mag) * (r / r_ref).ln();

    let g1 = match row.tect {
        Tect::Crust => 0.008f64.min(0.005 + 0.0016 * c.f.ln()),
        Tect::Interface => 0.006f64.min(0.0045 + 0.0014 * c.f.ln()),
        Tect::Slab => 0.005f64.min(0.004 + 0.0012 * c.f.ln()),
    };
    fmag + fz - g1 * row.rrup
}

/// Epistemic adjustment of the ln mean, as a signed shift. The magnitude
/// part (0.15 per unit above M7) was added in Atkinson's V8 revision.
fn epistemic_shift(branch: Branch, mag: f64, rrup: f64) -> f64 {
    let ln_r = rrup.ln();
    let m_term = 0.15 * (mag - 7.0).max(0.0);
    match branch {
        Branch::Central => 0.0,
        Branch::Lower => {
            let a = (0.6 - 0.13 * ln_r)
                .max(0.3)
                .max(-0.25 + 0.12 * ln_r)
                .min(0.5);
            -(a + m_term) * EPISTEMIC_Z
        }
        Branch::Upper => {
            let a = (1.0 - 0.3 * ln_r)
                .max(0.3)
                .max(-0.25 + 0.12 * ln_r)
                .min(0.8);
            (a + m_term) * EPISTEMIC_Z
        }
    }
}

/// (between-event, within-event) standard deviations of the linear model.
/// Within-event includes the site-to-site term.
fn linear_tau_phi(c: &Coeffs, tect: Tect) -> (f64, f64) {
    let (be, we) = match tect {
        Tect::Crust => (c.be_crust, c.we_crust),
        Tect::Interface => (c.be_inter, c.we_inter),
        Tect::Slab => (c.be_slab, c.we_slab),
    };
    (be, we.hypot(c.phis2s))
}

/// Nonlinear (modified sigma) standard deviations, with AG20's rho_W and rho_B
/// and the derivative of the Seyhan and Stewart (2014) site term, as P_21's
/// `stddevs_nonlinear`. PGA takes rho = 1 (OpenQuake's `period < 0.01`).
fn stddevs_nonlinear(
    c: &Coeffs,
    c_pga: &Coeffs,
    im: IM,
    tect: Tect,
    pgar: f64,
    vs30: f64,
) -> Result<(f64, f64, f64), GMMError> {
    let (tau, phi) = linear_tau_phi(c, tect);
    let (tau_pga, phi_pga) = linear_tau_phi(c_pga, tect);
    let phi_b = (phi.powi(2) - PHI_AMP.powi(2)).sqrt();
    let phi_b_pga = (phi_pga.powi(2) - PHI_AMP.powi(2)).sqrt();

    let (rho_w, rho_b) = match im {
        IM::SA { period } if period >= 0.01 => (
            log_interp(&PERIODS_AG20, &RHO_WS, period)?,
            log_interp(&PERIODS_AG20, &RHO_BS, period)?,
        ),
        _ => (1.0, 1.0),
    };

    let f2 = c.f4 * (c.f5 * (vs30.min(VS_REF) - 360.0)).exp() - c.f4 * (c.f5 * 400.0).exp();
    let partial_f_pga = f2 * pgar / (pgar + 0.1);

    let phi2_nl = phi.powi(2)
        + partial_f_pga.powi(2) * phi_b_pga.powi(2)
        + 2.0 * partial_f_pga * phi_b_pga * phi_b * rho_w;
    let tau2_nl = tau.powi(2)
        + partial_f_pga.powi(2) * tau_pga.powi(2)
        + 2.0 * partial_f_pga * tau_pga * tau * rho_b;
    Ok(((tau2_nl + phi2_nl).sqrt(), tau2_nl.sqrt(), phi2_nl.sqrt()))
}

/// The A_22 model: `Atkinson2022Crust` / `Atkinson2022SInter` /
/// `Atkinson2022SSlab`. SA periods outside the table are an error.
pub struct A22;

impl GMM for A22 {
    type Row = A22Row;
    type Coeffs = Coeffs;
    /// Epistemic branch and `modified_sigma`.
    type Params = (Branch, bool);

    fn coeffs(im: IM, _: Self::Params) -> Result<Coeffs, GMMError> {
        Coeffs::for_im(im)
    }

    fn compute(
        im: IM,
        c: &Coeffs,
        row: &A22Row,
        (branch, modified_sigma): Self::Params,
    ) -> Result<MeanStd, GMMError> {
        compute(im, c, row, branch, modified_sigma)
    }
}

/// [`A22`] for every row. The tectonic type is set by how `inputs` was built.
/// Exposed to Python as `gm_models._core.a22`.
#[pyfunction]
pub fn a22(
    py: Python<'_>,
    inputs: PyRef<'_, A22Inputs>,
    im: IM,
    branch: Branch,
    modified_sigma: bool,
) -> PyResult<GMMResult> {
    evaluate::<A22>(py, &inputs.rows, im, (branch, modified_sigma))
}

/// Computes mean + standard deviations for a single site/rupture pair.
///
/// `c` must be `Coeffs::for_im(im)`; it is passed in so callers looping over
/// sites look it up once rather than per site.
pub fn compute(
    im: IM,
    c: &Coeffs,
    row: &A22Row,
    branch: Branch,
    modified_sigma: bool,
) -> Result<MeanStd, GMMError> {
    let c_pga = &PGA_COEFFS;

    // Backarc term (slab only). It is added to the rock PGA in ln space
    // (OpenQuake multiplies by its exponential) and to the target IM's mean.
    let (fba_pga, fba) = if row.backarc {
        (
            backarc_term(IM::PGA {}, row.rrup)?,
            backarc_term(im, row.rrup)?,
        )
    } else {
        (0.0, 0.0)
    };
    let pga_rock = (ln_rock(c_pga, row) + fba_pga).exp();

    // Site term of Seyhan and Stewart (2014): linear Vs30 scaling up to Vc,
    // plus nonlinear term with f1 = 0 and f3 = 0.1 g.
    let flin = c.c * (row.vs30.min(c.vc) / VS_REF).ln();
    let f3 = 0.1 * G;
    let f2 = c.f4 * ((c.f5 * (row.vs30.min(VS_REF) - 360.0)).exp() - (c.f5 * 400.0).exp());
    let fnl = f2 * ((pga_rock + f3) / f3).ln();

    let mean =
        ln_rock(c, row) + flin + fnl + fba - G.ln() + epistemic_shift(branch, row.mag, row.rrup);

    let (total, inter, intra) = if modified_sigma {
        stddevs_nonlinear(c, c_pga, im, row.tect, pga_rock / G, row.vs30)?
    } else {
        let (tau, phi) = linear_tau_phi(c, row.tect);
        let total = match row.tect {
            Tect::Crust => c.sigma_crust,
            Tect::Interface => c.sigma_inter,
            Tect::Slab => c.sigma_slab,
        };
        (total, tau, phi)
    };

    Ok(MeanStd {
        mean,
        total,
        inter,
        intra,
    })
}

/// Layer 1 verification against OpenQuake's reference CSVs (GNS-produced),
/// read from the `external/oq-engine` submodule. `#[ignore]`d so a plain
/// `cargo test` works without the submodule; run with `cargo test -- --ignored`.
/// Mean checks use OpenQuake's `EPI_SCALE_CORRECTED` lower/upper files, as its
/// own tests do.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv_tol;
    use std::collections::HashMap;

    fn row(tect: Tect) -> impl Fn(&HashMap<&str, &str>) -> A22Row {
        move |rec| {
            let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
            A22Row {
                tect,
                mag: num("rup_mag"),
                rrup: num("dist_rrup"),
                vs30: num("site_vs30"),
                backarc: rec
                    .get("site_backarc")
                    .is_some_and(|b| b.trim().eq_ignore_ascii_case("true")),
            }
        }
    }

    fn check(
        file: &str,
        max_relative: f64,
        tect: Tect,
        branch: Branch,
        modified_sigma: bool,
        pick: fn(&MeanStd) -> f64,
    ) {
        check_csv_tol(
            &format!("nz22/ATKINSON2022/{file}"),
            row(tect),
            |im, row| {
                pick(
                    &compute(
                        im,
                        &Coeffs::for_im(im).unwrap(),
                        row,
                        branch,
                        modified_sigma,
                    )
                    .unwrap(),
                )
            },
            max_relative,
        );
    }

    /// Tolerance for the mean CSVs: OpenQuake's own tests allow a 0.5%
    /// discrepancy, and its output differs from them by up to that much
    /// (e.g. 0.33% for central crust PGA at M4, 1 km), so this is looser than
    /// the 0.1% of the other models.
    const MEAN_TOLERANCE: f64 = 5e-3;

    const TECTS: [(&str, &str, Tect); 3] = [
        ("CRUST", "CRUST", Tect::Crust),
        ("INTER", "INTERFACE", Tect::Interface),
        ("SLAB", "SLAB", Tect::Slab),
    ];

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake() {
        for (name, _, tect) in TECTS {
            for (branch, file) in [
                (Branch::Central, "CENTRAL_MEAN"),
                (Branch::Lower, "LOWER_MEAN_EPI_SCALE_CORRECTED"),
                (Branch::Upper, "UPPER_MEAN_EPI_SCALE_CORRECTED"),
            ] {
                check(
                    &format!("ATKINSON22_{name}_{file}.csv"),
                    MEAN_TOLERANCE,
                    tect,
                    branch,
                    false,
                    |r| r.mean.exp(),
                );
            }
        }
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn total_stddev_matches_openquake() {
        for (_, name, tect) in TECTS {
            for (sigma, modified_sigma) in [("ORIGINAL", false), ("MODIFIED", true)] {
                check(
                    &format!("ATKINSON2022_{name}_{sigma}_SIGMA.csv"),
                    1e-3,
                    tect,
                    Branch::Central,
                    modified_sigma,
                    |r| r.total,
                );
            }
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

    fn im() -> impl Strategy<Value = IM> {
        prop_oneof![
            Just(IM::PGA {}),
            (0.01..=10.0f64).prop_map(|period| IM::SA { period }),
        ]
    }

    fn row() -> impl Strategy<Value = A22Row> {
        (
            prop_oneof![Just(Tect::Crust), Just(Tect::Interface), Just(Tect::Slab)],
            5.0..9.5f64,
            0.1..1000.0f64,
            100.0..2000.0f64,
            any::<bool>(),
        )
            .prop_map(|(tect, mag, rrup, vs30, backarc)| A22Row {
                tect,
                mag,
                rrup,
                vs30,
                backarc: backarc && tect == Tect::Slab,
            })
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_modified_stddevs_consistent(
            im in im(),
            row in row(),
            branch in prop_oneof![Just(Branch::Lower), Just(Branch::Central), Just(Branch::Upper)],
        ) {
            let c = Coeffs::for_im(im).unwrap();
            let r = compute(im, &c, &row, branch, true).unwrap();
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }

        #[test]
        fn lower_mean_is_below_central_below_upper(im in im(), row in row()) {
            let c = Coeffs::for_im(im).unwrap();
            let mean = |b| compute(im, &c, &row, b, true).unwrap().mean;
            prop_assert!(mean(Branch::Lower) < mean(Branch::Central));
            prop_assert!(mean(Branch::Central) < mean(Branch::Upper));
        }
    }
}
