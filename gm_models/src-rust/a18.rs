//! Abrahamson et al. (2018) update of the BC Hydro subduction GMPE ("A_18" in
//! `oq_wrapper`), interface and in-slab.
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.abrahamson_2018`
//! (`AbrahamsonEtAl2018SInter`, `AbrahamsonEtAl2018SSlab`), specialised to the
//! configuration `oq_wrapper` instantiates them with: the central model (no
//! `EPISTEMIC_ADJUSTMENT`, so the High/Low classes and their table columns are
//! omitted), with the Cascadia adjustment `adj_int` (interface) or `adj_slab`
//! (slab) added to the mean. Only the Cascadia version is implemented in
//! OpenQuake.
//!
//! Equations follow Abrahamson et al. (2018), "Update of the BC Hydro
//! Subduction Ground-Motion Model using the NGA-Subduction Dataset", PEER
//! Report 2018/02, as organised in OpenQuake.
//!
//! Quirks reproduced from OpenQuake: the nonlinear site branch is selected by
//! the raw `vs30 >= vlin` while the site term uses `min(vs30, 1000)`; the
//! amplification derivative `dln` uses the raw `vs30`; the cross terms of tau
//! and phi use the target period's `tau0` and `phi0` with PGA's, as coded there
//! (`phi_b` of the target period is built from `phi0`, not from tau).
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/a18_coefficients.rs"));
}

use crate::error::GMMError;
use crate::gmm::{evaluate, GMM};
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::row_inputs;
use coeffs::{Coeffs, PGA_COEFFS};
use pyo3::prelude::*;

/// Site-term exponent n and constant c (shared with the BC Hydro 2016 model).
pub(crate) const N: f64 = 1.18;
pub(crate) const C: f64 = 1.88;
const C4: f64 = 10.0;
const A3: f64 = 0.10;
const A9: f64 = 0.4;
const A10: f64 = 1.73;
/// Hinge magnitude of the slab model.
const C1_SLAB: f64 = 7.2;
const PHI_AMP: f64 = 0.3;

row_inputs! {
    /// A18 inputs, one row per rupture-site pair. Exposed to Python as
    /// `gm_models._core.A18Inputs`, built with `A18Inputs.interface(...)` or
    /// `A18Inputs.slab(...)`.
    A18Inputs, A18Row {
        /// Subduction interface inputs.
        Interface => interface { mag: f64, rrup: f64, vs30: f64 },
        /// Subduction slab inputs. `ztor` is the top of rupture depth (km).
        Slab => slab { mag: f64, rrup: f64, vs30: f64, ztor: f64 },
    }
}
/// Site amplification of the Walling et al. (2008) form shared by the BC Hydro
/// models: `lin` is `a12` (A_18) or `theta12` (BC Hydro 2016), `b` and `vlin`
/// the period-dependent coefficients, `pga1000` the median PGA on
/// `vs30 = 1000 m/s` rock.
pub(crate) fn site_response(lin: f64, b: f64, vlin: f64, vs30: f64, pga1000: f64) -> f64 {
    let ln_arg = (vs30.min(1000.0) / vlin).ln();
    if vs30 >= vlin {
        (lin + b * N) * ln_arg
    } else {
        let arg_n = (vs30.min(1000.0) / vlin).powf(N);
        lin * ln_arg - b * (pga1000 + C).ln() + b * (pga1000 + C * arg_n).ln()
    }
}

/// The A_18 model: `AbrahamsonEtAl2018SInter` / `AbrahamsonEtAl2018SSlab`
/// (default arguments). SA periods outside the table are an error.
pub struct A18;

impl GMM for A18 {
    type Row = A18Row;
    type Coeffs = Coeffs;
    type Params = ();

    fn coeffs(im: IM, (): ()) -> Result<Coeffs, GMMError> {
        Coeffs::for_im(im)
    }

    fn compute(_im: IM, c: &Coeffs, row: &A18Row, (): ()) -> Result<MeanStd, GMMError> {
        Ok(compute(c, row))
    }
}

/// [`A18`] for every row. The tectonic type is set by how `inputs` was built.
/// Exposed to Python as `gm_models._core.a18`.
#[pyfunction]
pub fn a18(py: Python<'_>, inputs: PyRef<'_, A18Inputs>, im: IM) -> PyResult<GMMResult> {
    evaluate::<A18>(py, &inputs.rows, im, ())
}
/// Computes mean + standard deviations for a single rupture-site pair, with
/// `c` from `Coeffs::for_im`.
pub fn compute(c: &Coeffs, row: &A18Row) -> MeanStd {
    let c_pga = &PGA_COEFFS;
    let (mag, rrup, vs30, ztor) = match *row {
        A18Row::Interface { mag, rrup, vs30 } => (mag, rrup, vs30, None),
        A18Row::Slab {
            mag,
            rrup,
            vs30,
            ztor,
        } => (mag, rrup, vs30, Some(ztor)),
    };
    let adj = |c: &Coeffs| {
        if ztor.is_some() {
            c.adj_slab
        } else {
            c.adj_int
        }
    };

    // Base, magnitude, depth and distance terms of the model (without the
    // site term), for coefficients `c`.
    let ground = |c: &Coeffs| {
        let (base, hinge, scale) = match ztor {
            Some(_) => (
                c.a1 + c.a4 * (C1_SLAB - c.c1inter) + A10,
                C1_SLAB,
                c.a2 + c.a14 + A3 * (mag - 7.8),
            ),
            None => (c.a1, c.c1inter, c.a2 + A3 * (mag - 7.8)),
        };
        let f_mag = c.a13 * (10.0 - mag).powi(2);
        // a5 = 0, so there is no linear term above the hinge.
        let mag_term = if mag <= hinge {
            c.a4 * (mag - hinge) + f_mag
        } else {
            f_mag
        };
        // Slab depth term (PEER 2018/02 p. 11).
        let depth_term = match ztor {
            Some(z) if z <= 100.0 => c.a11 * (z - 60.0),
            Some(_) => c.a11 * 40.0,
            None => 0.0,
        };
        let dist_term = scale * (rrup + C4 * (A9 * (mag - 6.0)).exp()).ln() + c.a6 * rrup;
        base + mag_term + depth_term + dist_term
    };

    // Median PGA on rock (vs30 = 1000 m/s), the input of the nonlinear site term.
    let pga1000 =
        (ground(c_pga) + (c_pga.a12 + c_pga.b * N) * (1000.0 / c_pga.vlin).ln() + adj(c_pga)).exp();
    let mean = ground(c) + site_response(c.a12, c.b, c.vlin, vs30, pga1000) + adj(c);

    // Partial derivative of the amplification with respect to ln(pga1000).
    let dln = if vs30 < c.vlin {
        c.b * pga1000 * (-1.0 / (pga1000 + C) + 1.0 / (pga1000 + C * (vs30 / c.vlin).powf(N)))
    } else {
        0.0
    };
    let tau =
        (c.tau0.powi(2) * (1.0 + dln.powi(2)) + 2.0 * dln * c.tau0 * c_pga.tau0 * c.rho_b).sqrt();
    let phi_b = (c.phi0.powi(2) - PHI_AMP.powi(2)).sqrt();
    let phi_b_pga = (c_pga.phi0.powi(2) - PHI_AMP.powi(2)).sqrt();
    let phi =
        (c.phi0.powi(2) + dln.powi(2) * phi_b.powi(2) + 2.0 * dln * phi_b * phi_b_pga * c.rho_w)
            .sqrt();

    MeanStd {
        mean,
        total: tau.hypot(phi),
        inter: tau,
        intra: phi,
    }
}

/// Layer 1 verification against OpenQuake's reference tables for the
/// no-adjustment (central) models, read from the `external/oq-engine`
/// submodule. `#[ignore]`d so a plain `cargo test` works without it; run with
/// `cargo test -- --ignored`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    /// Slab CSVs are the only ones with `rup_ztor`.
    fn row(rec: &HashMap<&str, &str>) -> A18Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        let (mag, rrup, vs30) = (num("rup_mag"), num("dist_rrup"), num("site_vs30"));
        match rec.get("rup_ztor") {
            Some(_) => A18Row::Slab {
                mag,
                rrup,
                vs30,
                ztor: num("rup_ztor"),
            },
            None => A18Row::Interface { mag, rrup, vs30 },
        }
    }

    fn check(file: &str, pick: fn(&MeanStd) -> f64) {
        check_csv(&format!("bchydro18/{file}"), row, |im, row| {
            pick(&compute(&Coeffs::for_im(im).unwrap(), row))
        });
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn matches_openquake() {
        for tect in ["INTERFACE", "INSLAB"] {
            check(&format!("BCHYDRO2018_{tect}_MEAN_NO_ADJUST.csv"), |r| {
                r.mean.exp()
            });
            check(&format!("BCHYDRO2018_{tect}_TOTAL_STDDEV.csv"), |r| r.total);
            check(&format!("BCHYDRO2018_{tect}_INTER_EVENT_STDDEV.csv"), |r| {
                r.inter
            });
            check(&format!("BCHYDRO2018_{tect}_INTRA_EVENT_STDDEV.csv"), |r| {
                r.intra
            });
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

    fn row() -> impl Strategy<Value = A18Row> {
        (5.0..9.5f64, 0.1..1000.0f64, 100.0..2000.0f64, 0.0..200.0f64).prop_flat_map(
            |(mag, rrup, vs30, ztor)| {
                prop_oneof![
                    Just(A18Row::Interface { mag, rrup, vs30 }),
                    Just(A18Row::Slab {
                        mag,
                        rrup,
                        vs30,
                        ztor
                    }),
                ]
            },
        )
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(
            row in row(),
            period in prop_oneof![Just(None), (0.01..=10.0f64).prop_map(Some)],
        ) {
            let im = period.map_or(IM::PGA {}, |period| IM::SA { period });
            let r = compute(&Coeffs::for_im(im).unwrap(), &row);
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }

        #[test]
        fn periods_above_table_are_rejected(period in 10.0001..1e3f64) {
            prop_assert!(Coeffs::for_im(IM::SA { period }).is_err(), "period {period} accepted");
        }
    }
}
