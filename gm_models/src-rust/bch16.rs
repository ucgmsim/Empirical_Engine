//! BC Hydro (Abrahamson et al. 2016) subduction GMPE with the ESHM20 theta6
//! calibration ("BCH_16" in `oq_wrapper`), interface and in-slab.
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.eshm20_bchydro`
//! (`BCHydroESHM20SInter`, `BCHydroESHM20SSlab`) and the
//! `openquake.hazardlib.gsim.abrahamson_2015` base class, specialised to the
//! configuration `oq_wrapper` instantiates them with: `faba_taper_model="Step"`,
//! and otherwise the defaults (`ergodic=True`, `theta6_adjustment=0`,
//! `sigma_mu_epsilon=0`, central magnitude scaling). `oq_wrapper` fills a
//! missing `xvf` column with 0, which the Python wrapper reproduces. The
//! Low/High classes, the Montalva variants and the other forearc-backarc taper
//! models are not ported.
//!
//! Equations follow Abrahamson et al. (2016), "BC Hydro Ground Motion
//! Prediction Equations For Subduction Earthquakes", Earthquake Spectra
//! 32(1), as organised in OpenQuake: Eq. 1 (functional form), Eq. 2
//! (magnitude), Eq. 3 (slab depth).
//!
//! Slab coefficients: the slab table of OpenQuake differs from the interface
//! table only in theta6 (checked when the CSVs were exported), so only the
//! slab theta6 column is read.
//!
//! Quirks reproduced from OpenQuake: the nonlinear site branch uses the raw
//! `vs30 >= vlin` (see [`crate::a18::site_response`], shared with A_18, whose
//! site term is algebraically the same); the standard deviations come straight
//! from the table's `sigma`, `tau` and `phi` columns, and the table's total
//! sigma (0.74) is not `sqrt(tau^2 + phi^2)` (0.7382), so unlike the other
//! models total^2 != inter^2 + intra^2 here.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/bch16_sinter_coefficients.rs"));
}
pub mod sslab {
    include!(concat!(env!("OUT_DIR"), "/bch16_sslab_coefficients.rs"));
}
pub mod mag_scale {
    include!(concat!(env!("OUT_DIR"), "/bch16_sinter_mag_scale.rs"));
}

use crate::a18::{site_response, N};
use crate::error::GMMError;
use crate::gmm::{evaluate, GMM};
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::row_inputs;
use pyo3::prelude::*;

/// Period-independent coefficients (Table 2 of the paper).
const THETA3: f64 = 0.1;
const THETA4: f64 = 0.9;
const THETA9: f64 = 0.4;
const C4: f64 = 10.0;
/// Reference magnitude C1 of Eq. 2 (theta5 = 0, so there is no linear term
/// above the hinge).
const C1: f64 = 7.8;
/// Slab magnitude-scaling shift DeltaC1 (central branch).
const DC1_SLAB: f64 = -0.3;

row_inputs! {
    /// BCH_16 inputs, one row per rupture-site pair. `xvf` is the signed
    /// distance to the volcanic front (km, positive on the forearc side,
    /// negative on the backarc side, as in OpenQuake). The slab model uses
    /// hypocentral distance and depth. Exposed to Python as
    /// `gm_models._core.Bch16Inputs`, built with `Bch16Inputs.interface(...)` or
    /// `Bch16Inputs.slab(...)`.
    Bch16Inputs, Bch16Row {
        /// Subduction interface inputs.
        Interface => interface { mag: f64, rrup: f64, vs30: f64, xvf: f64 },
        /// Subduction slab inputs.
        Slab => slab { mag: f64, rhypo: f64, vs30: f64, xvf: f64, hypo_depth: f64 },
    }
}

/// Coefficients at one IM for one tectonic type: the (interface) table, the
/// tectonic-type theta6, and the magnitude-scaling shift DeltaC1.
#[derive(Debug, Clone, Copy)]
pub struct Bch16Coeffs {
    pub c: coeffs::Coeffs,
    pub theta6: f64,
    pub dc1: f64,
}

impl Bch16Coeffs {
    /// Coefficients for `im` (PGA or SA), for slab events if `slab`. Fails for
    /// other IMs and for SA periods outside the tables.
    pub fn for_im(im: IM, slab: bool) -> Result<Self, GMMError> {
        let c = coeffs::Coeffs::for_im(im)?;
        Ok(if slab {
            Self {
                c,
                theta6: sslab::Coeffs::for_im(im)?.theta6,
                dc1: DC1_SLAB,
            }
        } else {
            Self {
                c,
                theta6: c.theta6,
                dc1: mag_scale::Coeffs::for_im(im)?.dc1,
            }
        })
    }
}

/// The BCH_16 model: `BCHydroESHM20SInter` / `BCHydroESHM20SSlab`
/// (`faba_taper_model="Step"`). SA periods outside the table are an error.
pub struct Bch16;

impl GMM for Bch16 {
    type Row = Bch16Row;
    /// Coefficients at the IM and at PGA.
    type Coeffs = (Bch16Coeffs, Bch16Coeffs);
    /// Whether the rows are slab rows.
    type Params = bool;

    fn coeffs(im: IM, slab: bool) -> Result<Self::Coeffs, GMMError> {
        Ok((
            Bch16Coeffs::for_im(im, slab)?,
            Bch16Coeffs::for_im(IM::PGA {}, slab)?,
        ))
    }

    fn compute(
        _im: IM,
        (c, c_pga): &Self::Coeffs,
        row: &Bch16Row,
        _: bool,
    ) -> Result<MeanStd, GMMError> {
        Ok(compute(c, c_pga, row))
    }
}

/// [`Bch16`] for every row. The tectonic type is set by how `inputs` was
/// built. Exposed to Python as `gm_models._core.bch16`.
#[pyfunction]
pub fn bch16(py: Python<'_>, inputs: PyRef<'_, Bch16Inputs>, im: IM) -> PyResult<GMMResult> {
    let slab = matches!(inputs.rows.first(), Some(Bch16Row::Slab { .. }));
    evaluate::<Bch16>(py, &inputs.rows, im, slab)
}

/// Computes mean + standard deviations for a single rupture-site pair.
/// `c` and `c_pga` are `Bch16Coeffs::for_im` for the target IM and for PGA,
/// with `slab` matching `row`.
pub fn compute(c: &Bch16Coeffs, c_pga: &Bch16Coeffs, row: &Bch16Row) -> MeanStd {
    let (mag, dist, vs30, xvf) = match *row {
        Bch16Row::Interface {
            mag,
            rrup,
            vs30,
            xvf,
        } => (mag, rrup, vs30, xvf),
        Bch16Row::Slab {
            mag,
            rhypo,
            vs30,
            xvf,
            ..
        } => (mag, rhypo, vs30, xvf),
    };

    // Everything but the site term, for coefficients `c`.
    let ground = |c: &Bch16Coeffs| {
        let k = &c.c;
        // Magnitude scaling, Eq. 2.
        let dmag = C1 + c.dc1;
        let f_mag = if mag > dmag {
            0.0
        } else {
            THETA4 * (mag - dmag)
        };
        let mag_term = k.theta1 + THETA4 * c.dc1 + f_mag + k.theta13 * (10.0 - mag).powi(2);
        // Distance scaling and forearc/backarc terms. The slab model adds
        // theta10 and theta14, and the depth term of Eq. 3 (depth capped at 120 km).
        let (theta10, theta14, depth_term, faba_a, faba_b, faba_min_dist) = match *row {
            Bch16Row::Interface { .. } => (0.0, 0.0, 0.0, k.theta15, k.theta16, 100.0),
            Bch16Row::Slab { hypo_depth, .. } => (
                k.theta10,
                k.theta14,
                k.theta11 * (hypo_depth.min(120.0) - 60.0),
                k.theta7,
                k.theta8,
                85.0,
            ),
        };
        let dist_term = (k.theta2 + theta14 + THETA3 * (mag - C1))
            * (dist + C4 * ((mag - 6.0) * THETA9).exp()).ln()
            + c.theta6 * dist
            + theta10;
        // ESHM20 forearc/backarc term with the "Step" taper: applied in full
        // for backarc sites (xvf < 0), not at all for forearc ones.
        let faba_term = if -xvf > 0.0 {
            faba_a + faba_b * (dist.max(faba_min_dist) / 40.0).ln()
        } else {
            0.0
        };
        mag_term + dist_term + depth_term + faba_term
    };

    // Median PGA on rock (vs30 = 1000 m/s), the input of the nonlinear site term.
    let k_pga = &c_pga.c;
    let pga1000 =
        (ground(c_pga) + (k_pga.theta12 + k_pga.b * N) * (1000.0 / k_pga.vlin).ln()).exp();
    let k = &c.c;
    MeanStd {
        mean: ground(c) + site_response(k.theta12, k.b, k.vlin, vs30, pga1000),
        total: k.sigma,
        inter: k.tau,
        intra: k.phi,
    }
}

/// Layer 1 verification against OpenQuake's reference tables for the central
/// models (`faba_taper_model="Step"`), read from the `external/oq-engine`
/// submodule. These CSVs only hold the mean. `#[ignore]`d so a plain
/// `cargo test` works without it; run with `cargo test -- --ignored`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    /// Slab CSVs are the only ones with `dist_rhypo`.
    fn row(rec: &HashMap<&str, &str>) -> Bch16Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        let (mag, vs30, xvf) = (num("rup_mag"), num("site_vs30"), num("site_xvf"));
        match rec.get("dist_rhypo") {
            Some(_) => Bch16Row::Slab {
                mag,
                rhypo: num("dist_rhypo"),
                vs30,
                xvf,
                hypo_depth: num("rup_hypo_depth"),
            },
            None => Bch16Row::Interface {
                mag,
                rrup: num("dist_rrup"),
                vs30,
                xvf,
            },
        }
    }

    fn check(file: &str, slab: bool) {
        check_csv(&format!("eshm20_bchydro/{file}"), row, |im, row| {
            let c_pga = Bch16Coeffs::for_im(IM::PGA {}, slab).unwrap();
            compute(&Bch16Coeffs::for_im(im, slab).unwrap(), &c_pga, row)
                .mean
                .exp()
        });
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake() {
        check("BCHydroESHM20_SINTER_CENTRAL_MEAN.csv", false);
        check("BCHydroESHM20_SSLAB_CENTRAL_MEAN.csv", true);
    }
}

/// Property tests over the supported input domain. These need no external
/// data, so they run in a plain `cargo test`.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn row() -> impl Strategy<Value = Bch16Row> {
        (
            5.0..9.5f64,
            0.1..1000.0f64,
            100.0..2000.0f64,
            -300.0..300.0f64,
            0.0..200.0f64,
        )
            .prop_flat_map(|(mag, dist, vs30, xvf, hypo_depth)| {
                prop_oneof![
                    Just(Bch16Row::Interface {
                        mag,
                        rrup: dist,
                        vs30,
                        xvf
                    }),
                    Just(Bch16Row::Slab {
                        mag,
                        rhypo: dist,
                        vs30,
                        xvf,
                        hypo_depth
                    }),
                ]
            })
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_from_table(
            row in row(),
            period in prop_oneof![Just(None), (0.01..=10.0f64).prop_map(Some)],
        ) {
            let im = period.map_or(IM::PGA {}, |period| IM::SA { period });
            let slab = matches!(row, Bch16Row::Slab { .. });
            let c_pga = Bch16Coeffs::for_im(IM::PGA {}, slab).unwrap();
            let r = compute(&Bch16Coeffs::for_im(im, slab).unwrap(), &c_pga, &row);
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            // The table's total sigma is a free column (see the module doc), so
            // only its consistency to within 1 % is checked.
            prop_assert!((r.total.powi(2) / (r.inter.powi(2) + r.intra.powi(2)) - 1.0).abs() < 0.01);
        }

        #[test]
        fn periods_above_table_are_rejected(period in 10.0001..1e3f64) {
            prop_assert!(Bch16Coeffs::for_im(IM::SA { period }, false).is_err(), "period {period} accepted");
        }
    }
}
