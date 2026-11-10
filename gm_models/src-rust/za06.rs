//! Zhao et al. (2006) active shallow crust, subduction interface and subduction
//! slab GMPEs ("ZA_06" in `oq_wrapper`).
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.zhao_2006`
//! (`ZhaoEtAl2006Asc`, `ZhaoEtAl2006SInter`, `ZhaoEtAl2006SSlab`), specialised
//! to the configuration `oq_wrapper` instantiates them with: default
//! `cb14_basin_term=False, m9_basin_term=False`, so no basin term (and no
//! `z2pt5`) is needed. The NSHMP and Cascadia variants are not ported.
//!
//! Equation numbers ("Eq. N") refer to Zhao et al. (2006), "Attenuation
//! Relations of Strong Ground Motion in Japan Using Site Classification Based
//! on Predominant Period", BSSA 96(3), 898-913.
//!
//! Quirks reproduced from OpenQuake: the site class is chosen from `vs30` with
//! the Table 2 boundaries (hard rock above 1100 m/s); the slab model replaces
//! `rrup == 0` by 0.1 km in its distance terms; the magnitude-squared term
//! (Eq. 5) uses hinge 6.3 for crustal/interface and 6.5 for slab events.
//! The tables stop at 5 s and start at 0.05 s: the generated coefficient
//! lookup interpolates between PGA (anchored at 0.01 s) and 0.05 s, as
//! OpenQuake does. SA periods above 5 s are an error unless `extrapolate` is
//! set, which applies `oq_wrapper`'s constant-displacement extrapolation (see
//! [`za06`]).
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod asc {
    include!(concat!(env!("OUT_DIR"), "/za06_asc_coefficients.rs"));
}
pub mod sinter {
    include!(concat!(env!("OUT_DIR"), "/za06_sinter_coefficients.rs"));
}
pub mod sslab {
    include!(concat!(env!("OUT_DIR"), "/za06_sslab_coefficients.rs"));
}

use crate::error::GMMError;
use crate::gmm::{evaluate, GMM};
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::row_inputs;
use pyo3::prelude::*;

/// Standard gravity (m/s^2), `scipy.constants.g`.
const G: f64 = 9.80665;

row_inputs! {
    /// ZA_06 inputs, one row per rupture-site pair. Exposed to Python as
    /// `gm_models._core.Za06Inputs`, built with `Za06Inputs.asc(...)`,
    /// `.interface(...)` or `.slab(...)`.
    Za06Inputs, Za06Row {
        /// Active shallow crust inputs.
        Asc => asc { mag: f64, rrup: f64, vs30: f64, hypo_depth: f64, rake: f64 },
        /// Subduction interface inputs.
        Interface => interface { mag: f64, rrup: f64, vs30: f64, hypo_depth: f64 },
        /// Subduction slab inputs.
        Slab => slab { mag: f64, rrup: f64, vs30: f64, hypo_depth: f64 },
    }
}

/// Coefficients of all three tables at one IM. Only the tectonic-type table
/// matching the row is read by [`compute`].
#[derive(Debug, Clone, Copy)]
pub struct Za06Coeffs {
    pub asc: asc::Coeffs,
    pub sinter: sinter::Coeffs,
    pub sslab: sslab::Coeffs,
}

impl Za06Coeffs {
    /// Coefficients for `im` (PGA or SA). Fails for other IMs and for SA
    /// periods outside the tables.
    pub fn for_im(im: IM) -> Result<Self, GMMError> {
        Ok(Self {
            asc: asc::Coeffs::for_im(im)?,
            sinter: sinter::Coeffs::for_im(im)?,
            sslab: sslab::Coeffs::for_im(im)?,
        })
    }
}

/// Magnitude-squared correction, Eq. 5.
fn magnitude_squared(p: f64, hinge: f64, q: f64, w: f64, mag: f64) -> f64 {
    p * (mag - hinge) + q * (mag - hinge).powi(2) + w
}

/// The ZA_06 model: `ZhaoEtAl2006Asc`, `ZhaoEtAl2006SInter` or
/// `ZhaoEtAl2006SSlab`. SA periods above the table are an error unless
/// `extrapolate` ([`GMM::Params`]) is true, in which case they are extrapolated
/// as in `oq_wrapper` (`_oq_run_pSA`): evaluated at the largest tabulated
/// period `max`, with 2 ln(max / period) added to the ln mean (constant
/// spectral displacement).
pub struct Za06;

impl GMM for Za06 {
    type Row = Za06Row;
    /// Coefficients and the extrapolation shift of the ln mean.
    type Coeffs = (Za06Coeffs, f64);
    /// `extrapolate`.
    type Params = bool;

    fn coeffs(im: IM, extrapolate: bool) -> Result<Self::Coeffs, GMMError> {
        let max = asc::SA_PERIODS[asc::SA_PERIODS.len() - 1];
        let (im, ln_shift) = match im {
            IM::SA { period } if extrapolate && period > max => {
                (IM::SA { period: max }, 2.0 * (max / period).ln())
            }
            _ => (im, 0.0),
        };
        Ok((Za06Coeffs::for_im(im)?, ln_shift))
    }

    fn compute(
        _im: IM,
        (c, ln_shift): &Self::Coeffs,
        row: &Za06Row,
        _: bool,
    ) -> Result<MeanStd, GMMError> {
        let r = compute(c, row);
        Ok(MeanStd {
            mean: r.mean + ln_shift,
            ..r
        })
    }
}

/// [`Za06`] for every row. The tectonic type is set by how `inputs` was built.
/// Exposed to Python as `gm_models._core.za06`.
#[pyfunction]
pub fn za06(
    py: Python<'_>,
    inputs: PyRef<'_, Za06Inputs>,
    im: IM,
    extrapolate: bool,
) -> PyResult<GMMResult> {
    evaluate::<Za06>(py, &inputs.rows, im, extrapolate)
}

/// Computes mean + standard deviations for a single rupture-site pair, with
/// `c` from [`Za06Coeffs::for_im`].
pub fn compute(c: &Za06Coeffs, row: &Za06Row) -> MeanStd {
    let a = &c.asc;
    let (mag, rrup, vs30, hypo_depth) = match *row {
        Za06Row::Asc {
            mag,
            rrup,
            vs30,
            hypo_depth,
            ..
        }
        | Za06Row::Interface {
            mag,
            rrup,
            vs30,
            hypo_depth,
        }
        | Za06Row::Slab {
            mag,
            rrup,
            vs30,
            hypo_depth,
        } => (mag, rrup, vs30, hypo_depth),
    };
    // OpenQuake avoids the slab singularity at rrup = 0 by using 0.1 km.
    let d = match row {
        Za06Row::Slab { .. } if rrup == 0.0 => 0.1,
        _ => rrup,
    };

    // Site class term, Table 2 (hard rock, rock, hard/medium/soft soil).
    let site = if vs30 > 1100.0 {
        a.ch
    } else if vs30 > 600.0 {
        a.c1
    } else if vs30 > 300.0 {
        a.c2
    } else if vs30 > 200.0 {
        a.c3
    } else {
        a.c4
    };
    // Focal depth term of Eq. 1: depth capped at 125 km, active from hc = 15 km.
    let h = hypo_depth.clamp(0.0, 125.0);
    let focal_depth = if h >= 15.0 { a.e * (h - 15.0) } else { 0.0 };

    // Eq. 1 terms common to all types, then the type-specific terms and tau.
    let common = a.a * mag + a.b * d - (d + a.c * (a.d * mag).exp()).ln() + focal_depth + site;
    let (specific, tau) = match *row {
        // Faulting style: dip-slip for 45 < rake < 135.
        Za06Row::Asc { rake, .. } => (
            if rake > 45.0 && rake < 135.0 {
                a.fr
            } else {
                0.0
            } + magnitude_squared(0.0, 6.3, a.qc, a.wc, mag),
            a.tauc,
        ),
        Za06Row::Interface { .. } => (
            c.sinter.si + magnitude_squared(0.0, 6.3, c.sinter.qi, c.sinter.wi, mag),
            c.sinter.taui,
        ),
        Za06Row::Slab { .. } => (
            c.sslab.ss
                + c.sslab.ssl * d.ln()
                + magnitude_squared(c.sslab.ps, 6.5, c.sslab.qs, c.sslab.ws, mag),
            c.sslab.taus,
        ),
    };

    // Eq. 3: sigma is the intra-event and tau the inter-event standard deviation.
    MeanStd {
        // Eq. 1 is in cm/s^2: convert to g.
        mean: common + specific + (1e-2 / G).ln(),
        total: a.sigma.hypot(tau),
        inter: tau,
        intra: a.sigma,
    }
}

/// Layer 1 verification against OpenQuake's reference tables (generated from
/// the Fortran code by J. Zhao, and from Matlab for Vs30 > 1100 m/s), read from
/// the `external/oq-engine` submodule. `#[ignore]`d so a plain `cargo test`
/// works without it; run with `cargo test -- --ignored`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv_tol;
    use std::collections::HashMap;

    fn row(tect: &'static str) -> impl Fn(&HashMap<&str, &str>) -> Za06Row {
        move |rec| {
            let num = |k: &str| -> f64 { rec[k].trim().parse().unwrap() };
            let (mag, rrup, vs30, hypo_depth) = (
                num("rup_mag"),
                num("dist_rrup"),
                num("site_vs30"),
                num("rup_hypo_depth"),
            );
            match tect {
                "Asc" => Za06Row::Asc {
                    mag,
                    rrup,
                    vs30,
                    hypo_depth,
                    rake: num("rup_rake"),
                },
                "SInter" => Za06Row::Interface {
                    mag,
                    rrup,
                    vs30,
                    hypo_depth,
                },
                _ => Za06Row::Slab {
                    mag,
                    rrup,
                    vs30,
                    hypo_depth,
                },
            }
        }
    }

    /// OpenQuake's own tests match the means to 0.4 % and the standard
    /// deviations to 0.1 % (`max_discrep_percentage`, `std_discrep_percentage`).
    fn check(tect: &'static str, file: &str, pick: fn(&MeanStd) -> f64, max_relative: f64) {
        check_csv_tol(
            &format!("ZHAO06/{file}"),
            row(tect),
            |im, row| pick(&compute(&Za06Coeffs::for_im(im).unwrap(), row)),
            max_relative,
        );
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn matches_openquake() {
        for tect in ["Asc", "SInter", "SSlab"] {
            for file in [
                format!("Z06{tect}_MEAN.csv"),
                format!("Z06{tect}_MEAN_Vs30_1200.csv"),
            ] {
                check(tect, &file, |r| r.mean.exp(), 4e-3);
            }
            check(tect, &format!("Z06{tect}_STD_TOTAL.csv"), |r| r.total, 1e-3);
            check(tect, &format!("Z06{tect}_STD_INTER.csv"), |r| r.inter, 1e-3);
            check(tect, &format!("Z06{tect}_STD_INTRA.csv"), |r| r.intra, 1e-3);
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

    fn row() -> impl Strategy<Value = Za06Row> {
        (
            5.0..9.5f64,
            0.0..500.0f64,
            100.0..2000.0f64,
            0.0..200.0f64,
            -180.0..180.0f64,
        )
            .prop_flat_map(|(mag, rrup, vs30, hypo_depth, rake)| {
                prop_oneof![
                    Just(Za06Row::Asc {
                        mag,
                        rrup,
                        vs30,
                        hypo_depth,
                        rake
                    }),
                    Just(Za06Row::Interface {
                        mag,
                        rrup,
                        vs30,
                        hypo_depth
                    }),
                    Just(Za06Row::Slab {
                        mag,
                        rrup,
                        vs30,
                        hypo_depth
                    }),
                ]
            })
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(
            row in row(),
            period in prop_oneof![Just(None), (0.01..=5.0f64).prop_map(Some)],
        ) {
            let im = period.map_or(IM::PGA {}, |period| IM::SA { period });
            let r = compute(&Za06Coeffs::for_im(im).unwrap(), &row);
            prop_assert!(r.mean.is_finite());
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }

        #[test]
        fn periods_outside_table_are_rejected(period in prop_oneof![5.0001..1e3f64, 0.0..0.00999f64]) {
            prop_assert!(Za06Coeffs::for_im(IM::SA { period }).is_err(), "period {period} accepted");
        }
    }

    #[test]
    fn below_table_start_interpolates_between_pga_and_first_period() {
        let (pga, first) = (asc::PGA_COEFFS, asc::SA_COEFFS[0]);
        let c = asc::Coeffs::for_im(IM::SA { period: 0.02 }).unwrap();
        assert!(c.a > pga.a.min(first.a) && c.a < pga.a.max(first.a));
        assert_eq!(asc::Coeffs::for_im(IM::SA { period: 0.01 }).unwrap(), pga);
    }
}
