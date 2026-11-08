//! Parker et al. (2020) subduction GMPE ("P_20" in `oq_wrapper`), global
//! (no regional adjustment) version.
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.parker_2020`
//! (`ParkerEtAl2020SInter` / `ParkerEtAl2020SSlab`), specialised to the default
//! constructor (`region=None, saturation_region=None, basin=None,
//! sigma_mu_epsilon=0`), which is what `oq_wrapper` instantiates. That means
//! the basin term is always 0 (it is only defined for Cascadia and Japan) and
//! is omitted, the regional `a0`/`c0`/magnitude-break choices collapse to the
//! global columns, the linear site term's `s1`/`s2` split collapses to a single
//! `s2` coefficient, and the epistemic `sigma_mu` adjustment is omitted.
//!
//! Differences from the NZ NSHM2022 variant in [`crate::p21`]: no backarc
//! term, the original (linear) sigma model only, and the nonlinear site term is
//! switched off for SA periods of 3 s and above (`_non_linear_term`).
//!
//! Equation numbers ("Eq. N") refer to Parker et al. (2022), "NGA-Subduction
//! global ground motion models with regional adjustment factors", Earthquake
//! Spectra 38(1).
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/p20_coefficients.rs"));
}

use crate::error::GMMError;
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::utils::check_lengths;
use coeffs::{Coeffs, PGA_COEFFS};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// b4 in Eq. 2.
const B4: f64 = 0.1;
/// f3 in Eq. 9.
const F3: f64 = 0.05;
/// Constants 200 and 760 in Eq. 10.
const VB: f64 = 200.0;
const VREF_FNL: f64 = 760.0;
/// Vref in Eq. 8.
const VREF: f64 = 760.0;
/// R1, R2 in Eq. 19 and V1, V2 in Eq. 20.
const R1: f64 = 200.0;
const R2: f64 = 500.0;
const V1: f64 = 200.0;
const V2: f64 = 500.0;

/// Global mc (Table 3).
const MB_INTERFACE: f64 = 7.9;
const MB_SLAB: f64 = 7.6;

/// P20 input columns, one value per rupture-site pair. `hypo_depth` exists
/// only on the `Slab` variant. Exposed to Python as `gm_models._core.P20Inputs`,
/// built with `P20Inputs.interface(...)` or `P20Inputs.slab(...)`. Column
/// lengths are checked in [`P20Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub enum P20Inputs {
    Interface {
        mag: Vec<f64>,
        rrup: Vec<f64>,
        vs30: Vec<f64>,
    },
    Slab {
        mag: Vec<f64>,
        rrup: Vec<f64>,
        vs30: Vec<f64>,
        hypo_depth: Vec<f64>,
    },
}

/// A single row of [`P20Inputs`].
#[derive(Debug, Clone, Copy)]
pub enum P20Row {
    Interface {
        mag: f64,
        rrup: f64,
        vs30: f64,
    },
    Slab {
        mag: f64,
        rrup: f64,
        vs30: f64,
        hypo_depth: f64,
    },
}

impl P20Inputs {
    /// The rupture-site pairs, in order. Fails if the columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = P20Row> + '_, GMMError> {
        let n = match self {
            Self::Interface { mag, rrup, vs30 } => check_lengths(&[
                ("mag", mag.len()),
                ("rrup", rrup.len()),
                ("vs30", vs30.len()),
            ]),
            Self::Slab {
                mag,
                rrup,
                vs30,
                hypo_depth,
            } => check_lengths(&[
                ("mag", mag.len()),
                ("rrup", rrup.len()),
                ("vs30", vs30.len()),
                ("hypo_depth", hypo_depth.len()),
            ]),
        }?;
        Ok((0..n).map(move |i| match self {
            Self::Interface { mag, rrup, vs30 } => P20Row::Interface {
                mag: mag[i],
                rrup: rrup[i],
                vs30: vs30[i],
            },
            Self::Slab {
                mag,
                rrup,
                vs30,
                hypo_depth,
            } => P20Row::Slab {
                mag: mag[i],
                rrup: rrup[i],
                vs30: vs30[i],
                hypo_depth: hypo_depth[i],
            },
        }))
    }
}

#[pymethods]
impl P20Inputs {
    /// Subduction interface inputs.
    #[staticmethod]
    fn interface(
        mag: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
    ) -> Self {
        Self::Interface {
            mag: mag.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
        }
    }

    /// Subduction slab inputs.
    #[staticmethod]
    fn slab(
        mag: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        hypo_depth: PyReadonlyArray1<f64>,
    ) -> Self {
        Self::Slab {
            mag: mag.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
            hypo_depth: hypo_depth.as_array().to_vec(),
        }
    }
}

/// (c0, c1, a0, c4, c5, c6) for the given tectonic type: interface reads the
/// unsuffixed columns, slab reads the `*slab` columns (`SUFFIX` in the Python
/// source).
fn select(row: &P20Row, c: &Coeffs) -> (f64, f64, f64, f64, f64, f64) {
    match row {
        P20Row::Interface { .. } => (c.c0, c.c1, c.a0, c.c4, c.c5, c.c6),
        P20Row::Slab { .. } => (c.c0slab, c.c1slab, c.a0slab, c.c4slab, c.c5slab, c.c6slab),
    }
}

/// Magnitude scaling FM, Eq. 5.
fn magnitude_scaling(c6: f64, c4: f64, c5: f64, mag: f64, m_b: f64) -> f64 {
    let m_diff = mag - m_b;
    if m_diff > 0.0 {
        c6 * m_diff
    } else {
        c4 * m_diff + c5 * m_diff.powi(2)
    }
}

/// Path term FP, Eq. 2, with the near-source saturation term h from Eq. 4a
/// (interface) and Eq. 4b (slab), and R and Rref from Eqs. 3a and 3b.
fn path_term(row: &P20Row, c1: f64, a0: f64, mag: f64, rrup: f64, m_b: f64) -> f64 {
    let h = match row {
        P20Row::Interface { .. } => 10f64.powf(-0.82 + 0.252 * mag),
        P20Row::Slab { .. } => {
            let slope = (35f64.log10() - 3.12f64.log10()) / (m_b - 4.0);
            35.0 * 10f64.powf(slope * (mag - m_b).min(0.0))
        }
    };
    let r = rrup.hypot(h);
    let r_rref = (r / 1f64.hypot(h)).ln();
    c1 * r.ln() + (B4 * mag) * r_rref + a0 * r
}

/// Depth scaling FD, Eq. 6, with db1 = 20 km (subduction slab only; always 0
/// for interface).
fn depth_scaling(hypo_depth: f64, d: f64, m: f64, db: f64) -> f64 {
    if hypo_depth <= 20.0 {
        m * (20.0 - db) + d
    } else if hypo_depth >= db {
        d
    } else {
        m * (hypo_depth - db) + d
    }
}

/// Mean + standard deviations for `ParkerEtAl2020SInter` /
/// `ParkerEtAl2020SSlab` (default constructor), one value per rupture-site
/// pair in `inputs`. The tectonic type is set by how `inputs` was built
/// (`P20Inputs.interface` or `P20Inputs.slab`). SA periods outside the table
/// are an error. Exposed to Python as `gm_models._core.p20`.
#[pyfunction]
pub fn p20(py: Python<'_>, inputs: PyRef<'_, P20Inputs>, im: IM) -> PyResult<GMMResult> {
    let coeffs = Coeffs::for_im(im)?;
    let inputs = &*inputs;
    // Releases the GIL: `inputs` owns copies of the arrays, so no Python object
    // is touched while other threads run.
    Ok(py.detach(|| {
        Ok::<_, GMMError>(
            inputs
                .iter()?
                .map(|row| compute(im, &coeffs, &row))
                .collect::<GMMResult>(),
        )
    })?)
}

/// Computes mean + standard deviations for a single site/rupture pair.
///
/// `c` must be `Coeffs::for_im(im)`; it is passed in so callers looping over
/// sites look it up once rather than per site.
pub fn compute(im: IM, c: &Coeffs, row: &P20Row) -> MeanStd {
    let c_pga = &PGA_COEFFS;

    let (mag, rrup, vs30, m_b) = match *row {
        P20Row::Interface { mag, rrup, vs30 } => (mag, rrup, vs30, MB_INTERFACE),
        P20Row::Slab {
            mag, rrup, vs30, ..
        } => (mag, rrup, vs30, MB_SLAB),
    };

    let (c0, c1, a0, c4, c5, c6) = select(row, c);
    let (c0_pga, c1_pga, a0_pga, c4_pga, c5_pga, c6_pga) = select(row, c_pga);

    let fm = magnitude_scaling(c6, c4, c5, mag, m_b);
    let fm_pga = magnitude_scaling(c6_pga, c4_pga, c5_pga, mag, m_b);
    let fp = path_term(row, c1, a0, mag, rrup, m_b);
    let fp_pga = path_term(row, c1_pga, a0_pga, mag, rrup, m_b);

    let (fd, fd_pga) = match *row {
        P20Row::Slab { hypo_depth, .. } => (
            depth_scaling(hypo_depth, c.d, c.m, c.db),
            depth_scaling(hypo_depth, c_pga.d, c_pga.m, c_pga.db),
        ),
        P20Row::Interface { .. } => (0.0, 0.0),
    };

    // Linear site term Flin, Eq. 8. With `region=None`, OpenQuake's `s1 == s2`,
    // which collapses the three-branch piecewise form into a single clamp.
    let flin = c.s2 * (vs30.min(c.v2) / VREF).ln();

    // Non-linear site term Fnl, Eq. 9 (with f1 = 0), driven by the reference-rock
    // PGA from Eq. 1 with PGA coefficients and FS = 0. OpenQuake switches it off
    // for `imt.period >= 3`; `PGV().period` is 0.0, so PGV keeps it.
    let fnl = match im {
        IM::SA { period } if period >= 3.0 => 0.0,
        _ => {
            let pgar = (fp_pga + fm_pga + c0_pga + fd_pga).exp();
            let f2 =
                c.f4 * ((c.f5 * (vs30.min(VREF_FNL) - VB)).exp() - (c.f5 * (VREF_FNL - VB)).exp());
            f2 * ((pgar + F3) / F3).ln()
        }
    };

    // Eq. 1, with FS = Flin + Fnl from Eq. 7. The basin term (Fb) is 0 here.
    let mean = fp + fnl + flin + fm + c0 + fd;

    // Standard deviations: phi^2(Rrup) from Eq. 19 plus DeltaVar(Vs30) from
    // Eq. 20, then phi (Eq. 18) and total sigma (Eq. 17). DeltaVar weights are
    // 1 at vs30 <= V1 / rrup <= R1 and 0 at vs30 >= V2 / rrup >= R2.
    let w_v = (V2 / vs30.clamp(V1, V2)).ln() / (V2 / V1).ln();
    let w_r = (R2 / rrup.clamp(R1, R2)).ln() / (R2 / R1).ln();
    let phi2_rrup = c.phi21 + (c.phi22 - c.phi21) * (rrup.clamp(R1, R2) / R1).ln() / (R2 / R1).ln();
    let phi = (phi2_rrup + c.phi2v * w_v * w_r).sqrt();

    MeanStd {
        mean,
        total: c.tau.hypot(phi),
        inter: c.tau,
        intra: phi,
    }
}

/// Layer 1 verification against OpenQuake's reference tables (generated from
/// Grace Parker's R implementation, see `parker_2020_test.py`'s test data),
/// read from the `external/oq-engine` submodule. `#[ignore]`d so a plain
/// `cargo test` works without the submodule; run them with
/// `cargo test -- --ignored`. The reference tables only have standard
/// deviations for the interface model.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    /// Slab CSVs are the only ones with `rup_hypo_depth`.
    fn row(rec: &HashMap<&str, &str>) -> P20Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        let (mag, rrup, vs30) = (num("rup_mag"), num("dist_rrup"), num("site_vs30"));
        match rec.get("rup_hypo_depth") {
            Some(_) => P20Row::Slab {
                mag,
                rrup,
                vs30,
                hypo_depth: num("rup_hypo_depth"),
            },
            None => P20Row::Interface { mag, rrup, vs30 },
        }
    }

    fn check(file: &str, pick: fn(&MeanStd) -> f64) {
        check_csv(&format!("PARKER20/{file}"), row, |im, row| {
            pick(&compute(im, &Coeffs::for_im(im).unwrap(), row))
        });
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake() {
        for tect in ["SInter", "SSlab"] {
            check(&format!("ParkerEtAl2020{tect}_MEAN.csv"), |r| r.mean.exp());
        }
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn stddevs_match_openquake() {
        check("ParkerEtAl2020SInter_TOTAL_STDDEV.csv", |r| r.total);
        check("ParkerEtAl2020SInter_INTER_EVENT_STDDEV.csv", |r| r.inter);
        check("ParkerEtAl2020SInter_INTRA_EVENT_STDDEV.csv", |r| r.intra);
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

    fn row() -> impl Strategy<Value = P20Row> {
        let common = (5.0..9.5f64, 0.1..1000.0f64, 100.0..2000.0f64);
        prop_oneof![
            common
                .clone()
                .prop_map(|(mag, rrup, vs30)| P20Row::Interface { mag, rrup, vs30 }),
            (common, 0.0..200.0f64).prop_map(|((mag, rrup, vs30), hypo_depth)| P20Row::Slab {
                mag,
                rrup,
                vs30,
                hypo_depth,
            }),
        ]
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(im in im(), row in row()) {
            let r = compute(im, &Coeffs::for_im(im).unwrap(), &row);
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }
    }

    #[test]
    fn inputs_reject_mismatched_lengths() {
        let slab = |hypo_depth: Vec<f64>| P20Inputs::Slab {
            mag: vec![7.0, 8.0],
            rrup: vec![10.0, 20.0],
            vs30: vec![400.0, 500.0],
            hypo_depth,
        };
        assert!(slab(vec![30.0]).iter().is_err());
        assert_eq!(slab(vec![30.0, 40.0]).iter().unwrap().count(), 2);
    }
}
